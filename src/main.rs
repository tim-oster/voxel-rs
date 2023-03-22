extern crate gl;
#[macro_use]
extern crate memoffset;

use std::f32::consts::PI;
use std::ops::Add;
use std::os::raw::c_int;
use std::time::{Duration, Instant};

use cgmath;
use cgmath::{EuclideanSpace, Point3, Vector2, Vector3};
use cgmath::{ElementWise, InnerSpace};
use imgui::{Condition, Id, TreeNodeFlags, Window};

use crate::chunk::BlockId;
use crate::graphics::framebuffer::Framebuffer;
use crate::graphics::resource::Resource;
use crate::graphics::screen_quad::ScreenQuad;
use crate::graphics::svo;
use crate::graphics::svo::{ContentRegistry, Material, RenderParams};
use crate::graphics::svo_picker::{AABB, PickerBatch};
use crate::systems::{chunkloading, storage, worldgen, worldsvo};
use crate::systems::chunkloading::ChunkEvent;
use crate::systems::gameplay::blocks;
use crate::systems::jobs::JobSystem;
use crate::world::chunk;
use crate::world::chunk::ChunkPos;
use crate::world::generator::{Noise, SplinePoint};
use crate::world::octree::{Octree, Position};

mod core;
mod graphics;
mod systems;
mod world;

fn main() {
    run(false);
}

#[cfg(test)]
mod test {
    use image::GenericImageView;

    // source: https://rosettacode.org/wiki/Percentage_difference_between_images#Rust
    fn diff_rgba3(rgba1: image::Rgba<u8>, rgba2: image::Rgba<u8>) -> i32 {
        (rgba1[0] as i32 - rgba2[0] as i32).abs()
            + (rgba1[1] as i32 - rgba2[1] as i32).abs()
            + (rgba1[2] as i32 - rgba2[2] as i32).abs()
    }

    #[test]
    fn was_something_broken_during_refactoring() {
        // keep window in scope to not drop opengl context prematurely
        let (fb, _window) = super::run(true);

        let pixels = fb.read_pixels();
        let actual = image::RgbaImage::from_raw(fb.width() as u32, fb.height() as u32, pixels).unwrap();
        let actual = image::DynamicImage::ImageRgba8(actual).flipv();
        actual.save_with_format("./test_actual.png", image::ImageFormat::Png).unwrap();

        let expected = image::open("./test_expected.png").unwrap();

        let mut accum = 0;
        let zipper = actual.pixels().zip(expected.pixels());
        for (pixel1, pixel2) in zipper {
            accum += diff_rgba3(pixel1.2, pixel2.2);
        }
        let diff_percent = accum as f64 / (255.0 * 3.0 * (actual.width() * actual.height()) as f64);
        println!("difference: {:.5}", diff_percent);
        assert!(diff_percent < 0.001);
    }
}

fn run(testing_mode: bool) -> (Framebuffer, core::Window) {
    let msaa_samples = 0;
    let mut window = core::Window::new(core::Config {
        width: 1024,
        height: 768,
        title: "voxel engine",
        msaa_samples,
        headless: false,
    });
    window.request_grab_cursor(false);

    // NEW CONFIG START

    let jobs = JobSystem::new(num_cpus::get() - 1);

    let render_distance = 15;
    let mut chunk_loader = chunkloading::ChunkLoader::new(render_distance as u32, 0, 8);
    let mut storage = storage::Storage::new();

    let mut world_cfg = world::generator::Config {
        sea_level: 70,
        continentalness: Noise {
            frequency: 0.001,
            octaves: 3,
            spline_points: vec![
                SplinePoint { x: -1.0, y: 20.0 },
                SplinePoint { x: 0.4, y: 50.0 },
                SplinePoint { x: 0.6, y: 70.0 },
                SplinePoint { x: 0.8, y: 120.0 },
                SplinePoint { x: 0.9, y: 190.0 },
                SplinePoint { x: 1.0, y: 200.0 },
            ],
        },
        erosion: Noise {
            frequency: 0.01,
            octaves: 4,
            spline_points: vec![
                SplinePoint { x: -1.0, y: -10.0 },
                SplinePoint { x: 1.0, y: 4.0 },
            ],
        },
    };
    let mut world_generator = worldgen::Generator::new(jobs.new_handle(), 1, world_cfg.clone());
    let mut world = systems::world::World::new();

    let mut registry = ContentRegistry::new();
    registry
        .add_texture("dirt", "assets/textures/dirt.png")
        .add_texture("dirt_normal", "assets/textures/dirt_n.png")
        .add_texture("grass_side", "assets/textures/grass_side.png")
        .add_texture("grass_side_normal", "assets/textures/grass_side_n.png")
        .add_texture("grass_top", "assets/textures/grass_top.png")
        .add_texture("grass_top_normal", "assets/textures/grass_top_n.png")
        .add_texture("stone", "assets/textures/stone.png")
        .add_texture("stone_normal", "assets/textures/stone_n.png")
        .add_texture("stone_bricks", "assets/textures/stone_bricks.png")
        .add_texture("stone_bricks_normal", "assets/textures/stone_bricks_n.png")
        .add_texture("glass", "assets/textures/glass.png");
    registry
        .add_material(blocks::AIR, Material::new())
        .add_material(blocks::GRASS, Material::new().specular(14.0, 0.4).top("grass_top").side("grass_side").bottom("dirt").with_normals())
        .add_material(blocks::DIRT, Material::new().specular(14.0, 0.4).all_sides("dirt").with_normals())
        .add_material(blocks::STONE, Material::new().specular(70.0, 0.4).all_sides("stone").with_normals())
        .add_material(blocks::STONE_BRICKS, Material::new().specular(70.0, 0.4).all_sides("stone_bricks").with_normals())
        .add_material(blocks::GLASS, Material::new().specular(70.0, 0.4).all_sides("glass"));

    let mut world_svo = svo::Svo::new(registry);
    let mut world_svo_mgr = worldsvo::Manager::new(jobs.new_handle(), render_distance as u32);

    // NEW CONFIG END

    let mut crosshair_shader = Resource::new(
        || graphics::ShaderProgramBuilder::new().load_shader_bundle("assets/shaders/crosshair.glsl")?.build()
    ).unwrap();

    let mut absolute_position = Point3::new(-24.0, 80.0, 174.0); // TODO: 16.0, 80.0, 16.0

    let screen_quad = ScreenQuad::new();

    let mut camera = graphics::Camera::new(72.0, window.get_aspect(), 0.01, 1024.0);
    camera.position = absolute_position;

    let mut cam_speed = 1f32;
    let cam_rot_speed = 0.005f32;
    let mut cam_rot = Vector3::new(0.0, -90f32.to_radians(), 0.0);
    let mut fly_mode = true;
    let mut slow_mode = false;

    let ambient_intensity = 0.3f32;
    let mut light_dir = Vector3::new(-1.0, -1.0, -1.0).normalize();

    let mut use_mouse_input = true;

    unsafe {
        gl::Enable(gl::CULL_FACE);
        gl::CullFace(gl::BACK);
        gl::FrontFace(gl::CCW);

        if msaa_samples > 0 {
            gl::Enable(gl::MULTISAMPLE);
            gl::Enable(gl::SAMPLE_SHADING);
            gl::MinSampleShading(1.0);
        }
    }

    let (w, h) = window.get_size();
    let mut ui_view = cgmath::ortho(0.0, w as f32, h as f32, 0.0, -1.0, 1.0);

    let mut selected_block: BlockId = 1;

    let mut vertical_velocity = 0.0f32;
    let mut pre_jump_velocity = Vector3::new(0.0, 0.0, 0.0);
    let mut is_jumping = false;
    let mut was_grounded = false;

    let (w, h) = window.get_size();
    let fb = Framebuffer::new(w, h);
    let start_time = Instant::now();

    window.request_grab_cursor(!testing_mode);
    while !window.should_close() {
        {
            let pos = absolute_position;
            let mut current_chunk_pos = ChunkPos::from_block_pos(pos.x as i32, pos.y as i32, pos.z as i32);
            current_chunk_pos.y = 0;

            let chunk_world_pos = Point3::new(current_chunk_pos.x as f32, current_chunk_pos.y as f32, current_chunk_pos.z as f32) * 32.0;
            let delta = pos - chunk_world_pos;

            camera.position.y = pos.y;
            camera.position.x = render_distance as f32 * 32.0 + delta.x;
            camera.position.z = render_distance as f32 * 32.0 + delta.z;

            let mut generate_count = 0;
            let chunk_events = chunk_loader.update(absolute_position);
            for event in &chunk_events {
                match event {
                    ChunkEvent::Load { pos, lod } => {
                        let result = storage.load(pos);
                        if result.is_ok() {
                            let mut chunk = result.ok().unwrap();
                            chunk.lod = *lod;
                            world.set_chunk(chunk);
                            continue;
                        }

                        let err = result.err().unwrap();
                        match err {
                            storage::LoadError::NotFound => {
                                let mut chunk = storage.new_chunk(*pos);
                                chunk.lod = *lod;
                                world_generator.enqueue_chunk(chunk);
                                generate_count += 1;
                            }
                        }
                    }
                    ChunkEvent::Unload { pos } => {
                        world_generator.dequeue_chunk(pos);
                        world.remove_chunk(pos);
                    }
                    ChunkEvent::LodChange { pos, lod } => {
                        if let Some(chunk) = world.get_chunk_mut(pos) {
                            chunk.lod = *lod;
                        }
                    }
                }
            }
            if !chunk_events.is_empty() {
                println!("generate {} new chunks", generate_count);
            }
            for chunk in world_generator.get_generated_chunks(40) {
                if chunk_loader.is_loaded(&chunk.pos) {
                    world.set_chunk(chunk);
                }
            }
            for pos in world.get_changed_chunks() {
                if let Some(chunk) = world.get_chunk(&pos) {
                    world_svo_mgr.set_chunk(chunk);
                } else {
                    world_svo_mgr.remove_chunk(&pos);
                }
            }

            world_svo_mgr.update(&current_chunk_pos, &mut world_svo);
        }

        // picker logic
        let aabb = AABB::new(camera.position, -Vector3::new(0.4, 1.7, 0.4), Vector3::new(0.8, 1.8, 0.8));

        let mut batch = PickerBatch::new();
        let block_result = batch.ray(camera.position, camera.forward, 30.0);
        let aabb_result = batch.aabb(&aabb);
        world_svo.raycast(batch);

        let block_result = block_result.get();
        let aabb_result = aabb_result.get();

        window.update(|frame| {
            let mut selected_block_world = None;
            let mut selected_block_svo = None;

            if block_result.dst != -1.0 {
                let rel_chunk_pos = ChunkPos::from_block_pos(block_result.pos.x as i32, block_result.pos.y as i32, block_result.pos.z as i32);
                let rel_block_pos = Point3::new((block_result.pos.x as i32 & 31) as f32, (block_result.pos.y as i32 & 31) as f32, (block_result.pos.z as i32 & 31) as f32);

                let delta = Point3::new(rel_chunk_pos.x - render_distance, 0, rel_chunk_pos.z - render_distance);
                let abs_chunk_pos = ChunkPos::from_block_pos(absolute_position.x as i32, 0, absolute_position.z as i32);
                let abs_chunk_pos = Point3::new((abs_chunk_pos.x + delta.x) as f32, rel_chunk_pos.y as f32, (abs_chunk_pos.z + delta.z) as f32);

                selected_block_world = Some((abs_chunk_pos * 32.0) + rel_block_pos.to_vec());
                selected_block_svo = Some(block_result.pos.0);
            }

            Window::new("Debug")
                .size([300.0, 100.0], Condition::FirstUseEver)
                .build(&frame.ui, || {
                    if testing_mode {
                        frame.ui.text_colored([1.0, 0.0, 0.0, 1.0], format!(
                            "TESTING MODE: {}",
                            jobs.len(),
                        ));
                    }

                    frame.ui.text(format!(
                        "fps: {}, frame: {:.2}ms, update: {:.2}ms",
                        frame.stats.frames_per_second,
                        frame.stats.avg_frame_time_per_second * 1000.0,
                        frame.stats.avg_update_time_per_second * 1000.0,
                    ));
                    frame.ui.text(format!(
                        "abs pos: ({:.3},{:.3},{:.3})",
                        absolute_position.x, absolute_position.y, absolute_position.z,
                    ));
                    frame.ui.text(format!(
                        "cam pos: ({:.3},{:.3},{:.3})",
                        camera.position.x, camera.position.y, camera.position.z,
                    ));
                    frame.ui.text(format!(
                        "cam fwd: ({:.3},{:.3},{:.3})",
                        camera.forward.x, camera.forward.y, camera.forward.z,
                    ));

                    let block_pos = selected_block_world.unwrap_or(Point3::new(0.0, 0.0, 0.0));
                    frame.ui.text(format!(
                        "block pos: ({:.2},{:.2},{:.2})",
                        block_pos.x, block_pos.y, block_pos.z,
                    ));

                    let block_normal = block_result.normal;
                    frame.ui.text(format!(
                        "block normal: ({},{},{})",
                        block_normal.x as i32, block_normal.y as i32, block_normal.z as i32,
                    ));

                    let svo_stats = world_svo.get_stats();
                    frame.ui.text(format!(
                        "svo size: {:.3}mb, depth: {}",
                        svo_stats.size_bytes, svo_stats.depth,
                    ));

                    frame.ui.text(format!(
                        "queue length: {}",
                        jobs.len(),
                    ));

                    let mem_stats = storage.get_memory_stats();
                    frame.ui.text(format!(
                        "chunk allocs: {}, total: {}",
                        mem_stats.in_use, mem_stats.allocated,
                    ));
                });

            Window::new("World Gen")
                .size([300.0, 100.0], Condition::FirstUseEver)
                .build(&frame.ui, || {
                    frame.ui.input_int("sea level", &mut world_cfg.sea_level).build();

                    // TODO rewrite
                    // if frame.ui.button("generate") {
                    //     jobs.borrow().clear();
                    //
                    //     // last_chunk_pos = ChunkPos::new(-9999, 0, 0);
                    //     svo_octant_ids.clear();
                    //     currently_generating_chunks.clear();
                    //     did_cam_repos = false;
                    //
                    //     world = systems::world::World::new();
                    //     svo.lock().unwrap().clear();
                    //     fly_mode = true;
                    // }

                    frame.ui.new_line();

                    let display_noise = |label: &str, noise: &mut Noise| {
                        if !frame.ui.collapsing_header(label, TreeNodeFlags::DEFAULT_OPEN) {
                            return;
                        }

                        let stack = frame.ui.push_id(Id::Str(label));

                        frame.ui.input_float("frequency", &mut noise.frequency).step(0.01).build();
                        frame.ui.input_int("octaves", &mut noise.octaves).build();

                        frame.ui.new_line();

                        frame.ui.text("spline points");
                        frame.ui.same_line();
                        if frame.ui.small_button("add") {
                            noise.spline_points.push(SplinePoint { x: 0.0, y: 0.0 });
                        }

                        frame.ui.indent();

                        let mut i = 0;
                        while i < noise.spline_points.len() {
                            let stack = frame.ui.push_id(Id::Int(i as i32));

                            frame.ui.text(format!("#{}", i));
                            frame.ui.same_line();
                            if frame.ui.small_button("del") {
                                noise.spline_points.remove(i);
                                i -= 1;
                                continue;
                            }

                            if i > 0 {
                                frame.ui.same_line();
                                if frame.ui.small_button("up") {
                                    noise.spline_points.swap(i, i - 1);
                                }
                            }

                            if i < noise.spline_points.len() - 1 {
                                frame.ui.same_line();
                                if frame.ui.small_button("down") {
                                    noise.spline_points.swap(i, i + 1);
                                }
                            }

                            let mut sp = &mut noise.spline_points[i];
                            let mut values: [f32; 2] = [sp.x, sp.y];
                            frame.ui.input_float2("x, y", &mut values).build();
                            sp.x = values[0];
                            sp.y = values[1];

                            stack.end();

                            i += 1;
                        }

                        frame.ui.unindent();

                        stack.end();
                    };
                    display_noise("continentalness", &mut world_cfg.continentalness);
                    display_noise("erosion", &mut world_cfg.erosion);
                });

            if frame.was_resized {
                camera.update_projection(72.0, frame.get_aspect(), 0.01, 1024.0);

                let (w, h) = frame.size;
                ui_view = cgmath::ortho(0.0, w as f32, h as f32, 0.0, -1.0, 1.0);
            }
            if frame.input.was_key_pressed(&glfw::Key::Escape) {
                frame.request_close();
            }

            const PHYSICS_EPSILON: f32 = 0.005;
            let apply_horizontal_physics = |speed: Vector3<f32>| -> Vector3<f32> {
                let x_dot = speed.dot(Vector3::new(1.0, 0.0, 0.0));
                let z_dot = speed.dot(Vector3::new(0.0, 0.0, 1.0));
                let x_dst = if x_dot > 0.0 { aabb_result.x_pos } else { aabb_result.x_neg };
                let z_dst = if z_dot > 0.0 { aabb_result.z_pos } else { aabb_result.z_neg };

                let mut speed = speed;

                if x_dst == 0.0 {
                    if x_dot > 0.0 {
                        let actual = aabb.pos.x + aabb.offset.x + aabb.extents.x;
                        let expected = actual.floor() - 2.0 * PHYSICS_EPSILON;
                        speed.x = expected - actual;
                    } else {
                        let actual = aabb.pos.x + aabb.offset.x;
                        let expected = actual.ceil() + 2.0 * PHYSICS_EPSILON;
                        speed.x = expected - actual;
                    }
                } else if x_dst != -1.0 && speed.x.abs() > (x_dst - PHYSICS_EPSILON) {
                    speed.x = (x_dst - 2.0 * PHYSICS_EPSILON) * speed.x.signum();
                }

                if z_dst == 0.0 {
                    if z_dot > 0.0 {
                        let actual = aabb.pos.z + aabb.offset.z + aabb.extents.z;
                        let expected = actual.floor() - 2.0 * PHYSICS_EPSILON;
                        speed.x = expected - actual;
                    } else {
                        let actual = aabb.pos.z + aabb.offset.z;
                        let expected = actual.ceil() + 2.0 * PHYSICS_EPSILON;
                        speed.z = expected - actual;
                    }
                } else if z_dst != -1.0 && speed.z.abs() > (z_dst - PHYSICS_EPSILON) {
                    speed.z = (z_dst - 2.0 * PHYSICS_EPSILON) * speed.z.signum();
                }

                speed
            };
            let apply_vertical_physics = |speed: Vector3<f32>| -> Vector3<f32> {
                let y_dot = speed.dot(Vector3::new(0.0, 1.0, 0.0));
                let y_dst = if y_dot > 0.0 { aabb_result.y_pos } else { aabb_result.y_neg };

                let mut speed = speed;
                if y_dst == 0.0 {
                    if y_dot > 0.0 {
                        let actual = aabb.pos.y + aabb.offset.y + aabb.extents.y;
                        let expected = actual.floor() - 2.0 * PHYSICS_EPSILON;
                        speed.y = expected - actual;
                    } else {
                        let actual = aabb.pos.y + aabb.offset.y;
                        let expected = actual.ceil() + 2.0 * PHYSICS_EPSILON;
                        speed.y = expected - actual;
                    }
                } else if y_dst != -1.0 && speed.y.abs() > (y_dst - PHYSICS_EPSILON) {
                    speed.y = (y_dst - 2.0 * PHYSICS_EPSILON) * speed.y.signum();
                }
                speed
            };

            if !testing_mode {
                let mut horizontal_speed = Vector3::new(0.0, 0.0, 0.0);
                if frame.input.is_key_pressed(&glfw::Key::W) {
                    let dir = camera.forward.mul_element_wise(Vector3::new(1.0, 0.0, 1.0)).normalize();
                    let speed = dir * cam_speed * frame.stats.delta_time;
                    horizontal_speed += speed;
                }
                if frame.input.is_key_pressed(&glfw::Key::S) {
                    let dir = -camera.forward.mul_element_wise(Vector3::new(1.0, 0.0, 1.0)).normalize();
                    let speed = dir * cam_speed * frame.stats.delta_time;
                    horizontal_speed += speed;
                }
                if frame.input.is_key_pressed(&glfw::Key::A) {
                    let speed = -camera.right() * cam_speed * frame.stats.delta_time;
                    horizontal_speed += speed;
                }
                if frame.input.is_key_pressed(&glfw::Key::D) {
                    let speed = camera.right() * cam_speed * frame.stats.delta_time;
                    horizontal_speed += speed;
                }

                // clamp horizontal speed
                if !fly_mode {
                    horizontal_speed = apply_horizontal_physics(horizontal_speed);
                }
                let speed = horizontal_speed.magnitude();
                let max_speed = cam_speed * frame.stats.delta_time;
                if speed > max_speed {
                    horizontal_speed = horizontal_speed.normalize() * max_speed;
                }

                let mut vertical_speed = Vector3::new(0.0, 0.0, 0.0);
                if !fly_mode {
                    const MAX_FALL_VELOCITY: f32 = 2.0;
                    const ACCELERATION: f32 = 0.008;

                    let is_grounded = aabb_result.y_neg < 0.02 && aabb_result.y_neg != -1.0;
                    if is_grounded {
                        vertical_velocity = 0.0;
                        cam_speed = 0.15;
                        is_jumping = false;
                        was_grounded = true;
                        pre_jump_velocity = Vector3::new(0.0, 0.0, 0.0);

                        if frame.input.is_key_pressed(&glfw::Key::LeftShift) {
                            cam_speed = 0.22;
                        }
                        if frame.input.is_key_pressed(&glfw::Key::Space) {
                            vertical_velocity = 0.2;
                            is_jumping = true;
                            pre_jump_velocity = horizontal_speed;
                        }
                    } else {
                        cam_speed = 0.1;
                        vertical_velocity -= ACCELERATION;
                        vertical_velocity = vertical_velocity.clamp(-MAX_FALL_VELOCITY, MAX_FALL_VELOCITY);

                        horizontal_speed += pre_jump_velocity;
                    }

                    vertical_speed.y = vertical_velocity * frame.stats.delta_time;
                } else {
                    vertical_velocity = 0.0;
                    is_jumping = false;
                    was_grounded = false;
                    pre_jump_velocity = Vector3::new(0.0, 0.0, 0.0);

                    if frame.input.is_key_pressed(&glfw::Key::Space) {
                        let speed = cam_speed * frame.stats.delta_time;
                        let speed = Vector3::new(0.0, speed, 0.0);
                        vertical_speed += speed;
                    }
                    if frame.input.is_key_pressed(&glfw::Key::LeftShift) {
                        let speed = cam_speed * frame.stats.delta_time;
                        let speed = Vector3::new(0.0, -speed, 0.0);
                        vertical_speed += speed;
                    }
                }
                if !fly_mode {
                    vertical_speed = apply_vertical_physics(vertical_speed);
                }

                absolute_position += horizontal_speed;
                absolute_position += vertical_speed;

                if frame.input.was_key_pressed(&glfw::Key::F) {
                    fly_mode = !fly_mode;
                    cam_speed = if fly_mode { 1.0 } else { 0.15 };
                }
                if frame.input.was_key_pressed(&glfw::Key::G) {
                    slow_mode = !slow_mode;
                    cam_speed *= if slow_mode { 0.1 } else { 1.0 / 0.1 };
                }
                if frame.input.was_key_pressed(&glfw::Key::E) {
                    light_dir = camera.forward;
                }
                if frame.input.was_key_pressed(&glfw::Key::R) {
                    world_svo.reload_resources();
                    if let Err(e) = crosshair_shader.reload() {
                        println!("error reloading crosshair shader: {:?}", e);
                    }
                    println!("tried reloading all resources");
                }
                if frame.input.was_key_pressed(&glfw::Key::T) {
                    use_mouse_input = !use_mouse_input;
                    frame.request_grab_cursor(use_mouse_input);
                }

                let hot_bar = vec![blocks::GRASS, blocks::DIRT, blocks::STONE, blocks::STONE_BRICKS, blocks::GLASS];
                for i in 1..hot_bar.len() {
                    let key = glfw::Key::Num1 as c_int + (i - 1) as c_int;
                    let key = &key as *const c_int as *const glfw::Key;
                    let key = unsafe { &*key };

                    if frame.input.was_key_pressed(key) {
                        selected_block = i as u32;
                    }
                }

                if use_mouse_input {
                    let delta = frame.input.get_mouse_delta();
                    if delta.x.abs() > 0.01 {
                        cam_rot.y += delta.x * cam_rot_speed * frame.stats.delta_time;
                    }
                    if delta.y.abs() > 0.01 {
                        cam_rot.x -= delta.y * cam_rot_speed * frame.stats.delta_time;

                        let limit = PI / 2.0 - 0.01;
                        cam_rot.x = cam_rot.x.clamp(-limit, limit);
                    }
                    camera.set_forward_from_euler(cam_rot);
                }

                // removing blocks
                if frame.input.is_button_pressed_once(&glfw::MouseButton::Button1) {
                    if let Some(block_pos) = selected_block_world {
                        let x = block_pos.x as i32;
                        let y = block_pos.y as i32;
                        let z = block_pos.z as i32;
                        world.set_block(x, y, z, chunk::NO_BLOCK);
                    }
                }

                // block picking
                if frame.input.is_button_pressed_once(&glfw::MouseButton::Button3) {
                    if let Some(block_pos) = selected_block_world {
                        let x = block_pos.x as i32;
                        let y = block_pos.y as i32;
                        let z = block_pos.z as i32;
                        selected_block = world.get_block(x, y, z);
                    }
                }

                // adding blocks
                if frame.input.is_button_pressed_once(&glfw::MouseButton::Button2) {
                    if let Some(block_pos) = selected_block_world {
                        let block_normal = block_result.normal;
                        let block_pos = block_pos.add(block_normal.0);
                        let x = block_pos.x as i32 as f32;
                        let y = block_pos.y as i32 as f32;
                        let z = block_pos.z as i32 as f32;

                        let player_min_x = absolute_position.x + aabb.offset.x;
                        let player_min_y = absolute_position.y + aabb.offset.y - 0.1; // add offset to prevent physics glitches
                        let player_min_z = absolute_position.z + aabb.offset.z;
                        let player_max_x = absolute_position.x + aabb.extents.x;
                        let player_max_y = absolute_position.y + aabb.extents.y;
                        let player_max_z = absolute_position.z + aabb.extents.z;

                        if (player_max_x < x || player_min_x > x + 1.0) ||
                            (player_max_y < y || player_min_y > y + 1.0) ||
                            (player_max_z < z || player_min_z > z + 1.0) ||
                            fly_mode {
                            world.set_block(x as i32, y as i32, z as i32, selected_block);
                        }
                    }
                }
            }

            unsafe {
                if testing_mode { fb.bind(); }

                gl::ClearColor(0.0, 0.0, 0.0, 1.0);
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                world_svo.render(RenderParams {
                    ambient_intensity,
                    light_dir,
                    cam_pos: camera.position,
                    view_mat: camera.get_camera_to_world_matrix(),
                    fov_y_rad: 70.0f32.to_radians(),
                    aspect_ratio: frame.get_aspect(),
                    selected_block_svo_space: selected_block_svo,
                });

                if testing_mode {
                    fb.unbind();

                    gl::ClearColor(0.0, 0.0, 0.0, 1.0);
                    gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
                }

                // render ui
                crosshair_shader.bind();
                crosshair_shader.set_f32mat4("u_view", &ui_view);
                crosshair_shader.set_f32vec2("u_dimensions", &Vector2::new(frame.size.0 as f32, frame.size.1 as f32));
                gl::Enable(gl::BLEND);
                gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
                screen_quad.render();
                gl::Disable(gl::BLEND);
                crosshair_shader.unbind();

                gl_check_error!();
            }

            if testing_mode && start_time.elapsed() > Duration::from_secs(1) && jobs.len() == 0 {
                frame.request_close();
                return;
            }
        });
    }

    jobs.stop();

    (fb, window)
}
