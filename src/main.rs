extern crate gl;
#[macro_use]
extern crate memoffset;

use std::{mem, ptr};
use std::ffi::c_void;
use std::ops::Add;
use std::os::raw::c_int;
use std::time::Instant;

use cgmath;
use cgmath::{Point2, Point3, Vector2, Vector3};
use cgmath::{ElementWise, EuclideanSpace, InnerSpace};
use gl::types::*;
use imgui::{Condition, Window};

use crate::chunk::BlockId;
use crate::world::chunk;
use crate::world::octree::Position;

mod graphics;
mod core;
mod world;

unsafe fn gl_check_error_(file: &str, line: u32) -> u32 {
    let mut error_code = gl::GetError();
    while error_code != gl::NO_ERROR {
        let error = match error_code {
            gl::INVALID_ENUM => "INVALID_ENUM",
            gl::INVALID_VALUE => "INVALID_VALUE",
            gl::INVALID_OPERATION => "INVALID_OPERATION",
            gl::STACK_OVERFLOW => "STACK_OVERFLOW",
            gl::STACK_UNDERFLOW => "STACK_UNDERFLOW",
            gl::OUT_OF_MEMORY => "OUT_OF_MEMORY",
            gl::INVALID_FRAMEBUFFER_OPERATION => "INVALID_FRAMEBUFFER_OPERATION",
            _ => "unknown GL error code",
        };

        println!("{} | {} ({})", error, file, line);

        error_code = gl::GetError();
    }
    error_code
}

macro_rules! gl_check_error {
    () => {
        gl_check_error_(file!(), line!())
    };
}

fn main() {
    let mut window = core::Window::new(1024, 768, "voxel engine");
    window.request_grab_cursor(false);

    let mut world_shader = graphics::Resource::new(
        || graphics::ShaderProgramBuilder::new()
            .load_shader(graphics::ShaderType::Vertex, "assets/shaders/shader.vert")?
            .load_shader(graphics::ShaderType::Fragment, "assets/shaders/shader.frag")?
            .build()
    ).unwrap();

    let mut picker_shader = graphics::Resource::new(
        || graphics::ShaderProgramBuilder::new()
            .load_shader(graphics::ShaderType::Compute, "assets/shaders/picker.glsl")?
            .build()
    ).unwrap();

    let mut ui_shader = graphics::Resource::new(
        || graphics::ShaderProgramBuilder::new()
            .load_shader(graphics::ShaderType::Vertex, "assets/shaders/ui.vert")?
            .load_shader(graphics::ShaderType::Fragment, "assets/shaders/ui.frag")?
            .build()
    ).unwrap();

    let tex_array = graphics::Resource::new(
        || graphics::TextureArrayBuilder::new(4)
            .add_texture("dirt", "assets/textures/dirt.png")?
            .add_texture("dirt_normal", "assets/textures/dirt_n.png")?
            .add_texture("grass_side", "assets/textures/grass_side.png")?
            .add_texture("grass_side_normal", "assets/textures/grass_side_n.png")?
            .add_texture("grass_top", "assets/textures/grass_top.png")?
            .add_texture("grass_top_normal", "assets/textures/grass_top_n.png")?
            .add_texture("stone", "assets/textures/stone.png")?
            .add_texture("stone_normal", "assets/textures/stone_n.png")?
            .add_texture("stone_bricks", "assets/textures/stone_bricks.png")?
            .add_texture("stone_bricks_normal", "assets/textures/stone_bricks_n.png")?
            .add_texture("glass", "assets/textures/glass.png")?
            .add_texture("glass_light_blue", "assets/textures/glass_light_blue.png")?
            .build()
    ).unwrap();

    let (vao, indices_count) = build_vao();

    let mut camera = graphics::Camera::new(72.0, window.get_aspect(), 0.01, 1024.0);
    camera.position = Point3::new(128.0, 80.0, 128.0);

    let mut cam_speed = 1f32;
    let cam_rot_speed = 0.005f32;
    let mut cam_rot = cgmath::Vector3::new(0.0, -90f32.to_radians(), 0.0);

    let ambient_intensity = 0.3f32;
    let mut light_dir = Vector3::new(-1.0, -1.0, -1.0).normalize();

    let mut use_mouse_input = true;

    let mut world_ssbo = 0;
    let world_buffer;
    let world_buffer_size = 1000 * 1024 * 1024; // 1000 MB

    unsafe {
        gl::GenBuffers(1, &mut world_ssbo);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, world_ssbo);

        gl::BufferStorage(
            gl::SHADER_STORAGE_BUFFER,
            world_buffer_size,
            ptr::null(),
            gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT,
        );
        world_buffer = gl::MapBufferRange(
            gl::SHADER_STORAGE_BUFFER,
            0,
            world_buffer_size,
            gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT,
        ) as *mut u32;

        gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 0, world_ssbo);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
    }

    // WORLD LOADING START

    println!("loading model file");
    let start = Instant::now();
    let vox_data = dot_vox::load("assets/ignore/terrain.vox").unwrap();

    println!("{}s; converting into chunks", start.elapsed().as_secs_f32());
    let start = Instant::now();
    let mut world = storage::world::World::new();

    let square = 1;
    for x in 0..square {
        for z in 0..square {
            world.add_vox_at(&vox_data, x * 256, 0, z * 256);
        }
    }
    world.get_changed_chunks(); // drain all changed chunks

    println!("{}s; converting into svo", start.elapsed().as_secs_f32());
    let start = Instant::now();
    let mut svo = world.build_svo();

    println!("{}s; serializing svo", start.elapsed().as_secs_f32());
    let start = Instant::now();
    let mut svo_buffer = svo.serialize();

    println!("{}s; final size: {} MB", start.elapsed().as_secs_f32(), svo_buffer.buffer.bytes.len() as f32 * 4f32 / 1024f32 / 1024f32);

    let mut max_depth_exp2 = (-(svo_buffer.depth as f32)).exp2();
    unsafe {
        ptr::write(world_buffer, max_depth_exp2.to_bits());
        ptr::copy(svo_buffer.buffer.bytes.as_ptr(), world_buffer.offset(1), svo_buffer.buffer.bytes.len());
    }

    // WORLD LOADING END

    unsafe {
        // gl::Enable(gl::DEPTH_TEST);
        // gl::DepthFunc(gl::LESS);

        gl::Enable(gl::CULL_FACE);
        gl::CullFace(gl::BACK);
        gl::FrontFace(gl::CCW);
    }

    // BLOCK PICKING START

    #[repr(align(16))]
    struct AlignedVec3<T>(cgmath::Vector3<T>);
    #[repr(align(16))]
    struct AlignedPoint3<T>(cgmath::Point3<T>);

    #[repr(C)]
    struct PickerData {
        block_pos: AlignedPoint3<f32>,
        block_normal: AlignedVec3<f32>,
    }
    let picker_data;

    let mut picker_ssbo = 0;
    unsafe {
        gl::GenBuffers(1, &mut picker_ssbo);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, picker_ssbo);

        // create an immutable buffer and persistently map to it for stream reading
        // from GPU to CPU
        gl::BufferStorage(
            gl::SHADER_STORAGE_BUFFER,
            mem::size_of::<PickerData>() as GLsizeiptr,
            ptr::null(),
            gl::MAP_READ_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT,
        );
        picker_data = gl::MapBufferRange(
            gl::SHADER_STORAGE_BUFFER,
            0,
            mem::size_of::<PickerData>() as GLsizeiptr,
            gl::MAP_READ_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT,
        ) as *mut PickerData;

        gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 1, picker_ssbo);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
    }

    // BLOCK PICKING END

    // BLOCK PICKING START

    #[repr(C)]
    struct Material {
        specular_pow: f32,
        specular_strength: f32,
        tex_top: i32,
        tex_side: i32,
        tex_bottom: i32,
        tex_top_normal: i32,
        tex_side_normal: i32,
        tex_bottom_normal: i32,
    }

    let materials = vec![
        Material { // air
            specular_pow: 0.0,
            specular_strength: 0.0,
            tex_top: -1,
            tex_side: -1,
            tex_bottom: -1,
            tex_top_normal: -1,
            tex_side_normal: -1,
            tex_bottom_normal: -1,
        },
        Material { // grass
            specular_pow: 14.0,
            specular_strength: 0.4,
            tex_top: tex_array.lookup("grass_top").unwrap() as i32,
            tex_side: tex_array.lookup("grass_side").unwrap() as i32,
            tex_bottom: tex_array.lookup("dirt").unwrap() as i32,
            tex_top_normal: tex_array.lookup("grass_top_normal").unwrap() as i32,
            tex_side_normal: tex_array.lookup("grass_side_normal").unwrap() as i32,
            tex_bottom_normal: tex_array.lookup("dirt_normal").unwrap() as i32,
        },
        Material { // dirt
            specular_pow: 14.0,
            specular_strength: 0.4,
            tex_top: tex_array.lookup("dirt").unwrap() as i32,
            tex_side: tex_array.lookup("dirt").unwrap() as i32,
            tex_bottom: tex_array.lookup("dirt").unwrap() as i32,
            tex_top_normal: tex_array.lookup("dirt_normal").unwrap() as i32,
            tex_side_normal: tex_array.lookup("dirt_normal").unwrap() as i32,
            tex_bottom_normal: tex_array.lookup("dirt_normal").unwrap() as i32,
        },
        Material { // stone
            specular_pow: 70.0,
            specular_strength: 0.4,
            tex_top: tex_array.lookup("stone").unwrap() as i32,
            tex_side: tex_array.lookup("stone").unwrap() as i32,
            tex_bottom: tex_array.lookup("stone").unwrap() as i32,
            tex_top_normal: tex_array.lookup("stone_normal").unwrap() as i32,
            tex_side_normal: tex_array.lookup("stone_normal").unwrap() as i32,
            tex_bottom_normal: tex_array.lookup("stone_normal").unwrap() as i32,
        },
        Material { // stone bricks
            specular_pow: 70.0,
            specular_strength: 0.4,
            tex_top: tex_array.lookup("stone_bricks").unwrap() as i32,
            tex_side: tex_array.lookup("stone_bricks").unwrap() as i32,
            tex_bottom: tex_array.lookup("stone_bricks").unwrap() as i32,
            tex_top_normal: tex_array.lookup("stone_bricks_normal").unwrap() as i32,
            tex_side_normal: tex_array.lookup("stone_bricks_normal").unwrap() as i32,
            tex_bottom_normal: tex_array.lookup("stone_bricks_normal").unwrap() as i32,
        },
        Material { // glass
            specular_pow: 70.0,
            specular_strength: 0.4,
            tex_top: tex_array.lookup("glass").unwrap() as i32,
            tex_side: tex_array.lookup("glass").unwrap() as i32,
            tex_bottom: tex_array.lookup("glass").unwrap() as i32,
            tex_top_normal: -1,
            tex_side_normal: -1,
            tex_bottom_normal: -1,
        },
        Material { // glass light blue
            specular_pow: 70.0,
            specular_strength: 0.4,
            tex_top: tex_array.lookup("glass_light_blue").unwrap() as i32,
            tex_side: tex_array.lookup("glass_light_blue").unwrap() as i32,
            tex_bottom: tex_array.lookup("glass_light_blue").unwrap() as i32,
            tex_top_normal: -1,
            tex_side_normal: -1,
            tex_bottom_normal: -1,
        },
    ];

    let mut material_ssbo = 0;
    unsafe {
        gl::GenBuffers(1, &mut material_ssbo);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, material_ssbo);

        gl::BufferData(
            gl::SHADER_STORAGE_BUFFER,
            (mem::size_of::<Material>() * materials.len()) as GLsizeiptr,
            &materials[0] as *const Material as *const c_void,
            gl::STATIC_READ,
        );

        gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 2, material_ssbo);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
    }

    // BLOCK PICKING END

    let (w, h) = window.get_size();
    let mut ui_view = cgmath::ortho(0.0, w as f32, h as f32, 0.0, -1.0, 1.0);

    let mut selected_block: BlockId = 1;

    window.request_grab_cursor(true);
    while !window.should_close() {
        // TODO do this in the background
        {
            let changed = world.get_changed_chunks();
            if !changed.is_empty() {
                let start = Instant::now();
                for pos in &changed {
                    if let Some(chunk) = world.chunks.get(pos) {
                        let octree = chunk.build_octree();
                        svo.set(Position(pos.x as u32, pos.y as u32, pos.z as u32), Some(octree));
                    }
                }
                svo_buffer = svo.serialize_delta(svo_buffer);
                max_depth_exp2 = (-(svo_buffer.depth as f32)).exp2();

                unsafe {
                    ptr::write(world_buffer, max_depth_exp2.to_bits());
                }

                for id in &svo_buffer.changed_octants {
                    let range = svo_buffer.buffer.octant_to_range.get(id).unwrap();
                    unsafe {
                        ptr::copy(svo_buffer.buffer.bytes.as_ptr().offset(range.start as isize),
                                  world_buffer.offset(1 + range.start as isize),
                                  range.length);
                    }
                }

                println!("rebuild took {}ms", start.elapsed().as_millis());
            }
        }

        window.update(|frame| {
            Window::new("Debug")
                .size([300.0, 100.0], Condition::FirstUseEver)
                .build(&frame.ui, || {
                    frame.ui.text(format!(
                        "fps: {}, frame: {:.2}ms, update: {:.2}ms",
                        frame.stats.frames_per_second,
                        frame.stats.avg_frame_time_per_second * 1000.0,
                        frame.stats.avg_update_time_per_second * 1000.0,
                    ));
                    frame.ui.text(format!(
                        "cam pos: ({:.3},{:.3},{:.3})",
                        camera.position.x, camera.position.y, camera.position.z,
                    ));
                    frame.ui.text(format!(
                        "cam fwd: ({:.3},{:.3},{:.3})",
                        camera.forward.x, camera.forward.y, camera.forward.z,
                    ));

                    let mut block_pos = unsafe { (*picker_data).block_pos.0 };
                    if block_pos.x == f32::MAX {
                        block_pos = Point3::new(0.0, 0.0, 0.0);
                    }
                    frame.ui.text(format!(
                        "block pos: ({:.2},{:.2},{:.2})",
                        block_pos.x, block_pos.y, block_pos.z,
                    ));

                    let block_normal = unsafe { (*picker_data).block_normal.0 };
                    frame.ui.text(format!(
                        "block normal: ({},{},{})",
                        block_normal.x as i32, block_normal.y as i32, block_normal.z as i32,
                    ));
                });

            if frame.was_resized {
                camera.update_projection(72.0, frame.get_aspect(), 0.01, 1024.0);

                let (w, h) = frame.size;
                ui_view = cgmath::ortho(0.0, w as f32, h as f32, 0.0, -1.0, 1.0);
            }
            if frame.input.was_key_pressed(&glfw::Key::Escape) {
                frame.request_close();
            }
            if frame.input.is_key_pressed(&glfw::Key::W) {
                let dir = camera.forward.mul_element_wise(Vector3::new(1.0, 0.0, 1.0)).normalize();
                camera.position += dir * cam_speed * frame.stats.delta_time;
            }
            if frame.input.is_key_pressed(&glfw::Key::S) {
                let dir = camera.forward.mul_element_wise(Vector3::new(1.0, 0.0, 1.0)).normalize();
                camera.position -= dir * cam_speed * frame.stats.delta_time;
            }
            if frame.input.is_key_pressed(&glfw::Key::A) {
                camera.position -= camera.right() * cam_speed * frame.stats.delta_time;
            }
            if frame.input.is_key_pressed(&glfw::Key::D) {
                camera.position += camera.right() * cam_speed * frame.stats.delta_time;
            }
            if frame.input.is_key_pressed(&glfw::Key::Space) {
                camera.position.y += cam_speed * frame.stats.delta_time;
            }
            if frame.input.is_key_pressed(&glfw::Key::LeftShift) {
                camera.position.y -= cam_speed * frame.stats.delta_time;
            }
            if frame.input.was_key_pressed(&glfw::Key::G) {
                if cam_speed == 1.0 {
                    cam_speed = 0.25;
                } else {
                    cam_speed = 1.0;
                }
            }
            if frame.input.was_key_pressed(&glfw::Key::E) {
                light_dir = camera.forward;
            }
            if frame.input.was_key_pressed(&glfw::Key::R) {
                for shader in vec![
                    &mut world_shader,
                    &mut picker_shader,
                    &mut ui_shader,
                ] {
                    if let Err(err) = shader.reload() {
                        println!("error loading shader: {:?}", err);
                    } else {
                        println!("reload shader");
                    }
                }
            }
            if frame.input.was_key_pressed(&glfw::Key::T) {
                use_mouse_input = !use_mouse_input;
                frame.request_grab_cursor(use_mouse_input);
            }
            for i in 1..materials.len() {
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
                }
                camera.set_forward_from_euler(cam_rot);
            }

            // removing blocks
            if frame.input.is_button_pressed_once(&glfw::MouseButton::Button1) {
                let block_pos = unsafe { (*picker_data).block_pos.0 };
                if block_pos.x != f32::MAX {
                    let x = block_pos.x as i32;
                    let y = block_pos.y as i32;
                    let z = block_pos.z as i32;
                    world.set_block(x, y, z, chunk::NO_BLOCK);
                }
            }

            // block picking
            if frame.input.is_button_pressed_once(&glfw::MouseButton::Button3) {
                let block_pos = unsafe { (*picker_data).block_pos.0 };
                if block_pos.x != f32::MAX {
                    let x = block_pos.x as i32;
                    let y = block_pos.y as i32;
                    let z = block_pos.z as i32;
                    selected_block = world.get_block(x, y, z);
                }
            }

            // adding blocks
            if frame.input.is_button_pressed_once(&glfw::MouseButton::Button2) {
                let block_pos = unsafe { (*picker_data).block_pos.0 };
                let block_normal = unsafe { (*picker_data).block_normal.0 };
                if block_pos.x != f32::MAX {
                    let block_pos = block_pos.add(block_normal);
                    let x = block_pos.x as i32;
                    let y = block_pos.y as i32;
                    let z = block_pos.z as i32;
                    world.set_block(x, y, z, selected_block);
                }
            }

            unsafe {
                gl::ClearColor(0.0, 0.0, 0.0, 1.0);
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                gl::BindVertexArray(vao);

                // render world
                world_shader.bind();
                // shader.set_f32mat4("u_projection", camera.get_projection_matrix());
                // shader.set_f32mat4("u_view", &camera.get_view_matrix());
                world_shader.set_f32("u_ambient", ambient_intensity);
                world_shader.set_f32vec3("u_light_dir", &light_dir);
                world_shader.set_f32vec3("u_cam_pos", &camera.position.to_vec());

                world_shader.set_f32mat4("u_view", &camera.get_camera_to_world_matrix());
                world_shader.set_f32("u_fovy", 70.0f32.to_radians());
                world_shader.set_f32("u_aspect", frame.get_aspect());
                // shader.set_i32("u_max_depth", svo.max_depth);
                world_shader.set_f32vec3("u_highlight_pos", &(*picker_data).block_pos.0.to_vec());

                gl::ActiveTexture(gl::TEXTURE0);
                tex_array.bind();
                world_shader.set_i32("u_texture", 0);

                gl::DrawElements(gl::TRIANGLES, indices_count, gl::UNSIGNED_INT, ptr::null());
                world_shader.unbind();

                // render ui
                ui_shader.bind();
                ui_shader.set_f32mat4("u_view", &ui_view);
                ui_shader.set_f32vec2("u_dimensions", &Vector2::new(frame.size.0 as f32, frame.size.1 as f32));
                gl::Enable(gl::BLEND);
                gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
                gl::DrawElements(gl::TRIANGLES, indices_count, gl::UNSIGNED_INT, ptr::null());
                gl::Disable(gl::BLEND);
                ui_shader.unbind();

                gl::BindVertexArray(0);

                // picker logic
                picker_shader.bind();
                picker_shader.set_f32vec3("u_cam_pos", &camera.position.to_vec());
                picker_shader.set_f32vec3("u_cam_dir", &camera.forward);
                gl::DispatchCompute(1, 1, 1);
                picker_shader.unbind();

                gl_check_error!();
            }
        });
    }
}

fn build_vao() -> (GLuint, i32) {
    unsafe {
        #[repr(C)]
        struct Vertex {
            position: cgmath::Point3<f32>,
            uv: cgmath::Point2<f32>,
            normal: cgmath::Vector3<f32>,
        }

        // let pos = Point3::new(0.0, 0.0, -5.0);
        // let scl = 0.5;
        let vertices = vec![
            // screen quad
            Vertex { position: Point3::new(1.0, 1.0, -1.0), uv: Point2::new(1.0, 1.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            Vertex { position: Point3::new(-1.0, 1.0, -1.0), uv: Point2::new(0.0, 1.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            Vertex { position: Point3::new(-1.0, -1.0, -1.0), uv: Point2::new(0.0, 0.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            Vertex { position: Point3::new(1.0, -1.0, -1.0), uv: Point2::new(1.0, 0.0), normal: Vector3::new(0.0, 0.0, 1.0) },

            // // front
            // Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z + scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z + scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z + scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            // Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z + scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            // // back
            // Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z - scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(0.0, 0.0, -1.0) },
            // Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z - scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(0.0, 0.0, -1.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z - scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(0.0, 0.0, -1.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z - scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(0.0, 0.0, -1.0) },
            // // left
            // Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z + scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(-1.0, 0.0, 0.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z - scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(-1.0, 0.0, 0.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z - scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(-1.0, 0.0, 0.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z + scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(-1.0, 0.0, 0.0) },
            // // right
            // Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z + scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(1.0, 0.0, 0.0) },
            // Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z + scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(1.0, 0.0, 0.0) },
            // Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z - scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(1.0, 0.0, 0.0) },
            // Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z - scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(1.0, 0.0, 0.0) },
            // // top
            // Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z + scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(0.0, 1.0, 0.0) },
            // Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z - scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(0.0, 1.0, 0.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z - scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(0.0, 1.0, 0.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z + scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(0.0, 1.0, 0.0) },
            // // bottom
            // Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z + scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(0.0, -1.0, 0.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z + scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(0.0, -1.0, 0.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z - scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(0.0, -1.0, 0.0) },
            // Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z - scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(0.0, -1.0, 0.0) },
        ];

        let mut indices = Vec::<i32>::new();
        for i in 0..(vertices.len() / 4) {
            let mut base = vec![0i32, 1, 3, 1, 2, 3];
            for elem in base.iter_mut() {
                *elem += (i as i32) * 4;
            }
            indices.extend(&base);
        }

        let (mut vbo, mut vao, mut ebo) = (0, 0, 0);
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);
        gl::GenBuffers(1, &mut ebo);
        gl::BindVertexArray(vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (vertices.len() * mem::size_of::<Vertex>()) as GLsizeiptr,
            &vertices[0] as *const Vertex as *const c_void,
            gl::STATIC_DRAW,
        );

        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
        gl::BufferData(
            gl::ELEMENT_ARRAY_BUFFER,
            (indices.len() * mem::size_of::<GLint>()) as GLsizeiptr,
            &indices[0] as *const i32 as *const c_void,
            gl::STATIC_DRAW,
        );

        let stride = mem::size_of::<Vertex>() as i32;

        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, stride, offset_of!(Vertex, position) as *const c_void);
        gl::EnableVertexAttribArray(0);

        gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, stride, offset_of!(Vertex, uv) as *const c_void);
        gl::EnableVertexAttribArray(1);

        gl::VertexAttribPointer(2, 3, gl::FLOAT, gl::FALSE, stride, offset_of!(Vertex, normal) as *const c_void);
        gl::EnableVertexAttribArray(2);

        gl::BindBuffer(gl::ARRAY_BUFFER, 0);
        gl::BindVertexArray(0);

        (vao, vertices.len() as i32 / 4 * 6)
    }
}
