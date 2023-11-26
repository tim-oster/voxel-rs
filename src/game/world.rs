use std::rc::Rc;

use cgmath::{InnerSpace, Point3, Vector3};
use imgui::{Condition, Id, TreeNodeFlags};

use crate::{graphics, systems};
use crate::core::Frame;
use crate::game::gameplay::blocks;
use crate::game::worldgen;
use crate::game::worldgen::{Generator, Noise, SplinePoint};
use crate::graphics::camera::Camera;
use crate::graphics::framebuffer::Framebuffer;
use crate::graphics::svo::RenderParams;
use crate::systems::{storage, worldsvo};
use crate::systems::chunkloading::{ChunkEvent, ChunkLoader};
use crate::systems::jobs::JobSystem;
use crate::systems::physics::{Entity, Physics};
use crate::systems::storage::Storage;
use crate::world::chunk::ChunkPos;

pub struct World {
    loading_radius: u32,
    chunk_loader: ChunkLoader,
    pub storage: Storage,

    pub world: systems::world::World,
    world_generator: systems::worldgen::Generator,
    world_generator_cfg: worldgen::Config,
    pub world_svo: graphics::svo::Svo,
    world_svo_mgr: worldsvo::Manager,

    physics: Physics,

    pub camera: Camera,
    pub selected_voxel: Option<Point3<f32>>,
    pub target_framebuffer: Option<Framebuffer>, // TODO matrices are incorrect if FB is used

    pub ambient_intensity: f32,
    pub sun_direction: Vector3<f32>,
}

impl World {
    // TODO split into world & world_renderer?

    pub fn new(job_system: Rc<JobSystem>, loading_radius: u32, aspect_ratio: f32) -> World {
        let world_cfg = worldgen::Config {
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
        let chunk_generator = Generator::new(1, world_cfg.clone());

        World {
            loading_radius,
            chunk_loader: ChunkLoader::new(loading_radius, 0, 8),
            storage: Storage::new(),
            world: systems::world::World::new(),
            world_generator: systems::worldgen::Generator::new(Rc::clone(&job_system), chunk_generator),
            world_generator_cfg: world_cfg,
            world_svo: graphics::svo::Svo::new(blocks::new_registry()),
            world_svo_mgr: worldsvo::Manager::new(job_system, loading_radius),
            physics: Physics::new(),
            camera: Camera::new(72.0, aspect_ratio, 0.01, 1024.0),
            selected_voxel: None,
            target_framebuffer: None,
            ambient_intensity: 0.3,
            sun_direction: Vector3::new(-1.0, -1.0, -1.0).normalize(),
        }
    }

    pub fn update(&mut self, entity: &mut Entity, delta_time: f32) {
        self.handle_chunk_loading(entity.position);
        self.update_camera(entity);
        self.physics.step(delta_time, &self.world_svo, vec![entity]);
    }

    pub fn handle_resize(&mut self, aspect_ratio: f32) {
        self.camera.update_projection(72.0, aspect_ratio, 0.01, 1024.0);
    }

    pub fn reload_resources(&mut self) {
        self.world_svo.reload_resources();
    }

    fn handle_chunk_loading(&mut self, pos: Point3<f32>) {
        let mut generate_count = 0;
        let chunk_events = self.chunk_loader.update(pos);
        for event in &chunk_events {
            match event {
                ChunkEvent::Load { pos, lod } => {
                    let result = self.storage.load(pos);
                    if result.is_ok() {
                        let mut chunk = result.ok().unwrap();
                        chunk.lod = *lod;
                        self.world.set_chunk(chunk);
                        continue;
                    }

                    let err = result.err().unwrap();
                    match err {
                        storage::LoadError::NotFound => {
                            let mut chunk = self.storage.new_chunk(*pos);
                            chunk.lod = *lod;
                            self.world_generator.enqueue_chunk(chunk);
                            generate_count += 1;
                        }
                    }
                }
                ChunkEvent::Unload { pos } => {
                    self.world_generator.dequeue_chunk(pos);
                    self.world.remove_chunk(pos);
                }
                ChunkEvent::LodChange { pos, lod } => {
                    if let Some(chunk) = self.world.get_chunk_mut(pos) {
                        chunk.lod = *lod;
                    }
                }
            }
        }
        if !chunk_events.is_empty() {
            println!("generate {} new chunks", generate_count);
        }
        for chunk in self.world_generator.get_generated_chunks(40) {
            if self.chunk_loader.is_loaded(&chunk.pos) {
                self.world.set_chunk(chunk);
            }
        }
        for pos in self.world.get_changed_chunks() {
            if let Some(chunk) = self.world.get_chunk(&pos) {
                self.world_svo_mgr.set_chunk(chunk);
            } else {
                self.world_svo_mgr.remove_chunk(&pos);
            }
        }

        let current_chunk_pos = ChunkPos::from(pos);
        self.world_svo_mgr.update(&current_chunk_pos, &mut self.world_svo);
    }

    fn update_camera(&mut self, entity: &Entity) {
        let pos = entity.position;
        let current_chunk_pos = ChunkPos::from_block_pos(pos.x as i32, pos.y as i32, pos.z as i32);

        let chunk_world_pos = Point3::new(current_chunk_pos.x as f32, current_chunk_pos.y as f32, current_chunk_pos.z as f32) * 32.0;
        let delta = pos - chunk_world_pos;

        self.camera.position.x = self.loading_radius as f32 * 32.0 + delta.x;
        self.camera.position.y = self.loading_radius as f32 * 32.0 + delta.y;
        self.camera.position.z = self.loading_radius as f32 * 32.0 + delta.z;

        self.camera.set_forward(entity.get_forward());
    }

    pub fn render(&self, frame: &Frame) {
        if let Some(fb) = &self.target_framebuffer {
            fb.bind();
            fb.clear(0.0, 0.0, 0.0, 1.0);
        }

        self.world_svo.render(RenderParams {
            ambient_intensity: self.ambient_intensity,
            light_dir: self.sun_direction,
            cam_pos: self.camera.position,
            view_mat: self.camera.get_camera_to_world_matrix(),
            fov_y_rad: self.camera.get_fov_y_deg().to_radians(),
            aspect_ratio: frame.get_aspect(),
            selected_voxel: self.selected_voxel,
        });

        if let Some(fb) = &self.target_framebuffer {
            fb.unbind();
        }
    }

    pub fn render_debug_window(&mut self, frame: &mut Frame) {
        let cfg = &mut self.world_generator_cfg;

        imgui::Window::new("World Gen")
            .size([300.0, 100.0], Condition::FirstUseEver)
            .build(&frame.ui, || {
                frame.ui.input_int("sea level", &mut cfg.sea_level).build();

                // TODO
                if frame.ui.button("generate") {
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
                }

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
                display_noise("continentalness", &mut cfg.continentalness);
                display_noise("erosion", &mut cfg.erosion);
            });
    }
}

// TODO fix end to end test
// mod tests {
//     use std::time::{Duration, Instant};
//
//     use cgmath::{Point3, Vector3};
//     use image::GenericImageView;
//
//     use crate::core::{Config, Window};
//     use crate::game::game::Game;
//     use crate::graphics::framebuffer::{diff_rgba3, Framebuffer};
//     use crate::systems::physics::{AABBDef, Entity};
//
//     #[test]
//     fn test_game_end_to_end() {
//         let window = Window::new(Config {
//             width: 1024,
//             height: 768,
//             title: "",
//             msaa_samples: 0,
//             headless: true,
//             resizable: false,
//         });
//
//         let mut player = Entity::new(
//             Point3::new(-24.0, 80.0, 174.0),
//             AABBDef::new(Vector3::new(-0.4, -1.7, -0.4), Vector3::new(0.8, 1.8, 0.8)),
//         );
//         player.euler_rotation = Vector3::new(0.0, -90f32.to_radians(), 0.0);
//         player.caps.flying = true;
//
//         let fb = Framebuffer::new(window.get_size().0, window.get_size().1);
//
//         let mut game = Game::new_from(window, 15, player);
//         game.update_hook = Some(Box::new({
//             let start_time = Instant::now();
//             move |frame, state| {
//                 if start_time.elapsed() > Duration::from_secs(5) && state.job_system.len() == 0 {
//                     frame.request_close();
//                 }
//             }
//         }));
//         game.state.world.target_framebuffer = Some(fb);
//
//         let mut state = game.run_with_state();
//         let fb = state.world.target_framebuffer.take().unwrap();
//         let pixels = fb.read_pixels();
//         let actual = image::RgbaImage::from_raw(fb.width() as u32, fb.height() as u32, pixels).unwrap();
//         let actual = image::DynamicImage::ImageRgba8(actual).flipv();
//         actual.save_with_format("assets/tests/e2e_actual.png", image::ImageFormat::Png).unwrap();
//
//         let expected = image::open("assets/tests/e2e_expected.png").unwrap();
//
//         let mut accum = 0;
//         let zipper = actual.pixels().zip(expected.pixels());
//         for (pixel1, pixel2) in zipper {
//             accum += diff_rgba3(pixel1.2, pixel2.2);
//         }
//         let diff_percent = accum as f64 / (255.0 * 3.0 * (actual.width() * actual.height()) as f64);
//         println!("difference: {:.5}", diff_percent);
//         assert!(diff_percent < 0.001);
//     }
// }
