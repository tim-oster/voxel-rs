use std::ops::{Add, Sub};
use std::rc::Rc;

use cgmath::{EuclideanSpace, InnerSpace, Point3, Vector3};
use imgui::{Condition, Id, TreeNodeFlags};

use crate::{graphics, systems};
use crate::core::Frame;
use crate::game::content::blocks;
use crate::game::worldgen;
use crate::game::worldgen::{Generator, Noise, SplinePoint};
use crate::graphics::camera::Camera;
use crate::graphics::svo::RenderParams;
use crate::systems::{storage, worldsvo};
use crate::systems::chunkloader::{ChunkEvent, ChunkLoader};
use crate::systems::jobs::JobSystem;
use crate::systems::physics::{Entity, Physics};
use crate::systems::storage::Storage;
use crate::world::chunk::ChunkPos;
use crate::world::world;

/// World is the game system responsible for keeping all chunks in the voxel world loaded and
/// renders them. It delegates loading from memory or generating chunks, as well as serialization
/// of the chunks into a SVO instance.
pub struct World {
    job_system: Rc<JobSystem>,

    loading_radius: u32,
    chunk_loader: ChunkLoader,
    pub storage: Storage,

    pub world: world::World,
    world_generator: systems::worldgen::Generator,
    world_generator_cfg: worldgen::Config,
    pub world_svo: graphics::svo::Svo,
    world_svo_mgr: worldsvo::Manager,

    physics: Physics,

    pub camera: Camera,
    pub selected_voxel: Option<Point3<f32>>,
    pub ambient_intensity: f32,
    pub sun_direction: Vector3<f32>,
}

impl World {
    pub fn new(job_system: Rc<JobSystem>, loading_radius: u32) -> World {
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
            job_system: Rc::clone(&job_system),
            loading_radius,
            chunk_loader: ChunkLoader::new(loading_radius, 0, 8),
            storage: Storage::new(),
            world: world::World::new(),
            world_generator: systems::worldgen::Generator::new(Rc::clone(&job_system), chunk_generator),
            world_generator_cfg: world_cfg,
            world_svo: graphics::svo::Svo::new(blocks::new_registry()),
            world_svo_mgr: worldsvo::Manager::new(job_system, loading_radius),
            physics: Physics::new(),
            camera: Camera::new(72.0, 1.0, 0.01, 1024.0),
            selected_voxel: None,
            ambient_intensity: 0.3,
            sun_direction: Vector3::new(-1.0, -1.0, -1.0).normalize(),
        }
    }

    pub fn update(&mut self, entity: &mut Entity, delta_time: f32) {
        self.handle_chunk_loading(entity.position);
        self.update_camera(entity);
        self.physics.step(delta_time, &self.world_svo, vec![entity]);
    }

    pub fn handle_window_resize(&mut self, aspect_ratio: f32) {
        self.camera.update_projection(72.0, aspect_ratio, 0.01, 1024.0);
    }

    pub fn reload_resources(&mut self) {
        self.world_svo.reload_resources();
    }

    fn handle_chunk_loading(&mut self, pos: Point3<f32>) {
        let chunk_events = self.chunk_loader.update(pos);
        if !chunk_events.is_empty() {
            let mut generate_count = 0;

            // camera position is always kept in center of the SVO, so it has to be "moved" to
            // the actual world space for frustum culling
            let old_pos = self.camera.position;
            self.camera.position = pos;
            let chunk_events = Self::sort_chunks_by_view_frustum(chunk_events, &self.camera);
            self.camera.position = old_pos;

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

    /// sort_chunks_by_view_frustum sorts the given chunk event to contain all chunks that are in
    /// the camera's view first first. All other chunks are sorted radially from forward to backward
    /// camera vector.
    fn sort_chunks_by_view_frustum(events: Vec<ChunkEvent>, camera: &Camera) -> Vec<ChunkEvent> {
        let mut visible_chunks = Vec::new();
        let mut other_chunks = Vec::new();
        for evt in events {
            let pos = evt.get_pos().to_block_pos().add(Vector3::new(16, 16, 16));
            if camera.is_in_frustum(pos.cast().unwrap(), 32.0) {
                visible_chunks.push(evt);
            } else {
                other_chunks.push(evt);
            }
        }

        other_chunks.sort_by(|lhs, rhs| {
            let lhs: Vector3<f32> = lhs.get_pos().to_block_pos().to_vec().cast().unwrap();
            let rhs: Vector3<f32> = rhs.get_pos().to_block_pos().to_vec().cast().unwrap();

            let tl = lhs.sub(camera.position.to_vec()).normalize();
            let tr = rhs.sub(camera.position.to_vec()).normalize();
            let dl = -tl.dot(camera.forward);
            let dr = -tr.dot(camera.forward);

            dl.total_cmp(&dr)
        });

        visible_chunks.extend(other_chunks.iter());
        visible_chunks
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

    pub fn render(&self, aspect_ratio: f32) {
        self.world_svo.render(RenderParams {
            ambient_intensity: self.ambient_intensity,
            light_dir: self.sun_direction,
            cam_pos: self.camera.position,
            view_mat: self.camera.get_camera_to_world_matrix(),
            fov_y_rad: self.camera.get_fov_y_deg().to_radians(),
            aspect_ratio,
            selected_voxel: self.selected_voxel,
        });
    }

    pub fn render_debug_window(&mut self, frame: &mut Frame) {
        imgui::Window::new("World Gen")
            .size([300.0, 100.0], Condition::FirstUseEver)
            .build(&frame.ui, || {
                frame.ui.input_int("sea level", &mut self.world_generator_cfg.sea_level).build();

                if frame.ui.button("generate") {
                    // NOTE: this is a very inefficient approach of regenerating the world, intended
                    // for testing/debugging purposes only

                    self.job_system.clear();
                    self.job_system.wait_until_processed();

                    let chunk_generator = Generator::new(1, self.world_generator_cfg.clone());

                    self.chunk_loader = ChunkLoader::new(self.loading_radius, 0, 8);
                    self.storage = Storage::new();
                    self.world = world::World::new();
                    self.world_generator = systems::worldgen::Generator::new(Rc::clone(&self.job_system), chunk_generator);
                    self.world_svo = graphics::svo::Svo::new(blocks::new_registry());
                    self.world_svo_mgr = worldsvo::Manager::new(Rc::clone(&self.job_system), self.loading_radius);
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
                            if i > 0 {
                                i -= 1;
                            }
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
                display_noise("continentalness", &mut self.world_generator_cfg.continentalness);
                display_noise("erosion", &mut self.world_generator_cfg.erosion);
            });
    }
}

mod tests {
    use std::rc::Rc;

    use cgmath::{Point3, Vector3};

    use crate::core::GlContext;
    use crate::game::world::World;
    use crate::gl_assert_no_error;
    use crate::graphics::framebuffer::{diff_images, Framebuffer};
    use crate::systems::jobs::JobSystem;
    use crate::systems::physics::{AABBDef, Entity};

    /// Tests if a standalone world object generates chunks, adds them to the SVO and renders them
    /// correctly after given enough time to properly load everything.
    #[test]
    fn end_to_end() {
        let (width, height) = (1024, 768);
        let aspect_ratio = width as f32 / height as f32;
        let _context = GlContext::new_headless(width, height); // do not drop context

        let mut player = Entity::new(
            Point3::new(-24.0, 80.0, 174.0),
            AABBDef::new(Vector3::new(-0.4, -1.7, -0.4), Vector3::new(0.8, 1.8, 0.8)),
        );
        player.euler_rotation = Vector3::new(0.0, -90f32.to_radians(), 0.0);
        player.caps.flying = true;

        let job_system = Rc::new(JobSystem::new(num_cpus::get() - 1));
        let mut world = World::new(Rc::clone(&job_system), 15);

        loop {
            world.update(&mut player, 0.1);

            if job_system.len() == 0 {
                break;
            }
        }

        job_system.wait_until_processed();

        let fb = Framebuffer::new(width as i32, height as i32);
        fb.bind();
        fb.clear(0.0, 0.0, 0.0, 1.0);
        world.render(aspect_ratio);
        fb.unbind();
        gl_assert_no_error!();

        let actual = fb.as_image();
        actual.save_with_format("assets/tests/game_world_end_to_end_actual.png", image::ImageFormat::Png).unwrap();

        let expected = image::open("assets/tests/game_world_end_to_end_expected.png").unwrap();
        let diff_percent = diff_images(&actual, &expected);
        assert!(diff_percent < 0.001, "difference: {:.5} < 0.001", diff_percent);
    }
}
