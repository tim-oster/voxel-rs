use std::ops::{Add, Sub};
use std::rc::Rc;
use std::sync::Arc;

use cgmath::{EuclideanSpace, InnerSpace, Point3, Vector3};
use imgui::{Condition, TreeNodeFlags};

use crate::{graphics, systems};
use crate::core::Frame;
use crate::gamelogic::content::blocks;
use crate::gamelogic::worldgen;
use crate::gamelogic::worldgen::{Generator, Noise, SplinePoint};
use crate::graphics::camera::Camera;
use crate::graphics::framebuffer::Framebuffer;
use crate::graphics::svo::RenderParams;
use crate::systems::chunkloader::{ChunkEvent, ChunkLoader};
use crate::systems::jobs::JobSystem;
use crate::systems::physics::{Entity, Physics};
use crate::systems::storage::{MinecraftStorage, NopStorage, Storage};
use crate::systems::worldsvo;
use crate::world::chunk::{Chunk, ChunkPos, ChunkStorageAllocator};
use crate::world::world;

/// World is the game system responsible for keeping all chunks in the voxel world loaded and
/// renders them. It delegates loading from memory or generating chunks, as well as serialization
/// of the chunks into a SVO instance.
pub struct World {
    job_system: Rc<JobSystem>,

    chunk_loader: ChunkLoader,
    pub chunk_storage_allocator: Arc<ChunkStorageAllocator>,
    pub storage: Box<dyn Storage>,

    pub world: world::World,
    world_generator: systems::worldgen::Generator,
    world_generator_cfg: worldgen::Config,
    pub world_svo: worldsvo::Svo,
    world_fbo: Framebuffer,

    physics: Physics,

    pub camera: Camera,
    pub selected_voxel: Option<Point3<f32>>,
    pub ambient_intensity: f32,
    pub sun_direction: Vector3<f32>,
    pub render_shadows: bool,
    pub shadow_distance: f32,
}

impl World {
    pub fn new(job_system: Rc<JobSystem>, loading_radius: u32, mc_world_path: Option<&str>) -> Self {
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
        let chunk_allocator = Arc::new(ChunkStorageAllocator::new());
        let chunk_generator = Generator::new(1, world_cfg.clone());
        let graphics_svo = graphics::Svo::new(&blocks::new_registry());

        Self {
            job_system: Rc::clone(&job_system),
            chunk_loader: ChunkLoader::new(loading_radius, 0, 8),
            chunk_storage_allocator: chunk_allocator.clone(),
            storage: if let Some(path) = mc_world_path {
                Box::new(MinecraftStorage::new(job_system.clone(), chunk_allocator.clone(), path))
            } else {
                Box::new(NopStorage::new())
            },
            world: world::World::new(),
            world_generator: systems::worldgen::Generator::new(Rc::clone(&job_system), chunk_allocator, chunk_generator),
            world_generator_cfg: world_cfg,
            world_svo: worldsvo::Svo::new(job_system, graphics_svo, loading_radius),
            world_fbo: Framebuffer::new(1920, 1080, false, false),
            physics: Physics::new(),
            camera: Camera::new(80.0, 1.0, 0.01, 1024.0),
            selected_voxel: None,
            ambient_intensity: 0.3,
            sun_direction: Vector3::new(-1.0, -1.0, -1.0).normalize(),
            render_shadows: true,
            shadow_distance: 500.0,
        }
    }

    pub fn update_fixed(&mut self, entity: &mut Entity, delta_time: f32) {
        self.physics.step(delta_time, &self.world_svo, entity);
    }

    pub fn update(&mut self, entity: &Entity) {
        self.camera.position = entity.position;
        self.camera.forward = entity.get_forward();

        self.handle_chunk_loading();
    }

    pub fn handle_window_resize(&mut self, width: i32, height: i32, aspect_ratio: f32) {
        self.camera.update_projection(72.0, aspect_ratio, 0.01, 1024.0);
        self.world_fbo = Framebuffer::new(width, height, false, false);
    }

    pub fn reload_resources(&mut self) {
        self.world_svo.reload_resources();
    }

    fn handle_chunk_loading(&mut self) {
        let chunk_events = self.chunk_loader.update(self.camera.position);
        if !chunk_events.is_empty() {
            let mut loaded_count = 0;

            let chunk_events = Self::sort_chunks_by_view_frustum(chunk_events, &self.camera);
            for event in &chunk_events {
                match event {
                    ChunkEvent::Load { pos, lod } => {
                        self.storage.load(pos, *lod);
                        loaded_count += 1;
                    }
                    ChunkEvent::Unload { pos } => {
                        self.storage.dequeue_chunk(pos);
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
                println!("loaded {loaded_count} new chunks");
            }
        }
        for chunk in self.storage.get_load_results(400) {
            if self.chunk_loader.is_loaded(&chunk.pos) {
                if chunk.value.0.is_none() {
                    self.world_generator.enqueue_chunk(chunk.pos, chunk.value.1);
                    continue;
                }
                let chunk = chunk.value.0.unwrap();
                self.world.set_chunk(chunk);
            }
        }
        for chunk in self.world_generator.get_generated_chunks(400) {
            if self.chunk_loader.is_loaded(&chunk.pos) {
                let pos = chunk.pos;

                // set chunk to world but shortcut the change detection mechanism to avoid unnecessary iterations
                self.world.set_chunk_unchanged(chunk);

                let chunk = self.world.borrow_chunk(&pos).unwrap();
                self.world_svo.set_chunk(chunk);
            }
        }
        for pos in self.world.get_changed_chunks(400) {
            if let Some(chunk) = self.world.get_chunk(&pos) {
                if chunk.storage.is_some() {
                    let chunk = self.world.borrow_chunk(&pos).unwrap();
                    self.world_svo.set_chunk(chunk);
                }
            } else {
                self.world_svo.remove_chunk(&pos);
            }
        }

        let current_chunk_pos = ChunkPos::from(self.camera.position);
        let chunks = self.world_svo.update(&current_chunk_pos);
        for chunk in chunks {
            self.world.return_chunk(chunk);
        }
    }

    /// `sort_chunks_by_view_frustum` sorts the given chunk event to contain all chunks that are in
    /// the camera's view first. All other chunks are sorted radially from forward to backward
    /// camera vector.
    fn sort_chunks_by_view_frustum(events: Vec<ChunkEvent>, camera: &Camera) -> Vec<ChunkEvent> {
        let mut visible_chunks = Vec::new();
        let mut other_chunks = Vec::new();
        for evt in events {
            let pos = evt.get_pos().as_block_pos().add(Vector3::new(16, 16, 16));
            if camera.is_in_frustum(pos.cast().unwrap(), 32.0) {
                visible_chunks.push(evt);
            } else {
                other_chunks.push(evt);
            }
        }

        other_chunks.sort_by(|lhs, rhs| {
            let lhs: Vector3<f32> = lhs.get_pos().as_block_pos().to_vec().cast().unwrap();
            let rhs: Vector3<f32> = rhs.get_pos().as_block_pos().to_vec().cast().unwrap();

            let tl = lhs.sub(camera.position.to_vec()).normalize();
            let tr = rhs.sub(camera.position.to_vec()).normalize();
            let dl = -tl.dot(camera.forward);
            let dr = -tr.dot(camera.forward);

            dl.total_cmp(&dr)
        });

        visible_chunks.extend(other_chunks.iter());
        visible_chunks
    }

    pub fn add_chunk(&mut self, chunk: Chunk) {
        self.chunk_loader.add_loaded_chunk(chunk.pos, chunk.lod);
        self.world.set_chunk(chunk);
    }

    pub fn render(&self, aspect_ratio: f32) {
        self.world_svo.render(RenderParams {
            ambient_intensity: self.ambient_intensity,
            light_dir: self.sun_direction,
            cam_pos: self.camera.position,
            cam_fwd: self.camera.forward,
            cam_up: self.camera.up,
            fov_y_rad: self.camera.get_fov_y_deg().to_radians(),
            aspect_ratio,
            selected_voxel: self.selected_voxel,
            render_shadows: self.render_shadows,
            shadow_distance: self.shadow_distance,
        }, &self.world_fbo);
        self.world_fbo.blit_to_default();
    }

    pub fn render_debug_window(&mut self, frame: &mut Frame) {
        frame.ui.window("World Gen")
            .position([8.0, 2.0f32.mul_add(8.0, 290.0)], Condition::Once)
            .size([400.0, 400.0], Condition::Once)
            .collapsed(true, Condition::Once)
            .build(|| {
                frame.ui.input_int("sea level", &mut self.world_generator_cfg.sea_level).build();

                if frame.ui.button("generate") {
                    // NOTE: this is a very inefficient approach of regenerating the world, intended
                    // for testing/debugging purposes only

                    self.job_system.clear();
                    self.job_system.wait_until_processed();

                    let chunk_generator = Generator::new(1, self.world_generator_cfg.clone());
                    let graphics_svo = graphics::Svo::new(&blocks::new_registry());

                    self.chunk_loader = ChunkLoader::new(self.chunk_loader.get_radius(), 0, 8);
                    self.world = world::World::new();
                    self.world_generator = systems::worldgen::Generator::new(Rc::clone(&self.job_system), self.chunk_storage_allocator.clone(), chunk_generator);
                    self.world_svo = worldsvo::Svo::new(Rc::clone(&self.job_system), graphics_svo, self.world_svo.get_render_distance());
                }

                frame.ui.new_line();

                let display_noise = |label: &str, noise: &mut Noise| {
                    if !frame.ui.collapsing_header(label, TreeNodeFlags::DEFAULT_OPEN) {
                        return;
                    }

                    let stack = frame.ui.push_id(label);

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
                        let stack = frame.ui.push_id_int(i as i32);

                        frame.ui.text(format!("#{i}"));
                        frame.ui.same_line();
                        if frame.ui.small_button("del") {
                            noise.spline_points.remove(i);
                            i = i.saturating_sub(1);
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

                        let sp = &mut noise.spline_points[i];
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

        frame.ui.window("Settings")
            .position([frame.size.0 as f32 - 400.0 - 8.0, 2.0f32.mul_add(8.0, 170.0)], Condition::Once)
            .size([400.0, 320.0], Condition::Once)
            .collapsed(true, Condition::Once)
            .build(|| {
                fn render_control_list(ui: &imgui::Ui, header: &str, controls: &[&str]) {
                    ui.text(header);
                    ui.separator();

                    ui.columns(2, header, true);

                    for (i, c) in controls.iter().enumerate() {
                        ui.text(c);
                        if i == (controls.len() - 1) / 2 {
                            ui.next_column();
                        }
                    }

                    ui.columns(1, "", false);
                }

                let old_rd = self.chunk_loader.get_radius() as i32;
                let mut new_rd = old_rd;
                frame.ui.input_int("render distance", &mut new_rd).build();
                new_rd = new_rd.clamp(1, 50);
                if new_rd != old_rd {
                    self.chunk_loader.set_radius(new_rd as u32);
                    self.world_svo.set_radius(new_rd as u32);
                }

                let old_fov = self.camera.get_fov_y_deg();
                let mut new_fov = old_fov;
                frame.ui.input_float("FOV", &mut new_fov).step(2.0).build();
                new_fov = new_fov.clamp(1.0, 130.0);
                if (new_fov - old_fov).abs() > f32::EPSILON {
                    self.camera.set_fov_y_deg(new_fov);
                }

                frame.ui.checkbox("render shadows", &mut self.render_shadows);
                frame.ui.input_float("shadow distance", &mut self.shadow_distance).step(1.0).build();

                frame.ui.new_line();
                frame.ui.separator();
                frame.ui.new_line();

                render_control_list(frame.ui, "Debug Controls", &[
                    "P: toggle debug UI",
                    "E: set sun to view dir",
                    "R: reload assets",
                    "T: toggle mouse grab",
                    "Esc: close game"
                ]);

                frame.ui.new_line();

                render_control_list(frame.ui, "Game Controls", &[
                    "L Click: break block",
                    "R Click: place block",
                    "M Click: select block",
                    "0..9: select from item bar",
                    "WASD: movement",
                    "F: toggle fly mode",
                    "LShift: sprint / descend",
                    "Space: jump / ascend",
                ]);
            });
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::rc::Rc;

    use cgmath::{Point3, Vector3};

    use crate::core::GlContext;
    use crate::gamelogic::world::World;
    use crate::gl_assert_no_error;
    use crate::graphics::framebuffer::diff_images;
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
        let mut world = World::new(Rc::clone(&job_system), 15, None);
        world.handle_window_resize(width as i32, height as i32, aspect_ratio);

        loop {
            world.update(&player);

            if !world.storage.has_pending_jobs() && !world.world_generator.has_pending_jobs() && !world.world_svo.has_pending_jobs() {
                break;
            }
        }

        job_system.wait_until_empty_and_processed();

        world.render(aspect_ratio);
        gl_assert_no_error!();

        let actual = world.world_fbo.as_image();
        actual.save_with_format("assets/tests/gamelogic_world_end_to_end_actual.png", image::ImageFormat::Png).unwrap();

        let expected = image::open("assets/tests/gamelogic_world_end_to_end_expected.png").unwrap();
        let diff_percent = diff_images(&actual, &expected);
        let threshold = env::var("TEST_WORLD_E2E_THRESHOLD").map_or(0.001, |x| x.parse::<f64>().unwrap());
        assert!(diff_percent < threshold, "difference: {:.5} < {:.5}", diff_percent, threshold);
    }
}
