use std::rc::Rc;

use cgmath::{Point3, Vector3};
use imgui::Condition;

use crate::core::{Config, Frame, Window};
use crate::game::gameplay::Gameplay;
use crate::game::world::World;
use crate::systems::jobs::JobSystem;
use crate::systems::physics::{AABBDef, Entity};
use crate::world::chunk::ChunkPos;

pub struct Game {
    window: Window,
    job_system: Rc<JobSystem>,
    state: State,
}

struct State {
    job_system: Rc<JobSystem>,
    world: World,
    gameplay: Gameplay,
    player: Entity,
}

impl Game {
    pub fn new() -> Game {
        let mut window = Window::new(Config {
            width: 1024,
            height: 768,
            title: "voxel engine",
            msaa_samples: 0,
            headless: false,
            resizable: true,
        });
        window.request_grab_cursor(true);

        let mut player = Entity::new(
            Point3::new(-24.0, 80.0, 174.0),
            AABBDef::new(Vector3::new(-0.4, -1.7, -0.4), Vector3::new(0.8, 1.8, 0.8)),
        );
        player.euler_rotation = Vector3::new(0.0, -90f32.to_radians(), 0.0);
        player.caps.flying = true;

        let job_system = Rc::new(JobSystem::new(num_cpus::get() - 1));
        let world = World::new(Rc::clone(&job_system), 15, window.get_aspect());
        let gameplay = Gameplay::new();

        Game {
            window,
            job_system: Rc::clone(&job_system),
            state: State { job_system, world, gameplay, player },
        }
    }

    pub fn run(self) {
        let mut window = self.window;
        let mut state = self.state;

        loop {
            if window.should_close() {
                break;
            }
            window.update(|frame| {
                if frame.was_resized {
                    state.handle_window_resize(frame.size.0, frame.size.1, frame.get_aspect());
                }

                state.handle_update(frame);

                unsafe {
                    gl::ClearColor(0.0, 0.0, 0.0, 1.0);
                    gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);
                }

                state.handle_render(frame);
            });
        }

        // drop all refs to job system before stopping it
        drop(state);

        Rc::try_unwrap(self.job_system)
            .unwrap_or_else(|_| panic!("job_system is still used"))
            .stop();
    }
}

impl State {
    fn handle_update(&mut self, frame: &mut Frame) {
        self.handle_debug_keys(frame);

        self.world.update(&mut self.player, frame.stats.delta_time);
        self.gameplay.update(frame, &mut self.player, &mut self.world.world, &mut self.world.world_svo);
        self.world.selected_voxel = self.gameplay.looking_at_block.map(|result| result.pos);
    }

    fn handle_render(&mut self, frame: &mut Frame) {
        self.world.render(frame);
        self.gameplay.render_ui(frame.size);

        self.render_debug_window(frame);
        self.world.render_debug_window(frame);
    }

    fn handle_window_resize(&mut self, width: i32, height: i32, aspect_ratio: f32) {
        self.world.handle_resize(aspect_ratio);
        self.gameplay.handle_resize(width, height);
    }

    fn handle_resource_reload(&mut self) {
        self.world.reload_resources();
        self.gameplay.reload_resources();
        println!("tried reloading all resources");
    }

    fn render_debug_window(&self, frame: &mut Frame) {
        let camera = &self.world.camera;

        imgui::Window::new("Debug")
            .size([300.0, 100.0], Condition::FirstUseEver)
            .build(&frame.ui, || {
                frame.ui.text(format!(
                    "fps: {}, frame: {:.2}ms, update: {:.2}ms",
                    frame.stats.frames_per_second,
                    frame.stats.avg_frame_time_per_second * 1000.0,
                    frame.stats.avg_update_time_per_second * 1000.0,
                ));
                frame.ui.text(format!(
                    "abs pos: ({:.3},{:.3},{:.3})",
                    self.player.position.x, self.player.position.y, self.player.position.z,
                ));
                frame.ui.text(format!(
                    "cam pos: ({:.3},{:.3},{:.3})",
                    camera.position.x, camera.position.y, camera.position.z,
                ));
                frame.ui.text(format!(
                    "cam fwd: ({:.3},{:.3},{:.3})",
                    camera.forward.x, camera.forward.y, camera.forward.z,
                ));

                let mut pos = Point3::new(0.0, 0.0, 0.0);
                let mut norm = Vector3::new(0.0, 0.0, 0.0);
                if let Some(result) = self.gameplay.looking_at_block {
                    pos = result.pos;
                    norm = result.normal;
                }
                frame.ui.text(format!(
                    "block pos: ({:.2},{:.2},{:.2})",
                    pos.x, pos.y, pos.z,
                ));
                frame.ui.text(format!(
                    "block normal: ({},{},{})",
                    norm.x as i32, norm.y as i32, norm.z as i32,
                ));

                let chunk_pos = ChunkPos::from_block_pos(
                    self.player.position.x as i32,
                    self.player.position.y as i32,
                    self.player.position.z as i32,
                );
                frame.ui.text(format!(
                    "chunk pos: ({},{},{})",
                    chunk_pos.x, chunk_pos.y, chunk_pos.z,
                ));

                let svo_stats = self.world.world_svo.get_stats();
                frame.ui.text(format!(
                    "svo size: {:.3}mb, depth: {}",
                    svo_stats.size_bytes as f32 / 1024f32 / 1024f32,
                    svo_stats.depth,
                ));

                frame.ui.text(format!(
                    "queue length: {}",
                    self.job_system.len(),
                ));

                let mem_stats = self.world.storage.get_memory_stats();
                frame.ui.text(format!(
                    "chunk allocs used: {}, total: {}",
                    mem_stats.in_use, mem_stats.allocated,
                ));
            });
    }

    fn handle_debug_keys(&mut self, frame: &mut Frame) {
        if frame.input.was_key_pressed(&glfw::Key::E) {
            self.world.sun_direction = self.world.camera.forward;
        }
        if frame.input.was_key_pressed(&glfw::Key::R) {
            self.handle_resource_reload();
        }
        if frame.input.was_key_pressed(&glfw::Key::T) {
            let is_grabbed = frame.is_cursor_grabbed();
            frame.request_grab_cursor(!is_grabbed);
        }
    }
}
