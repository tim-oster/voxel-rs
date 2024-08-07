use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use cgmath::{Point3, Vector3};
use imgui::Condition;

use crate::core::{Buffering, Config, Frame, Window};
use crate::gamelogic::benchmark;
use crate::gamelogic::gameplay::Gameplay;
use crate::gamelogic::world::World;
use crate::global_allocated_bytes;
use crate::systems::jobs::JobSystem;
use crate::systems::physics::{AABBDef, Entity};
use crate::systems::worldsvo;
use crate::world::chunk::ChunkPos;

pub struct GameArgs {
    pub mc_world: Option<String>,
    pub player_pos: Point3<f32>,
    pub player_euler_rot: Vector3<f32>,
    pub detach_input: bool,
    pub render_distance: u32,
    pub fov_y_deg: f32,
    pub render_shadows: bool,
    pub no_lod: bool,
    pub gpu_buffer_size_mb: usize,
}

/// Game runs the actual game loop and handles communication and calling to the different game
/// systems.
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

    physics_target_fps: u32,
    physics_fps: u32,

    render_debug_ui: bool,
    plot_refresh: Instant,
    plot_fps: Plot,
    plot_frame_time: Plot,
    plot_jobs: Plot,
    plot_memory: Plot,
}

impl Game {
    pub fn new(args: GameArgs) -> Self {
        let mut window = Window::new(&Config {
            width: 1920,
            height: 1080,
            title: "voxel engine",
            msaa_samples: 0,
            headless: false,
            resizable: true,
            buffering: Buffering::Single,
            target_fps: None,
        });

        window.request_grab_cursor(!args.detach_input);

        let mut player = Entity::new(
            args.player_pos,
            AABBDef::new(Vector3::new(-0.4, -1.7, -0.4), Vector3::new(0.8, 1.8, 0.8)),
        );
        player.euler_rotation = args.player_euler_rot;
        player.caps.flying = true;

        let job_system = Rc::new(JobSystem::new(num_cpus::get() - 1));
        let world = World::new(Rc::clone(&job_system), args.fov_y_deg, args.render_shadows, args.render_distance, args.no_lod, args.mc_world, args.gpu_buffer_size_mb);
        let gameplay = Gameplay::new();

        Self {
            window,
            job_system: Rc::clone(&job_system),
            state: State {
                job_system,
                world,
                gameplay,
                player,
                physics_target_fps: 250,
                physics_fps: 0,
                render_debug_ui: true,
                plot_refresh: Instant::now(),
                plot_fps: Plot::new(),
                plot_frame_time: Plot::new(),
                plot_jobs: Plot::new(),
                plot_memory: Plot::new(),
            },
        }
    }

    pub fn run(self, closer: &Arc<AtomicBool>) {
        let mut window = self.window;
        let mut state = self.state;

        let fixed_frame_time = 1.0 / state.physics_target_fps as f32;
        let mut frame_time_accumulator = 0.0;
        let mut last_fixed_frame_measurement = Instant::now();
        let mut fixed_frames = 0;

        loop {
            if closer.load(Ordering::Relaxed) {
                window.request_close();
            }
            if window.should_close() {
                break;
            }
            window.update(|frame| {
                // per frame update
                if frame.was_resized {
                    state.handle_window_resize(frame.size.0, frame.size.1, frame.get_aspect());
                }
                state.update(frame);

                // accumulate frame time for fixed update
                frame_time_accumulator += frame.stats.delta_time;

                // consume accumulated time for fixed physics updates
                #[allow(clippy::while_float)]
                while frame_time_accumulator >= fixed_frame_time {
                    state.update_fixed(frame, fixed_frame_time);
                    frame_time_accumulator -= fixed_frame_time;
                    fixed_frames += 1;
                }
                if last_fixed_frame_measurement.elapsed().as_secs() >= 1 {
                    state.physics_fps = fixed_frames;
                    last_fixed_frame_measurement = Instant::now();
                    fixed_frames = 0;
                }

                // draw frame
                unsafe {
                    gl::ClearColor(0.0, 0.0, 0.0, 1.0);
                    gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);
                }
                state.render(frame);
            });
        }

        self.job_system.clear();
        self.job_system.wait_until_processed();

        // drop all refs to job system before stopping it
        drop(state);

        Rc::try_unwrap(self.job_system)
            .unwrap_or_else(|_| panic!("job_system is still used"))
            .stop();
    }
}

impl State {
    fn update_fixed(&mut self, frame: &mut Frame, delta_time: f32) {
        self.world.update_fixed(&mut self.player, delta_time);
    }

    fn update(&mut self, frame: &mut Frame) {
        self.handle_debug_keys(frame);

        self.world.update(&self.player);
        self.gameplay.update(frame, &mut self.player, &mut self.world);
        self.world.selected_voxel = self.gameplay.looking_at_block.map(|result| result.pos);
    }

    fn render(&mut self, frame: &mut Frame) {
        self.world.render(frame.get_aspect());
        self.gameplay.render_ui(frame.size);

        if self.render_debug_ui {
            self.render_debug_window(frame);
            self.world.render_debug_window(frame);
        }
    }

    fn handle_window_resize(&mut self, width: i32, height: i32, aspect_ratio: f32) {
        self.world.handle_window_resize(width, height, aspect_ratio);
        self.gameplay.handle_window_resize(width, height);
    }

    fn handle_resource_reload(&mut self) {
        self.world.reload_resources();
        self.gameplay.reload_resources();
        println!("tried reloading all resources");
    }

    fn render_debug_window(&mut self, frame: &mut Frame) {
        let camera = &self.world.camera;

        frame.ui.window("Debug")
            .position([8.0, 8.0], Condition::Once)
            .size([400.0, 290.0], Condition::Once)
            .collapsed(false, Condition::Once)
            .build(|| {
                frame.ui.text(format!(
                    "fps: {}, frame: {:.2}ms, update: {:.2}ms ({})",
                    frame.stats.frames_per_second,
                    frame.stats.avg_frame_time_per_second * 1000.0,
                    frame.stats.avg_update_time_per_second * 1000.0,
                    worldsvo::SVO_TYPE.name,
                ));
                frame.ui.text(format!(
                    "physics fps: {}, every: {:.2}ms",
                    self.physics_fps,
                    (1.0 / self.physics_target_fps as f32) * 1000.0,
                ));

                frame.ui.separator();

                frame.ui.text(format!(
                    "abs pos: ({:.3}, {:.3}, {:.3})",
                    self.player.position.x, self.player.position.y, self.player.position.z,
                ));
                frame.ui.text(format!(
                    "cam pos: ({:.3}, {:.3}, {:.3})",
                    camera.position.x, camera.position.y, camera.position.z,
                ));
                frame.ui.text(format!(
                    "cam fwd: ({:.3}, {:.3}, {:.3})",
                    camera.forward.x, camera.forward.y, camera.forward.z,
                ));
                frame.ui.text(format!(
                    "velocity: ({:.3}, {:.3}, {:.3})",
                    self.player.velocity.x, self.player.velocity.y, self.player.velocity.z,
                ));

                frame.ui.separator();

                let mut pos = Point3::new(0.0, 0.0, 0.0);
                let mut norm = Vector3::new(0.0, 0.0, 0.0);
                if let Some(result) = self.gameplay.looking_at_block {
                    pos = result.pos;
                    norm = result.normal;
                }
                frame.ui.text(format!(
                    "block pos: ({:.2}, {:.2}, {:.2})",
                    pos.x, pos.y, pos.z,
                ));
                frame.ui.text(format!(
                    "block normal: ({}, {}, {})",
                    norm.x as i32, norm.y as i32, norm.z as i32,
                ));

                let chunk_pos = ChunkPos::from_block_pos(
                    self.player.position.x as i32,
                    self.player.position.y as i32,
                    self.player.position.z as i32,
                );
                frame.ui.text(format!(
                    "chunk pos: ({}, {}, {})",
                    chunk_pos.x, chunk_pos.y, chunk_pos.z,
                ));

                frame.ui.separator();

                frame.ui.text(format!(
                    "world chunks used: {}, total: {}, size: {:.3}mb",
                    self.world.chunk_storage_allocator.used_count(),
                    self.world.chunk_storage_allocator.allocated_count(),
                    self.world.chunk_storage_allocator.allocated_bytes() as f32 / 1024f32 / 1024f32,
                ));

                let bytes = global_allocated_bytes();
                frame.ui.text(format!(
                    "engine memory: {:.3}mb",
                    bytes as f32 / 1024f32 / 1024f32,
                ));

                frame.ui.separator();

                let svo_stats = self.world.world_svo.get_stats();
                benchmark::track_svo_gpu_bytes(svo_stats.used_bytes);
                frame.ui.text(format!(
                    "gpu svo size: {:.3}mb / {:.3}mb, depth: {}",
                    svo_stats.used_bytes as f32 / 1024f32 / 1024f32,
                    svo_stats.capacity_bytes as f32 / 1024f32 / 1024f32,
                    svo_stats.depth,
                ));

                let alloc_stats = self.world.world_svo.get_alloc_stats();
                frame.ui.text(format!(
                    "cpu svo size: {:.3}mb",
                    alloc_stats.world_svo_buffer_bytes as f32 / 1024f32 / 1024f32,
                ));
                frame.ui.text(format!(
                    "chunk buffers used: {}, total: {}, size: {:.3}mb",
                    alloc_stats.chunk_buffers_used,
                    alloc_stats.chunk_buffers_allocated,
                    alloc_stats.chunk_buffers_bytes_total as f32 / 1024f32 / 1024f32,
                ));
            });

        let now = Instant::now();
        if now > self.plot_refresh {
            self.plot_refresh += Duration::from_secs_f32(1.0 / 60.0);

            benchmark::track_fps(frame.stats.frames_per_second, frame.stats.avg_frame_time_per_second);

            self.plot_fps.add(frame.stats.frames_per_second as f32);
            self.plot_frame_time.add(frame.stats.avg_frame_time_per_second);
            self.plot_jobs.add(self.job_system.queue_len() as f32);

            let bytes = global_allocated_bytes();
            self.plot_memory.add(bytes as f32 / 1024.0 / 1024.0);
        }

        frame.ui.window("Graphs")
            .position([frame.size.0 as f32 - 400.0 - 8.0, 8.0], Condition::Once)
            .size([400.0, 170.0], Condition::Once)
            .collapsed(false, Condition::Once)
            .build(|| {
                self.plot_fps.render(frame.ui, "fps");
                self.plot_frame_time.render(frame.ui, "frame time");
                self.plot_jobs.render(frame.ui, "job queue");
                self.plot_memory.render(frame.ui, "memory");
            });
    }

    fn handle_debug_keys(&mut self, frame: &mut Frame) {
        if frame.input.was_key_pressed(glfw::Key::P) {
            self.render_debug_ui = !self.render_debug_ui;
        }
        if frame.input.was_key_pressed(glfw::Key::E) {
            self.world.sun_direction = self.world.camera.forward;
        }
        if frame.input.was_key_pressed(glfw::Key::R) {
            self.handle_resource_reload();
        }
        if frame.input.was_key_pressed(glfw::Key::T) {
            let is_grabbed = frame.is_cursor_grabbed();
            frame.request_grab_cursor(!is_grabbed);
        }
    }
}

/// Plot is a convenience wrapper for drawing imgui plot lines.
struct Plot<const N: usize = 90> {
    data: [f32; N],
}

impl<const N: usize> Plot<N> {
    fn new() -> Self {
        Self {
            data: [0.0; N],
        }
    }

    fn add(&mut self, elem: f32) {
        self.data.rotate_left(1);
        self.data[N - 1] = elem;
    }

    fn render(&self, ui: &imgui::Ui, label: &str) {
        ui.plot_lines(label, &self.data)
            .scale_min(0.0)
            .graph_size([0.0, 30.0])
            .build();
    }
}
