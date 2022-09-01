use std::cell::RefCell;
use std::time::{Duration, Instant};

use glfw::Context;

use crate::core::imgui as imgui_wrapper;
use crate::core::Input;

pub struct Window {
    context: glfw::Glfw,
    imgui: imgui_wrapper::Wrapper,

    window: RefCell<glfw::Window>,
    events: std::sync::mpsc::Receiver<(f64, glfw::WindowEvent)>,

    current_stats: FrameStats,
    is_cursor_grabbed: bool,
    input: Input,
}

pub struct FrameStats {
    last_frame: Instant,
    last_measurement: Instant,
    frame_count: i32,
    frame_time_accumulation: Duration,
    update_time_accumulation: Duration,

    pub delta_time: f32,
    pub frames_per_second: i32,
    pub avg_frame_time_per_second: f32,
    pub avg_update_time_per_second: f32,
}

impl Window {
    pub fn new(width: u32, height: u32, title: &str, msaa_samples: u32) -> Self {
        let mut context = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
        context.window_hint(glfw::WindowHint::ContextVersion(4, 6));
        context.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));

        if msaa_samples > 0 {
            context.window_hint(glfw::WindowHint::Samples(Some(msaa_samples)));
        }

        let (mut window, events) = context
            .create_window(width, height, title, glfw::WindowMode::Windowed)
            .expect("failed to create window");

        window.make_current();
        window.set_all_polling(true);
        window.set_cursor_mode(glfw::CursorMode::Disabled);

        gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

        let imgui = imgui_wrapper::Wrapper::new(&mut window);

        Window {
            context,
            imgui,
            window: RefCell::new(window),
            events,
            current_stats: FrameStats {
                last_frame: Instant::now(),
                last_measurement: Instant::now(),
                frame_count: 0,
                frame_time_accumulation: Duration::new(0, 0),
                update_time_accumulation: Duration::new(0, 0),

                delta_time: 1.0,
                frames_per_second: 0,
                avg_frame_time_per_second: 0.0,
                avg_update_time_per_second: 0.0,
            },
            is_cursor_grabbed: false,
            input: Input::new(),
        }
    }

    pub fn update<F: FnOnce(&mut Frame)>(&mut self, f: F) {
        let delta_time = self.current_stats.last_frame.elapsed();
        self.current_stats.frame_time_accumulation += delta_time;
        self.current_stats.delta_time = delta_time.as_secs_f32() / (1.0 / 60.0);
        self.current_stats.last_frame = Instant::now();

        self.current_stats.frame_count += 1;
        if self.current_stats.last_measurement.elapsed() > Duration::from_secs(1) {
            self.current_stats.frames_per_second = self.current_stats.frame_count;
            self.current_stats.avg_frame_time_per_second = self.current_stats.frame_time_accumulation.as_secs_f32() / self.current_stats.frame_count as f32;
            self.current_stats.avg_update_time_per_second = self.current_stats.update_time_accumulation.as_secs_f32() / self.current_stats.frame_count as f32;

            self.current_stats.frame_count = 0;
            self.current_stats.frame_time_accumulation = Duration::new(0, 0);
            self.current_stats.update_time_accumulation = Duration::new(0, 0);
            self.current_stats.last_measurement = Instant::now();
        }

        self.input.update();

        let mut was_resized = false;
        let size = self.get_size();

        for (_, event) in glfw::flush_messages(&self.events) {
            match event {
                glfw::WindowEvent::FramebufferSize(width, height) => {
                    unsafe { gl::Viewport(0, 0, width, height); }
                    was_resized = true;
                }
                _ => self.input.handle_event(event),
            }
        }

        let request_close: Option<bool>;
        let request_grab_cursor: Option<bool>;

        {
            let io = self.imgui.context.io_mut();
            io.delta_time = self.current_stats.delta_time;
            io.display_size = [size.0 as f32, size.1 as f32];
            self.input.apply_imgui_io(io, !self.is_cursor_grabbed);

            let ui = self.imgui.context.frame();

            let mut frame = Frame {
                input: &self.input,
                stats: &self.current_stats,
                was_resized,
                size,
                ui,
                request_close: None,
                request_grab_cursor: None,
            };
            f(&mut frame);

            request_close = frame.request_close;
            request_grab_cursor = frame.request_grab_cursor;

            self.imgui.renderer.render(frame.ui);
        }

        if let Some(true) = request_close {
            self.request_close();
        }
        if let Some(grab) = request_grab_cursor {
            self.request_grab_cursor(grab);
        }

        self.context.poll_events();

        let delta_time = self.current_stats.last_frame.elapsed();
        self.current_stats.update_time_accumulation += delta_time;

        self.window.borrow_mut().swap_buffers();
    }

    pub fn should_close(&self) -> bool {
        self.window.borrow().should_close()
    }

    pub fn request_close(&self) {
        self.window.borrow_mut().set_should_close(true);
    }

    pub fn request_grab_cursor(&mut self, grab: bool) {
        self.is_cursor_grabbed = grab;

        if grab {
            self.window.borrow_mut().set_cursor_mode(glfw::CursorMode::Disabled);
        } else {
            self.window.borrow_mut().set_cursor_mode(glfw::CursorMode::Normal);
        }
    }

    pub fn get_size(&self) -> (i32, i32) {
        self.window.borrow().get_size()
    }

    pub fn get_aspect(&self) -> f32 {
        let (w, h) = self.get_size();
        w as f32 / h as f32
    }
}

pub struct Frame<'window> {
    pub input: &'window Input,
    pub stats: &'window FrameStats,
    pub was_resized: bool,
    pub size: (i32, i32),
    pub ui: imgui::Ui<'window>,

    request_close: Option<bool>,
    request_grab_cursor: Option<bool>,
}

impl<'window> Frame<'window> {
    pub fn request_close(&mut self) {
        self.request_close = Some(true);
    }

    pub fn request_grab_cursor(&mut self, grab: bool) {
        self.request_grab_cursor = Some(grab);
    }

    pub fn get_aspect(&self) -> f32 {
        let (w, h) = self.size;
        w as f32 / h as f32
    }
}
