use std::cell::RefCell;
use std::sync::{mpsc, Mutex};
use std::time::{Duration, Instant};

use glfw::Context;
use once_cell::sync::Lazy;

use crate::core::imgui as imgui_wrapper;
use crate::core::Input;

pub struct Config {
    pub width: u32,
    pub height: u32,
    pub title: &'static str,
    pub msaa_samples: u32,
    pub headless: bool,
}

pub struct GlContext {
    window: glfw::Window,
    events: mpsc::Receiver<(f64, glfw::WindowEvent)>,
}

// GLFW_CONTEXT is represented as a singleton because it can only be created once per process.
// To allow multiple tests to initialise windows, for graphical testing in parallel, this is
// the only viable option of sharing this state without adding a custom test execution framework
// on top of rust's inbuilt one.
static GLFW_CONTEXT: Lazy<Mutex<glfw::Glfw>> = Lazy::new(|| {
    let mut context = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    context.window_hint(glfw::WindowHint::ContextVersion(4, 6));
    context.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));
    Mutex::new(context)
});

impl GlContext {
    pub fn new(cfg: Config) -> GlContext {
        let mut context = GLFW_CONTEXT.lock().unwrap();

        if cfg.msaa_samples > 0 {
            context.window_hint(glfw::WindowHint::Samples(Some(cfg.msaa_samples)));
        }
        if cfg.headless {
            context.window_hint(glfw::WindowHint::Visible(false));
        }

        let (mut window, events) = context
            .create_window(cfg.width, cfg.height, &cfg.title, glfw::WindowMode::Windowed)
            .expect("failed to create window");

        window.make_current();

        gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

        GlContext { window, events }
    }
}

pub struct Window {
    context: RefCell<GlContext>,
    imgui: imgui_wrapper::Wrapper,

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
    pub fn new(cfg: Config) -> Self {
        let mut context = GlContext::new(cfg);

        context.window.set_all_polling(true);
        context.window.set_cursor_mode(glfw::CursorMode::Disabled);

        let imgui = imgui_wrapper::Wrapper::new(&mut context.window);

        Window {
            context: RefCell::new(context),
            imgui,
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

        for (_, event) in glfw::flush_messages(&self.context.borrow().events) {
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

        GLFW_CONTEXT.lock().unwrap().poll_events();

        let delta_time = self.current_stats.last_frame.elapsed();
        self.current_stats.update_time_accumulation += delta_time;

        self.context.borrow_mut().window.swap_buffers();
    }

    pub fn should_close(&self) -> bool {
        self.context.borrow_mut().window.should_close()
    }

    pub fn request_close(&self) {
        self.context.borrow_mut().window.set_should_close(true);
    }

    pub fn request_grab_cursor(&mut self, grab: bool) {
        self.is_cursor_grabbed = grab;

        if grab {
            self.context.borrow_mut().window.set_cursor_mode(glfw::CursorMode::Disabled);
        } else {
            self.context.borrow_mut().window.set_cursor_mode(glfw::CursorMode::Normal);
        }
    }

    pub fn get_size(&self) -> (i32, i32) {
        self.context.borrow().window.get_size()
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
