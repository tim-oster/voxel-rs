use std::cell::RefCell;
use std::sync::{mpsc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use glfw::{Context, SwapInterval};
use once_cell::sync::Lazy;

use crate::core::imgui as imgui_wrapper;
use crate::core::Input;

pub struct Config {
    pub width: u32,
    pub height: u32,
    pub title: &'static str,
    pub msaa_samples: u32,
    pub headless: bool,
    pub resizable: bool,
    pub buffering: Buffering,
    pub target_fps: Option<u32>,
}

#[derive(Default)]
pub enum Buffering {
    Single,
    #[default]
    Double,
    Adaptive,
}

/// `GlContext` holds the native OpenGL rendering context for glfw as well as the associated event
/// queue. Unless for headless testing, creating it directly is discouraged. Use [`Window`]
/// instead.
pub struct GlContext {
    window: glfw::Window,
    events: mpsc::Receiver<(f64, glfw::WindowEvent)>,
}

/// True if `GL_ARB_texture_filter_anisotropic` extension is loaded.
pub static mut SUPPORTS_GL_ARB_TEXTURE_FILTER_ANISOTROPIC: bool = false;

// GLFW_CONTEXT is represented as a singleton because it can only be created once per process.
// To allow multiple tests to initialise windows, for graphical testing in parallel, this is
// the only viable option of sharing this state without adding a custom test execution framework
// on top of rust's inbuilt one.
static GLFW_CONTEXT: Lazy<Mutex<glfw::Glfw>> = Lazy::new(|| {
    let mut context = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    context.window_hint(glfw::WindowHint::ContextVersion(4, 5));
    context.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));
    Mutex::new(context)
});

#[allow(dead_code)]
impl GlContext {
    fn new(cfg: &Config) -> Self {
        let mut context = GLFW_CONTEXT.lock().unwrap();

        if cfg.msaa_samples > 0 {
            context.window_hint(glfw::WindowHint::Samples(Some(cfg.msaa_samples)));
        }
        context.window_hint(glfw::WindowHint::Visible(!cfg.headless));
        context.window_hint(glfw::WindowHint::Resizable(cfg.resizable));

        let (mut window, events) = context
            .create_window(cfg.width, cfg.height, cfg.title, glfw::WindowMode::Windowed)
            .expect("failed to create window");

        window.make_current();

        gl::load_with(|symbol| window.get_proc_address(symbol).cast());

        context.set_swap_interval(match cfg.buffering {
            Buffering::Single => SwapInterval::None,
            Buffering::Double => SwapInterval::Sync(1),
            Buffering::Adaptive => SwapInterval::Adaptive,
        });

        // apply OpenGL default settings
        unsafe {
            SUPPORTS_GL_ARB_TEXTURE_FILTER_ANISOTROPIC = context.extension_supported("GL_ARB_texture_filter_anisotropic");

            gl::Enable(gl::CULL_FACE);
            gl::CullFace(gl::BACK);
            gl::FrontFace(gl::CCW);

            if cfg.msaa_samples > 0 {
                gl::Enable(gl::MULTISAMPLE);
                gl::Enable(gl::SAMPLE_SHADING);
                gl::MinSampleShading(1.0);
            }
        }

        Self { window, events }
    }

    pub fn new_headless(width: u32, height: u32) -> Self {
        Self::new(&Config {
            width,
            height,
            title: "",
            msaa_samples: 0,
            headless: true,
            resizable: false,
            buffering: Buffering::Single,
            target_fps: None,
        })
    }
}

/// Window holds the native window in which OpenGL renders to. Additionally, it handles all
/// input events to that window.
pub struct Window {
    context: RefCell<GlContext>,
    imgui: imgui_wrapper::Wrapper,
    target_fps: Option<u32>,

    current_stats: FrameStats,
    is_cursor_grabbed: bool,
    input: Input,
    first_update: bool,
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
    pub fn new(cfg: &Config) -> Self {
        let target_fps = cfg.target_fps;
        let mut context = GlContext::new(cfg);

        context.window.set_all_polling(true);
        context.window.set_cursor_mode(glfw::CursorMode::Disabled);

        let imgui = imgui_wrapper::Wrapper::new(&context.window);

        Self {
            context: RefCell::new(context),
            imgui,
            target_fps,
            current_stats: FrameStats {
                last_frame: Instant::now(),
                last_measurement: Instant::now(),
                frame_count: 0,
                frame_time_accumulation: Duration::new(0, 0),
                update_time_accumulation: Duration::new(0, 0),

                delta_time: 0.0,
                frames_per_second: 0,
                avg_frame_time_per_second: 0.0,
                avg_update_time_per_second: 0.0,
            },
            is_cursor_grabbed: false,
            input: Input::new(),
            first_update: true,
        }
    }

    /// update should be called from the main run loop of the program. It handles all input
    /// processing and native OpenGL buffer swapping. If v-sync or other settings are enabled,
    /// it will handle the throttling of updates accordingly.
    /// Perform all rendering logic inside the passed function handle, as well as any input
    /// handling.
    pub fn update<F: FnOnce(&mut Frame)>(&mut self, f: F) {
        self.update_frame_state();
        let (size, was_resized) = self.handle_input_events();

        // run update function and handle imgui rendering
        let request_close: Option<bool>;
        let request_grab_cursor: Option<bool>;
        {
            let ui = self.imgui.context.new_frame();
            let mut frame = Frame {
                input: &self.input,
                stats: &self.current_stats,
                was_resized: was_resized || self.first_update,
                size,
                ui,
                is_cursor_grabbed: self.is_cursor_grabbed,
                request_close: None,
                request_grab_cursor: None,
            };
            f(&mut frame);

            request_close = frame.request_close;
            request_grab_cursor = frame.request_grab_cursor;

            self.imgui.renderer.render(&mut self.imgui.context);
        }
        if request_close == Some(true) {
            self.request_close();
        }
        if let Some(grab) = request_grab_cursor {
            self.request_grab_cursor(grab);
        }

        GLFW_CONTEXT.lock().unwrap().poll_events();

        // measure update timing
        let delta_time = self.current_stats.last_frame.elapsed();
        self.current_stats.update_time_accumulation += delta_time;

        self.context.borrow_mut().window.swap_buffers();
        self.first_update = false;

        // if enabled, limit fps to target
        if let Some(target) = self.target_fps {
            let target_delta = 1.0 / f64::from(target);
            let actual_delta = self.current_stats.last_frame.elapsed().as_secs_f64();
            let diff = target_delta - actual_delta;
            if diff > 0.0 {
                thread::sleep(Duration::from_secs_f64(diff));
            }
        }
    }

    fn update_frame_state(&mut self) {
        let delta_time = self.current_stats.last_frame.elapsed();
        self.current_stats.frame_time_accumulation += delta_time;
        self.current_stats.delta_time = delta_time.as_secs_f32();
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
    }

    fn handle_input_events(&mut self) -> ((i32, i32), bool) {
        self.input.update();

        let mut was_resized = false;

        for (_, event) in glfw::flush_messages(&self.context.borrow().events) {
            match event {
                glfw::WindowEvent::FramebufferSize(width, height) => {
                    unsafe { gl::Viewport(0, 0, width, height); }
                    was_resized = true;
                }
                _ => self.input.handle_event(&event),
            }
        }

        let size = self.get_size();
        let io = self.imgui.context.io_mut();
        io.delta_time = self.current_stats.delta_time;
        io.display_size = [size.0 as f32, size.1 as f32];

        self.input.apply_imgui_io(io, !self.is_cursor_grabbed);

        (size, was_resized)
    }

    /// `should_close` returns true, if a close was requested on this window.
    pub fn should_close(&self) -> bool {
        self.context.borrow_mut().window.should_close()
    }

    /// `request_close` will cause the window to close in the next update cycle.
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

    /// `get_size` returns the current window's width and height in pixels.
    pub fn get_size(&self) -> (i32, i32) {
        self.context.borrow().window.get_size()
    }

    /// `get_aspect` returns the current window's aspect ration in (width / height).
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
    pub ui: &'window mut imgui::Ui,

    is_cursor_grabbed: bool,
    request_close: Option<bool>,
    request_grab_cursor: Option<bool>,
}

impl<'window> Frame<'window> {
    pub fn request_close(&mut self) {
        self.request_close = Some(true);
    }

    pub fn is_cursor_grabbed(&self) -> bool {
        self.is_cursor_grabbed
    }

    pub fn request_grab_cursor(&mut self, grab: bool) {
        self.request_grab_cursor = Some(grab);
    }

    pub fn get_aspect(&self) -> f32 {
        let (w, h) = self.size;
        w as f32 / h as f32
    }
}
