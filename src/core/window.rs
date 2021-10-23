extern crate glfw;

use std::cell::RefCell;
use std::collections::HashSet;
use std::time::{Instant, Duration};

use cgmath::MetricSpace;
use glfw::Context;

pub struct Window {
    context: glfw::Glfw,
    window: RefCell<glfw::Window>,
    events: std::sync::mpsc::Receiver<(f64, glfw::WindowEvent)>,

    was_resized: bool,
    current_stats: FrameStats,
    input: Input,
}

pub struct FrameStats {
    last_frame: Instant,
    last_measurement: Instant,
    frame_count: i32,
    frame_time_accumulation: Duration,
    pub delta_time: f32,
}

impl Window {
    pub fn new(width: u32, height: u32, title: &str) -> Self {
        let mut context = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
        context.window_hint(glfw::WindowHint::ContextVersion(4, 6));
        context.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));

        let (mut window, events) = context
            .create_window(width, height, title, glfw::WindowMode::Windowed)
            .expect("failed to create window");

        window.make_current();
        window.set_key_polling(true);
        window.set_cursor_pos_polling(true);
        window.set_scroll_polling(true);
        window.set_mouse_button_polling(true);
        window.set_framebuffer_size_polling(true);

        window.set_cursor_mode(glfw::CursorMode::Disabled);

        gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

        Window {
            context,
            window: RefCell::new(window),
            events,
            was_resized: false,
            current_stats: FrameStats {
                last_frame: Instant::now(),
                last_measurement: Instant::now(),
                frame_count: 0,
                frame_time_accumulation: Duration::new(0, 0),
                delta_time: 1.0,
            },
            input: Input::new(),
        }
    }

    pub fn update(&mut self) -> bool {
        if self.window.borrow().should_close() {
            return false;
        }

        let delta_time = self.current_stats.last_frame.elapsed();
        self.current_stats.frame_time_accumulation += delta_time;
        self.current_stats.delta_time = (delta_time.as_secs_f32() / (1.0 / 60.0));
        self.current_stats.last_frame = Instant::now();

        self.current_stats.frame_count += 1;
        if self.current_stats.last_measurement.elapsed() > Duration::from_secs(1) {
            // TODO move to main?
            let frame_time = self.current_stats.frame_time_accumulation.as_secs_f32() / self.current_stats.frame_count as f32;
            println!("frames: {}, frame time: {}ms", self.current_stats.frame_count, frame_time * 1000.0);

            self.current_stats.frame_count = 0;
            self.current_stats.frame_time_accumulation = Duration::new(0, 0);
            self.current_stats.last_measurement = Instant::now();
        }

        self.was_resized = false;
        self.input.update();

        for (_, event) in glfw::flush_messages(&self.events) {
            match event {
                glfw::WindowEvent::FramebufferSize(width, height) => {
                    unsafe { gl::Viewport(0, 0, width, height); }
                    self.was_resized = true;
                }
                _ => self.input.handle_event(event),
            }
        }

        self.context.poll_events();
        self.window.borrow_mut().swap_buffers();

        true
    }

    pub fn was_resized(&self) -> bool {
        self.was_resized
    }

    pub fn get_frame_stats(&self) -> &FrameStats {
        &self.current_stats
    }

    pub fn get_input(&self) -> &Input {
        &self.input
    }

    pub fn close(&self) {
        self.window.borrow_mut().set_should_close(true);
    }

    //noinspection RsSelfConvention
    pub fn set_grab_cursor(&self, grab: bool) {
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

pub struct Input {
    key_states: HashSet<glfw::Key>,
    old_key_states: HashSet<glfw::Key>,
    last_mouse_pos: cgmath::Point2<f32>,
    mouse_delta: cgmath::Vector2<f32>,
}

impl Input {
    fn new() -> Input {
        Input {
            key_states: HashSet::new(),
            old_key_states: HashSet::new(),
            last_mouse_pos: cgmath::Point2::new(0.0, 0.0),
            mouse_delta: cgmath::Vector2::new(0.0, 0.0),
        }
    }

    fn update(&mut self) {
        self.mouse_delta = cgmath::Vector2::new(0.0, 0.0);
        self.old_key_states = self.key_states.clone();
    }

    fn handle_event(&mut self, event: glfw::WindowEvent) {
        match event {
            glfw::WindowEvent::Key(key, _, action, _) => match action {
                glfw::Action::Press => {
                    self.key_states.insert(key);
                }
                glfw::Action::Release => {
                    self.key_states.remove(&key);
                }
                _ => (),
            },
            glfw::WindowEvent::CursorPos(x, y) => {
                let new_mouse_pos = cgmath::Point2::new(x as f32, y as f32);
                if self.last_mouse_pos.distance2(cgmath::Point2::new(0.0, 0.0)) > 0.0 {
                    self.mouse_delta = new_mouse_pos - self.last_mouse_pos;
                }
                self.last_mouse_pos = new_mouse_pos;
            }
            _ => (),
        }
    }

    pub fn is_key_pressed(&self, key: &glfw::Key) -> bool {
        self.key_states.contains(key)
    }

    pub fn was_key_pressed(&self, key: &glfw::Key) -> bool {
        !self.key_states.contains(key) && self.old_key_states.contains(key)
    }

    pub fn get_mouse_delta(&self) -> cgmath::Vector2<f32> {
        self.mouse_delta
    }
}
