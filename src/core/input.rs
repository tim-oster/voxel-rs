use std::collections::HashSet;

use cgmath::MetricSpace;

pub struct Input {
    pressed_keys: HashSet<glfw::Key>,
    released_keys: HashSet<glfw::Key>,
    last_key_modifiers: glfw::Modifiers,
    char_input_buffer: Vec<char>,

    last_mouse_pos: cgmath::Point2<f32>,
    mouse_delta: cgmath::Vector2<f32>,
    mouse_wheel_delta: f32,
    pressed_buttons: HashSet<glfw::MouseButton>,
    released_buttons: HashSet<glfw::MouseButton>,
    last_button_state: HashSet<glfw::MouseButton>,
}

impl Input {
    pub fn new() -> Input {
        Input {
            pressed_keys: HashSet::new(),
            released_keys: HashSet::new(),
            last_key_modifiers: glfw::Modifiers::empty(),
            char_input_buffer: Vec::new(),

            last_mouse_pos: cgmath::Point2::new(0.0, 0.0),
            mouse_delta: cgmath::Vector2::new(0.0, 0.0),
            mouse_wheel_delta: 0.0,
            pressed_buttons: HashSet::new(),
            released_buttons: HashSet::new(),
            last_button_state: HashSet::new(),
        }
    }

    pub fn update(&mut self) {
        self.released_keys.clear();
        self.last_key_modifiers = glfw::Modifiers::empty();
        self.char_input_buffer.clear();

        self.mouse_delta = cgmath::Vector2::new(0.0, 0.0);
        self.mouse_wheel_delta = 0.0;
        self.last_button_state = self.pressed_buttons.clone();
    }

    pub fn handle_event(&mut self, event: glfw::WindowEvent) {
        match event {
            glfw::WindowEvent::Key(key, _, action, modifiers) => match action {
                glfw::Action::Press => {
                    self.last_key_modifiers = modifiers;
                    self.pressed_keys.insert(key);
                    self.released_keys.remove(&key);
                }
                glfw::Action::Release => {
                    self.released_keys.insert(key);
                    self.pressed_keys.remove(&key);
                }
                _ => (),
            },
            glfw::WindowEvent::Char(character) => {
                self.char_input_buffer.push(character);
            }
            glfw::WindowEvent::CursorPos(x, y) => {
                let new_mouse_pos = cgmath::Point2::new(x as f32, y as f32);
                if self.last_mouse_pos.distance2(cgmath::Point2::new(0.0, 0.0)) > 0.0 {
                    self.mouse_delta = new_mouse_pos - self.last_mouse_pos;
                }
                self.last_mouse_pos = new_mouse_pos;
            }
            glfw::WindowEvent::Scroll(_, d) => {
                self.mouse_wheel_delta = d as f32;
            }
            glfw::WindowEvent::MouseButton(button, action, _) => match action {
                glfw::Action::Press => {
                    self.pressed_buttons.insert(button);
                    self.released_buttons.remove(&button);
                }
                glfw::Action::Release => {
                    self.released_buttons.insert(button);
                    self.pressed_buttons.remove(&button);
                }
                _ => (),
            },
            _ => (),
        }
    }

    pub fn is_key_pressed(&self, key: &glfw::Key) -> bool {
        self.pressed_keys.contains(key)
    }

    pub fn was_key_pressed(&self, key: &glfw::Key) -> bool {
        self.released_keys.contains(key)
    }

    pub fn is_button_pressed(&self, button: &glfw::MouseButton) -> bool {
        self.pressed_buttons.contains(button)
    }

    pub fn is_button_pressed_once(&self, button: &glfw::MouseButton) -> bool {
        self.pressed_buttons.contains(button) && !self.last_button_state.contains(button)
    }

    pub fn was_button_pressed(&self, button: &glfw::MouseButton) -> bool {
        self.released_buttons.contains(button)
    }

    pub fn get_mouse_delta(&self) -> cgmath::Vector2<f32> {
        self.mouse_delta
    }

    pub fn apply_imgui_io(&self, io: &mut imgui::Io, forward_input_events: bool) {
        if forward_input_events {
            io.mouse_pos = [self.last_mouse_pos.x, self.last_mouse_pos.y];
            io.mouse_delta = [self.mouse_delta.x, self.mouse_delta.y];
            io.mouse_wheel = self.mouse_wheel_delta;

            for ch in self.char_input_buffer.iter() {
                io.add_input_character(*ch);
            }
        } else {
            io.mouse_pos = [f32::MAX, f32::MAX];
            io.mouse_delta = [0.0, 0.0];
            io.mouse_wheel = 0.0;

            io.clear_input_characters();
        }

        for key in self.pressed_keys.iter() {
            io.keys_down[*key as usize] = forward_input_events;
        }
        for key in self.released_keys.iter() {
            io.keys_down[*key as usize] = false;
        }

        for button in self.pressed_buttons.iter() {
            let idx = *button as usize;
            if idx < io.mouse_down.len() {
                io.mouse_down[idx] = forward_input_events;
            }
        }
        for button in self.released_buttons.iter() {
            let idx = *button as usize;
            if idx < io.mouse_down.len() {
                io.mouse_down[idx] = false;
            }
        }

        let mut mods = self.last_key_modifiers;
        if !forward_input_events {
            mods = glfw::Modifiers::empty();
        }
        io.key_ctrl = mods.intersects(glfw::Modifiers::Control);
        io.key_alt = mods.intersects(glfw::Modifiers::Alt);
        io.key_shift = mods.intersects(glfw::Modifiers::Shift);
        io.key_super = mods.intersects(glfw::Modifiers::Super);
    }
}
