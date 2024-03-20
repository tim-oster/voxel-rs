/// This module is inspired by [glfw-rs](https://github.com/K4ugummi/imgui-glfw-rs).

use std::ffi::{c_void, CStr, CString};

use crate::core::imgui_opengl;

struct GlfwClipboardBackend(*mut c_void);

impl imgui::ClipboardBackend for GlfwClipboardBackend {
    fn get(&mut self) -> Option<String> {
        let char_ptr = unsafe { glfw::ffi::glfwGetClipboardString(self.0.cast()) };
        let c_str = unsafe { CStr::from_ptr(char_ptr) };
        Some(c_str.to_str().unwrap().to_owned())
    }

    fn set(&mut self, value: &str) {
        let c_str = CString::new(value).unwrap();
        unsafe {
            glfw::ffi::glfwSetClipboardString(self.0.cast(), c_str.as_ptr());
        };
    }
}

pub struct Wrapper {
    pub context: imgui::Context,
    pub renderer: imgui_opengl::Renderer,
}

impl Wrapper {
    pub fn new(window: &glfw::Window) -> Self {
        let mut context = imgui::Context::create();

        let io = context.io_mut();
        io.key_map[imgui::Key::Tab as usize] = glfw::Key::Tab as u32;
        io.key_map[imgui::Key::LeftArrow as usize] = glfw::Key::Left as u32;
        io.key_map[imgui::Key::RightArrow as usize] = glfw::Key::Right as u32;
        io.key_map[imgui::Key::UpArrow as usize] = glfw::Key::Up as u32;
        io.key_map[imgui::Key::DownArrow as usize] = glfw::Key::Down as u32;
        io.key_map[imgui::Key::PageUp as usize] = glfw::Key::PageUp as u32;
        io.key_map[imgui::Key::PageDown as usize] = glfw::Key::PageDown as u32;
        io.key_map[imgui::Key::Home as usize] = glfw::Key::Home as u32;
        io.key_map[imgui::Key::End as usize] = glfw::Key::End as u32;
        io.key_map[imgui::Key::Insert as usize] = glfw::Key::Insert as u32;
        io.key_map[imgui::Key::Delete as usize] = glfw::Key::Delete as u32;
        io.key_map[imgui::Key::Backspace as usize] = glfw::Key::Backspace as u32;
        io.key_map[imgui::Key::Space as usize] = glfw::Key::Space as u32;
        io.key_map[imgui::Key::Enter as usize] = glfw::Key::Enter as u32;
        io.key_map[imgui::Key::Escape as usize] = glfw::Key::Escape as u32;
        io.key_map[imgui::Key::A as usize] = glfw::Key::A as u32;
        io.key_map[imgui::Key::C as usize] = glfw::Key::C as u32;
        io.key_map[imgui::Key::V as usize] = glfw::Key::V as u32;
        io.key_map[imgui::Key::X as usize] = glfw::Key::X as u32;
        io.key_map[imgui::Key::Y as usize] = glfw::Key::Y as u32;
        io.key_map[imgui::Key::Z as usize] = glfw::Key::Z as u32;

        unsafe {
            let ptr = glfw::ffi::glfwGetCurrentContext().cast();
            context.set_clipboard_backend(GlfwClipboardBackend(ptr));
        }

        let renderer = imgui_opengl::Renderer::new(&mut context);
        Self {
            context,
            renderer,
        }
    }
}
