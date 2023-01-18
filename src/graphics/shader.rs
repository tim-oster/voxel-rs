use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CString;
use std::fmt::{format, Write};
use std::io::Error;
use std::ptr;

use cgmath::{Array, Matrix};
use gl::types::*;
use regex::Regex;

use crate::graphics::resource::Bind;
use crate::graphics::ShaderType::Compute;

#[derive(Debug)]
pub enum ShaderError {
    Io(std::io::Error),
    Compile(GlError),
    Link(GlError),
    Wrapped(Box<ShaderError>, String),
    Other(String),
}

impl From<std::io::Error> for ShaderError {
    fn from(err: Error) -> Self {
        ShaderError::Io(err)
    }
}

#[derive(Debug)]
pub struct GlError {
    details: String,
}

impl GlError {
    unsafe fn new(id: GLuint, status_name: GLenum,
                  status_fn: unsafe fn(GLuint, GLenum, *mut GLint),
                  log_fn: unsafe fn(GLuint, GLsizei, *mut GLsizei, *mut GLchar)) -> Option<Self> {
        let mut success = gl::FALSE as GLint;
        let mut length = 0;
        let mut info_log = [0; 512];

        status_fn(id, status_name, &mut success);
        if success == gl::TRUE as GLint {
            return None;
        }
        log_fn(id, info_log.len() as GLsizei, &mut length, info_log.as_mut_ptr() as *mut GLchar);
        println!("{:?}", status_name);
        Some(GlError {
            details: String::from_utf8_lossy(&info_log[..(length as usize)]).to_string(),
        })
    }
}

pub struct ShaderProgramBuilder {
    shaders: HashMap<ShaderType, Shader>,
    include_cache: HashMap<String, String>,
}

impl ShaderProgramBuilder {
    pub fn new() -> ShaderProgramBuilder {
        ShaderProgramBuilder {
            shaders: HashMap::new(),
            include_cache: HashMap::new(),
        }
    }

    pub fn load_shader(&mut self, type_: ShaderType, path: &str) -> Result<&mut ShaderProgramBuilder, ShaderError> {
        let src = self.load_file(path)?;
        if src.len() != 1 || !src.keys().next().unwrap().is_empty() {
            return Err(ShaderError::Other(String::from("file must not contain any #shader_type directives")));
        }
        self.add_shader(type_, src.get("").unwrap())
    }

    pub fn load_shader_bundle(&mut self, path: &str) -> Result<&mut ShaderProgramBuilder, ShaderError> {
        let src = self.load_file(path)?;
        for (type_, src) in src {
            let type_ = match type_.as_str() {
                "vertex" => ShaderType::Vertex,
                "fragment" => ShaderType::Fragment,
                "compute" => ShaderType::Compute,
                _ => {
                    return Err(ShaderError::Other(format!("unsupported shader type: {}", type_)));
                }
            };
            self.add_shader(type_, &src)?;
        }
        Ok(self)
    }

    fn load_file(&mut self, path: &str) -> Result<HashMap<String, String>, ShaderError> {
        let re_include = Regex::new("^#include\\s\"(.*)\"$").unwrap();

        let src = std::fs::read_to_string(path)?;
        let mut current_type = String::from("");
        let mut final_srcs = HashMap::new();

        for line in src.split("\n") {
            let line = line.trim_end();

            if line.starts_with("#shader_type") {
                let parts: Vec<_> = line.split(" ").collect();
                if parts.len() != 2 {
                    return Err(ShaderError::Other(format!(
                        "invalid shader type directive: {}", line,
                    )));
                }
                current_type = String::from(parts[1].to_lowercase());
                continue;
            }

            if !final_srcs.contains_key(&current_type) {
                final_srcs.insert(current_type.clone(), String::new());
            }
            let final_src = final_srcs.get_mut(&current_type).unwrap();

            if line.starts_with("#include") {
                let caps = re_include.captures(line).unwrap();
                let rel_path = caps.get(1).map_or("", |m| m.as_str());

                let base_path = std::path::Path::new(path).parent().unwrap();
                let include_path = base_path.join(rel_path);
                let include_path = include_path.to_str().unwrap();

                let include_src = self.include_cache.get(include_path);

                if let Some(ok) = include_src {
                    if ok.len() == 0 {
                        return Err(ShaderError::Other(format!(
                            "cyclic include of {} in {}",
                            rel_path, path,
                        )));
                    }

                    final_src.write_str(ok).unwrap();
                    final_src.write_char('\n').unwrap();
                } else {
                    // add empty string to detect cyclic includes
                    self.include_cache.insert(String::from(include_path), String::new());
                    let src = self.load_file(include_path);
                    if let Err(err) = src {
                        return Err(ShaderError::Wrapped(
                            Box::new(err),
                            format!("while including {} in {}", rel_path, path),
                        ));
                    }

                    let mut src = src.unwrap();
                    if src.len() != 1 || !src.keys().next().unwrap().is_empty() {
                        return Err(ShaderError::Other(
                            format!("error including {} in {}: included files must not contain any #shader_type directives", rel_path, path),
                        ));
                    }

                    let src = src.remove("").unwrap();
                    final_src.write_str(&src).unwrap();
                    final_src.write_char('\n').unwrap();

                    self.include_cache.insert(String::from(include_path), src);
                }

                continue;
            }

            final_src.write_str(line).unwrap();
            final_src.write_char('\n').unwrap();
        }

        Ok(final_srcs)
    }

    pub fn add_shader(&mut self, type_: ShaderType, src: &str) -> Result<&mut ShaderProgramBuilder, ShaderError> {
        if self.shaders.get(&type_).is_some() {
            return Err(ShaderError::Other(format!("type {:?} is already registered", type_)));
        }
        let shader = Shader::new(type_, src)?;
        self.shaders.insert(type_, shader);
        Ok(self)
    }

    pub fn build(&self) -> Result<ShaderProgram, ShaderError> {
        let program = ShaderProgram {
            gl_id: unsafe { gl::CreateProgram() },
            uniform_location_cache: RefCell::new(HashMap::new()),
        };
        for shader in self.shaders.values() {
            unsafe { gl::AttachShader(program.gl_id, shader.gl_id) }
        }
        unsafe {
            gl::LinkProgram(program.gl_id);
            if let Some(err) = GlError::new(program.gl_id, gl::LINK_STATUS, gl::GetProgramiv, gl::GetProgramInfoLog) {
                return Err(ShaderError::Link(err));
            }
        }
        Ok(program)
    }
}

pub struct ShaderProgram {
    gl_id: GLuint,
    uniform_location_cache: RefCell<HashMap<&'static str, GLint>>,
}

impl Drop for ShaderProgram {
    fn drop(&mut self) {
        unsafe { gl::DeleteProgram(self.gl_id) }
    }
}

impl ShaderProgram {
    pub fn bind(&self) {
        unsafe { gl::UseProgram(self.gl_id) }
    }

    pub fn unbind(&self) {
        unsafe { gl::UseProgram(0) }
    }

    //noinspection RsSelfConvention
    pub fn set_f32mat4(&self, name: &'static str, value: &cgmath::Matrix4<f32>) {
        unsafe {
            gl::UniformMatrix4fv(self.get_uniform_location(name), 1, gl::FALSE, value.as_ptr());
        }
    }

    //noinspection RsSelfConvention
    pub fn set_f32(&self, name: &'static str, value: f32) {
        unsafe {
            gl::Uniform1f(self.get_uniform_location(name), value);
        }
    }

    //noinspection RsSelfConvention
    pub fn set_f32vec2(&self, name: &'static str, value: &cgmath::Vector2<f32>) {
        unsafe {
            gl::Uniform2fv(self.get_uniform_location(name), 1, value.as_ptr());
        }
    }

    //noinspection RsSelfConvention
    pub fn set_f32vec3(&self, name: &'static str, value: &cgmath::Vector3<f32>) {
        unsafe {
            gl::Uniform3fv(self.get_uniform_location(name), 1, value.as_ptr());
        }
    }

    //noinspection RsSelfConvention
    pub fn set_f32vec3s(&self, name: &'static str, values: Vec<cgmath::Vector3<f32>>) {
        unsafe {
            gl::Uniform3fv(self.get_uniform_location(name), values.len() as GLsizei, values.as_ptr() as *const f32);
        }
    }

    //noinspection RsSelfConvention
    pub fn set_i32(&self, name: &'static str, value: i32) {
        unsafe {
            gl::Uniform1i(self.get_uniform_location(name), value);
        }
    }

    //noinspection RsSelfConvention
    pub fn set_texture<T: Bind>(&self, name: &'static str, slot: u8, texture: &T) {
        unsafe {
            gl::ActiveTexture(gl::TEXTURE0 + slot as GLenum);
            texture.bind();
        }
        self.set_i32(name, slot as i32);
    }

    fn get_uniform_location(&self, name: &'static str) -> GLint {
        let location = self.uniform_location_cache.borrow().get(name).copied();
        location.unwrap_or_else(|| {
            let cname = CString::new(name).unwrap();
            let value = unsafe { gl::GetUniformLocation(self.gl_id, cname.as_ptr()) };
            self.uniform_location_cache.borrow_mut().insert(name, value);
            value
        })
    }
}

struct Shader {
    gl_id: GLuint,
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe { gl::DeleteShader(self.gl_id) }
    }
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub enum ShaderType {
    Vertex,
    Fragment,
    Compute,
}

impl Shader {
    fn new(type_: ShaderType, src: &str) -> Result<Self, ShaderError> {
        let type_ = match type_ {
            ShaderType::Vertex => gl::VERTEX_SHADER,
            ShaderType::Fragment => gl::FRAGMENT_SHADER,
            ShaderType::Compute => gl::COMPUTE_SHADER,
        };
        let shader = Shader {
            gl_id: unsafe { gl::CreateShader(type_) },
        };
        unsafe {
            let src = CString::new(src).unwrap();
            gl::ShaderSource(shader.gl_id, 1, &src.as_ptr(), ptr::null());
            gl::CompileShader(shader.gl_id);
            if let Some(err) = GlError::new(shader.gl_id, gl::COMPILE_STATUS, gl::GetShaderiv, gl::GetShaderInfoLog) {
                return Err(ShaderError::Compile(err));
            }
        }
        Ok(shader)
    }
}
