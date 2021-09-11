use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CString;
use std::io::Error;
use std::ptr;

use cgmath::{Array, Matrix};
use gl::types::*;

#[derive(Debug)]
pub enum ShaderError {
    Io(std::io::Error),
    Compile(GlError),
    Link(GlError),
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

pub struct ShaderProgram<'a> {
    pub gl_id: GLuint,
    uniform_location_cache: RefCell<HashMap<&'a str, GLint>>,
}

impl Drop for ShaderProgram<'_> {
    fn drop(&mut self) {
        unsafe { gl::DeleteProgram(self.gl_id) }
    }
}

impl<'a> ShaderProgram<'a> {
    pub fn new_from_files(vert_path: &str, frag_path: &str) -> Result<Self, ShaderError> {
        let vert_src = std::fs::read_to_string(vert_path)?;
        let frag_src = std::fs::read_to_string(frag_path)?;
        Self::new_from_source(&vert_src, &frag_src)
    }

    pub fn new_from_source(vert_src: &str, frag_src: &str) -> Result<Self, ShaderError> {
        let vert_shader = Shader::new(vert_src, ShaderType::Vertex)?;
        let frag_shader = Shader::new(frag_src, ShaderType::Fragment)?;

        let program = ShaderProgram {
            gl_id: unsafe { gl::CreateProgram() },
            uniform_location_cache: RefCell::new(HashMap::new()),
        };
        unsafe {
            gl::AttachShader(program.gl_id, vert_shader.gl_id);
            gl::AttachShader(program.gl_id, frag_shader.gl_id);
            gl::LinkProgram(program.gl_id);
            if let Some(err) = GlError::new(program.gl_id, gl::LINK_STATUS, gl::GetProgramiv, gl::GetProgramInfoLog) {
                return Err(ShaderError::Link(err));
            }
        }
        Ok(program)
    }

    pub fn bind(&self) {
        unsafe { gl::UseProgram(self.gl_id) }
    }

    pub fn unbind(&self) {
        unsafe { gl::UseProgram(0) }
    }

    //noinspection RsSelfConvention
    pub fn set_f32mat4(&self, name: &'a str, value: &cgmath::Matrix4<f32>) {
        unsafe {
            gl::UniformMatrix4fv(self.get_uniform_location(name), 1, gl::FALSE, value.as_ptr());
        }
    }

    //noinspection RsSelfConvention
    pub fn set_f32(&self, name: &'a str, value: f32) {
        unsafe {
            gl::Uniform1f(self.get_uniform_location(name), value);
        }
    }

    //noinspection RsSelfConvention
    pub fn set_f32vec3(&self, name: &'a str, value: &cgmath::Vector3<f32>) {
        unsafe {
            gl::Uniform3fv(self.get_uniform_location(name), 1, value.as_ptr());
        }
    }

    //noinspection RsSelfConvention
    pub fn set_i32(&self, name: &'a str, value: i32) {
        unsafe {
            gl::Uniform1i(self.get_uniform_location(name), value);
        }
    }

    fn get_uniform_location(&self, name: &'a str) -> GLint {
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

enum ShaderType {
    Vertex,
    Fragment,
}

impl Shader {
    fn new(src: &str, type_: ShaderType) -> Result<Self, ShaderError> {
        let type_ = match type_ {
            ShaderType::Vertex => gl::VERTEX_SHADER,
            ShaderType::Fragment => gl::FRAGMENT_SHADER,
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
