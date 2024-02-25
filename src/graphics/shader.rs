#![allow(dead_code)]

use std::{io, ptr};
use std::cell::RefCell;
use std::ffi::CString;
use std::fmt::Write;
use std::path::Path;

use cgmath::{Array, Matrix};
use gl::types::{GLchar, GLenum, GLint, GLsizei, GLuint};
use indoc::formatdoc;
use regex::Regex;
use rustc_hash::FxHashMap;

use crate::core::assets;
use crate::graphics::resource::Bind;

#[derive(Debug)]
pub enum ShaderError {
    Io(io::Error),
    Compile(GlError),
    Link(GlError),
    Wrapped(Box<ShaderError>, String),
    Other(String),
}

impl From<io::Error> for ShaderError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
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
        status_fn(id, status_name, &mut success);
        if success == gl::TRUE as GLint {
            return None;
        }

        let mut length = 0;
        let mut info_log = [0; 512];
        log_fn(id, info_log.len() as GLsizei, &mut length, info_log.as_mut_ptr().cast());

        Some(Self {
            details: String::from_utf8_lossy(&info_log[..(length as usize)]).to_string(),
        })
    }
}

/// `ShaderProgramBuilder` allows for loading multiple GLSL source files and compiling them into one
/// OpenGL shader program.
pub struct ShaderProgramBuilder {
    shaders: FxHashMap<ShaderType, Shader>,
    include_cache: FxHashMap<String, String>,
}

impl ShaderProgramBuilder {
    pub fn new() -> Self {
        Self {
            shaders: FxHashMap::default(),
            include_cache: FxHashMap::default(),
        }
    }

    /// Reads the given file at `path` and assigns it to a shader of type `type`.
    ///
    /// Special directives:
    /// - `#inlucde "file.glsl"` can be used to include other file's contents
    /// - Adds `SHADER_COMPILE_TYPE` definition
    pub fn load_shader(&mut self, type_: ShaderType, path: &str) -> Result<&mut Self, ShaderError> {
        let mut src = self.load_file(path)?;
        if src.len() != 1 || !src.keys().next().unwrap().is_empty() {
            return Err(ShaderError::Other(String::from("file must not contain any #shader_type directives")));
        }
        self.add_shader(type_, src.remove("").unwrap())
    }

    /// Reads a shader bundle file that can define multiple shader types in one file.
    ///
    /// Special directives:
    /// - `#inlucde "file.glsl"` can be used to include other file's contents
    /// - `#shader_type <vertex|fragment|compute>` will use all lines until the next type directive for compiling the given shader type
    /// - Adds `SHADER_COMPILE_TYPE` definition
    pub fn load_shader_bundle(&mut self, path: &str) -> Result<&mut Self, ShaderError> {
        let src = self.load_file(path)?;
        for (type_, src) in src {
            let type_ = match type_.as_str() {
                "vertex" => ShaderType::Vertex,
                "fragment" => ShaderType::Fragment,
                "compute" => ShaderType::Compute,
                _ => {
                    return Err(ShaderError::Other(format!("unsupported shader type: {type_}")));
                }
            };
            self.add_shader(type_, src)?;
        }
        Ok(self)
    }

    fn load_file(&mut self, path: &str) -> Result<FxHashMap<String, String>, ShaderError> {
        let re_include = Regex::new("^#include\\s\"(.*)\"$").unwrap();

        let mut current_type = String::new();
        let mut shader_types = FxHashMap::default();

        let source = assets::read(path)?;
        let source = String::from_utf8_lossy(&source);
        for line in source.split('\n') {
            let line = line.trim_end();

            // handle shader type directives
            if line.starts_with("#shader_type") {
                let parts: Vec<_> = line.split(' ').collect();
                if parts.len() != 2 {
                    return Err(ShaderError::Other(format!(
                        "invalid shader type directive: {line}",
                    )));
                }

                current_type = parts[1].to_lowercase();
                if shader_types.contains_key(&current_type) {
                    return Err(ShaderError::Other(format!(
                        "same shader type used a second time: {line}",
                    )));
                }

                continue;
            }

            if !shader_types.contains_key(&current_type) {
                shader_types.insert(current_type.clone(), String::new());
            }
            let buffer = shader_types.get_mut(&current_type).unwrap();

            // handle include directives
            if line.starts_with("#include") {
                let caps = re_include.captures(line).unwrap();
                let rel_path = caps.get(1).map_or("", |m| m.as_str());

                // if given include path is not absolute, join include path to current shader's path
                let include_path = {
                    if Path::new(rel_path).is_absolute() {
                        rel_path.to_string()
                    } else {
                        let base_path = Path::new(path).parent().unwrap();
                        format!("{}/{}", base_path.to_str().unwrap(), rel_path)
                    }
                };

                self.write_included_file_to(&include_path, buffer)?;

                continue;
            }

            buffer.write_str(line).unwrap();
            buffer.write_char('\n').unwrap();
        }

        Ok(shader_types)
    }

    fn write_included_file_to(&mut self, path: &str, dst: &mut String) -> Result<(), ShaderError> {
        let include_src = self.include_cache.get(path);

        // use cache and prevent cyclic includes
        if let Some(ok) = include_src {
            if ok.is_empty() {
                return Err(ShaderError::Other(
                    format!("cyclic include of {path}"),
                ));
            }

            dst.write_str(ok).unwrap();
            dst.write_char('\n').unwrap();

            return Ok(());
        }

        // add empty string to detect cyclic includes
        self.include_cache.insert(String::from(path), String::new());

        // try loading the included file
        let src = self.load_file(path);
        if let Err(err) = src {
            return Err(ShaderError::Wrapped(
                Box::new(err),
                format!("while loading included file {path}"),
            ));
        }

        // error if any #shader_type directives were used
        let mut src = src.unwrap();
        if src.len() != 1 || !src.keys().next().unwrap().is_empty() {
            return Err(ShaderError::Other(
                format!("error including {path}: included files must not contain any #shader_type directives"),
            ));
        }

        // write to dst buffer
        let src = src.remove("").unwrap();
        dst.write_str(&src).unwrap();
        dst.write_char('\n').unwrap();

        // cache for next inclusion
        self.include_cache.insert(String::from(path), src);

        Ok(())
    }

    pub fn add_shader(&mut self, type_: ShaderType, src: String) -> Result<&mut Self, ShaderError> {
        if self.shaders.get(&type_).is_some() {
            return Err(ShaderError::Other(format!("type {type_:?} is already registered")));
        }

        let mut src = src;
        Self::inject_preprocessor_defines(&mut src, type_);

        let shader = Shader::new(type_, &src)?;
        self.shaders.insert(type_, shader);
        Ok(self)
    }

    fn inject_preprocessor_defines(src: &mut String, type_: ShaderType) {
        let mut offset = 0;
        if let Some(version_start) = src.find("#version") {
            if let Some(line_end) = src[version_start..].find('\n') {
                offset = version_start + line_end + 1;
            }
        }

        let inject = formatdoc! {r#"
            #define SHADER_TYPE_VERTEX      0
            #define SHADER_TYPE_FRAGMENT    1
            #define SHADER_TYPE_COMPUTE     2
            #define SHADER_COMPILE_TYPE     SHADER_TYPE_{}
        "#, type_.string()};
        src.insert_str(offset, &inject);
    }

    pub fn build(&self) -> Result<ShaderProgram, ShaderError> {
        let program = ShaderProgram {
            gl_id: unsafe { gl::CreateProgram() },
            uniform_location_cache: RefCell::new(FxHashMap::default()),
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

#[cfg(test)]
mod shader_program_builder_tests {
    use std::io::Write;
    use std::iter::FromIterator;

    use indoc::{formatdoc, indoc};
    use rustc_hash::FxHashMap;
    use tempfile::NamedTempFile;

    use crate::graphics::shader::{ShaderProgramBuilder, ShaderType};

    /// Tests if files with includes and other directives are loaded correctly without actually
    /// compiling the shader.
    #[test]
    fn load_file() {
        let mut include_file = NamedTempFile::new().unwrap();
        include_file.write_all(indoc! {r#"
            // test include
        "#}.as_bytes()).unwrap();
        let include_path = include_file.path().as_os_str().to_str().unwrap();

        let mut shader_file = NamedTempFile::new().unwrap();
        shader_file.write_all(formatdoc! {r#"
            #shader_type vertex
            #include "{}"
            void main() {{ gl_Position = vec4(position, 1.0); }}

            #shader_type fragment
            void main() {{ }}
        "#, include_path}.as_bytes()).unwrap();

        let mut builder = ShaderProgramBuilder::new();
        let shader_path = shader_file.path().as_os_str().to_str().unwrap();
        let result = builder.load_file(shader_path).unwrap();

        assert_eq!(result, FxHashMap::from_iter([
            ("vertex".to_string(), "// test include\n\n\nvoid main() { gl_Position = vec4(position, 1.0); }\n\n".to_string()),
            ("fragment".to_string(), "void main() { }\n\n".to_string()),
        ]));
        assert_eq!(builder.include_cache, FxHashMap::from_iter([
            (include_path.to_string(), "// test include\n\n".to_string()),
        ]));

        shader_file.close().unwrap();
        include_file.close().unwrap();
    }

    /// Tests if preprocessor defines are injected correctly into shader source.
    #[test]
    fn inject_preprocessor_defines() {
        let mut code = String::from(indoc! {r#"
            #version 450

            void main() {
                gl_Position = vec4(position, 1.0);
            }
        "#});

        ShaderProgramBuilder::inject_preprocessor_defines(&mut code, ShaderType::Vertex);

        assert_eq!(code, String::from(indoc! {r#"
            #version 450
            #define SHADER_TYPE_VERTEX      0
            #define SHADER_TYPE_FRAGMENT    1
            #define SHADER_TYPE_COMPUTE     2
            #define SHADER_COMPILE_TYPE     SHADER_TYPE_VERTEX

            void main() {
                gl_Position = vec4(position, 1.0);
            }
        "#}));
    }
}

pub struct ShaderProgram {
    gl_id: GLuint,
    uniform_location_cache: RefCell<FxHashMap<&'static str, GLint>>,
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

    #[allow(clippy::unused_self)]
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
    pub fn set_f32vec3s(&self, name: &'static str, values: &[cgmath::Vector3<f32>]) {
        unsafe {
            gl::Uniform3fv(self.get_uniform_location(name), values.len() as GLsizei, values.as_ptr().cast());
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

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub enum ShaderType {
    Vertex,
    Fragment,
    Compute,
}

impl ShaderType {
    fn string(self) -> &'static str {
        match self {
            Self::Vertex => "VERTEX",
            Self::Fragment => "FRAGMENT",
            Self::Compute => "COMPUTE",
        }
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

impl Shader {
    fn new(type_: ShaderType, src: &str) -> Result<Self, ShaderError> {
        let type_ = match type_ {
            ShaderType::Vertex => gl::VERTEX_SHADER,
            ShaderType::Fragment => gl::FRAGMENT_SHADER,
            ShaderType::Compute => gl::COMPUTE_SHADER,
        };
        let shader = Self {
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
