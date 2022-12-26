use std::mem;
use std::ops::{Deref, DerefMut};

use gl::types::{GLenum, GLsizeiptr, GLuint, GLvoid};

// doc: https://registry.khronos.org/OpenGL-Refpages/gl4/html/glBufferData.xhtml
type BufferUsage = u32;

pub const STREAM_READ: BufferUsage = gl::STREAM_READ;
pub const STREAM_DRAW: BufferUsage = gl::STREAM_DRAW;
pub const STREAM_COPY: BufferUsage = gl::STREAM_COPY;
pub const STATIC_READ: BufferUsage = gl::STATIC_READ;
pub const STATIC_DRAW: BufferUsage = gl::STATIC_DRAW;
pub const STATIC_COPY: BufferUsage = gl::STATIC_COPY;
pub const DYNAMIC_READ: BufferUsage = gl::DYNAMIC_READ;
pub const DYNAMIC_DRAW: BufferUsage = gl::DYNAMIC_DRAW;
pub const DYNAMIC_COPY: BufferUsage = gl::DYNAMIC_COPY;

pub struct Buffer<T> {
    data: T,
    handle: GLuint,
}

impl<T> Deref for Buffer<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self. handle);
        }
    }
}

impl<T> Buffer<T> {
    pub fn new(data: T, usage: BufferUsage) -> Buffer<T> {
        let mut handle = 0;
        unsafe {
            gl::CreateBuffers(1, &mut handle);
            gl::NamedBufferData(
                handle,
                mem::size_of::<T>() as GLsizeiptr,
                &data as *const T as *const GLvoid,
                usage,
            );
        }
        Buffer { data, handle }
    }

    pub fn pull_data(&mut self) {
        unsafe {
            gl::GetNamedBufferSubData(
                self.handle,
                0,
                mem::size_of::<T>() as GLsizeiptr,
                &mut self.data as *mut T as *mut GLvoid,
            );
        }
    }

    pub fn bind_base_as_ssbo(&self, index: u32) {
        unsafe {
            gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, index, self.handle);
        }
    }
}
