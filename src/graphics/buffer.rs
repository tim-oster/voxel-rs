#![allow(dead_code)]

use std::{mem, ptr};
use std::ops::{Deref, DerefMut};

use gl::types::{GLsizeiptr, GLuint};

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

/// Buffer is a wrapper around a native OpenGL buffer.
pub struct Buffer<T> {
    data: Option<Vec<T>>,
    handle: GLuint,
}

impl<T> Deref for Buffer<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        self.data.as_ref().unwrap()
    }
}

impl<T> DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.as_mut().unwrap()
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.handle);
        }
    }
}

impl<T> Buffer<T> {
    pub fn new(data: Vec<T>, usage: BufferUsage) -> Self {
        let mut handle = 0;
        unsafe {
            gl::CreateBuffers(1, &mut handle);
            gl::NamedBufferData(
                handle,
                (mem::size_of::<T>() * data.len()) as GLsizeiptr,
                ptr::addr_of!(data[0]).cast(),
                usage,
            );
        }
        Self { data: Some(data), handle }
    }

    pub fn pull_data(&mut self) {
        unsafe {
            gl::GetNamedBufferSubData(
                self.handle,
                0,
                (mem::size_of::<T>() * self.data.as_ref().unwrap().len()) as GLsizeiptr,
                (self.data.as_mut().unwrap().get_mut(0).unwrap() as *mut T).cast(),
            );
        }
    }

    pub fn bind_as_storage_buffer(&self, index: u32) {
        unsafe {
            gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, index, self.handle);
        }
    }

    pub fn take(mut self) -> Vec<T> {
        self.data.take().unwrap()
    }
}

/// `MappedBuffer` is a wrapper for a persistently mapped OpenGL buffer. Both client and server
/// side changes are reflected in the buffer without pulling or flushing. For synchronizing
/// CPU & GPU, look into `Fences` and `Memory Barriers`.
pub struct MappedBuffer<T> {
    handle: GLuint,
    len: usize,
    mapped_ptr: *mut T,
}

impl<T> Drop for MappedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.handle);
        }
    }
}

impl<T> Deref for MappedBuffer<T> {
    type Target = *mut T;

    fn deref(&self) -> &Self::Target {
        &self.mapped_ptr
    }
}

impl<T> DerefMut for MappedBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.mapped_ptr
    }
}

impl<T> MappedBuffer<T> {
    pub fn new(len: usize) -> Self {
        let mut handle = 0;
        let mapped_ptr;
        unsafe {
            gl::CreateBuffers(1, &mut handle);

            let size_bytes = mem::size_of::<T>() * len;
            gl::NamedBufferStorage(
                handle,
                size_bytes as GLsizeiptr,
                ptr::null(),
                gl::MAP_READ_BIT | gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT,
            );
            mapped_ptr = gl::MapNamedBufferRange(
                handle,
                0,
                size_bytes as isize,
                gl::MAP_READ_BIT | gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT,
            ).cast();
        }
        Self { handle, len, mapped_ptr }
    }

    pub fn bind_as_storage_buffer(&self, index: u32) {
        unsafe {
            gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, index, self.handle);
        }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn as_slice_mut(&self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.mapped_ptr, self.len) }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.mapped_ptr, self.len) }
    }

    pub fn size_in_bytes(&self) -> usize {
        mem::size_of::<T>() * self.len
    }

    pub fn len(&self) -> usize {
        self.len
    }
}