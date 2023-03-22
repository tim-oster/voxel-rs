use std::ffi::c_void;
use std::{mem, ptr};

use gl::types::{GLint, GLsizeiptr, GLuint};

use crate::graphics::resource::Bind;
use crate::graphics::macros::{AlignedPoint2, AlignedPoint3, AlignedVec3};

#[repr(C)]
struct Vertex {
    position: AlignedPoint3<f32>,
    uv: AlignedPoint2<f32>,
    normal: AlignedVec3<f32>,
}

pub struct ScreenQuad {
    vao: GLuint,
    vbo: GLuint,
    ebo: GLuint,
}

impl ScreenQuad {
    pub fn new() -> ScreenQuad {
        let vertices = vec![
            Vertex { position: AlignedPoint3::new(1.0, 1.0, -1.0), uv: AlignedPoint2::new(1.0, 1.0), normal: AlignedVec3::new(0.0, 0.0, 1.0) },
            Vertex { position: AlignedPoint3::new(-1.0, 1.0, -1.0), uv: AlignedPoint2::new(0.0, 1.0), normal: AlignedVec3::new(0.0, 0.0, 1.0) },
            Vertex { position: AlignedPoint3::new(-1.0, -1.0, -1.0), uv: AlignedPoint2::new(0.0, 0.0), normal: AlignedVec3::new(0.0, 0.0, 1.0) },
            Vertex { position: AlignedPoint3::new(1.0, -1.0, -1.0), uv: AlignedPoint2::new(1.0, 0.0), normal: AlignedVec3::new(0.0, 0.0, 1.0) },
        ];
        let indices = vec![0i32, 1, 3, 1, 2, 3];

        unsafe {
            let (mut vao, mut vbo, mut ebo) = (0, 0, 0);
            gl::GenVertexArrays(1, &mut vao);
            gl::GenBuffers(1, &mut vbo);
            gl::GenBuffers(1, &mut ebo);

            gl::BindVertexArray(vao);

            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (vertices.len() * mem::size_of::<Vertex>()) as GLsizeiptr,
                &vertices[0] as *const Vertex as *const c_void,
                gl::STATIC_DRAW,
            );

            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
            gl::BufferData(
                gl::ELEMENT_ARRAY_BUFFER,
                (indices.len() * mem::size_of::<GLint>()) as GLsizeiptr,
                &indices[0] as *const i32 as *const c_void,
                gl::STATIC_DRAW,
            );

            let stride = mem::size_of::<Vertex>() as i32;

            gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, stride, offset_of!(Vertex, position) as *const c_void);
            gl::EnableVertexAttribArray(0);

            gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, stride, offset_of!(Vertex, uv) as *const c_void);
            gl::EnableVertexAttribArray(1);

            gl::VertexAttribPointer(2, 3, gl::FLOAT, gl::FALSE, stride, offset_of!(Vertex, normal) as *const c_void);
            gl::EnableVertexAttribArray(2);

            gl::BindBuffer(gl::ARRAY_BUFFER, 0);
            gl::BindVertexArray(0);

            ScreenQuad { vao, vbo, ebo }
        }
    }

    pub fn render(&self) {
        unsafe {
            gl::BindVertexArray(self.vao);
            gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, ptr::null());
            gl::BindVertexArray(0);
        }
    }
}

impl Drop for ScreenQuad {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteVertexArrays(1, &self.vao);
            gl::DeleteBuffers(1, &self.vbo);
            gl::DeleteBuffers(1, &self.ebo);
        }
    }
}

impl Bind for ScreenQuad {
    fn bind(&self) {
        unsafe { gl::BindVertexArray(self.vao); }
    }

    fn unbind(&self) {
        unsafe { gl::BindVertexArray(0); }
    }
}
