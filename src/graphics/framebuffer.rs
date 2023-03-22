#![allow(dead_code)]

use std::ptr;

use gl::types::{GLint, GLuint, GLvoid};

use crate::gl_assert_no_error;

pub struct Framebuffer {
    handle: GLuint,
    width: i32,
    height: i32,
}

impl Framebuffer {
    pub fn new(width: i32, height: i32) -> Framebuffer {
        let mut handle = 0;
        unsafe {
            gl::GenFramebuffers(1, &mut handle);
            gl::BindFramebuffer(gl::FRAMEBUFFER, handle);

            let mut color_attachment = 0;
            gl::GenTextures(1, &mut color_attachment);
            gl::BindTexture(gl::TEXTURE_2D, color_attachment);
            gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGBA8 as GLint, width, height, 0, gl::RGBA, gl::UNSIGNED_BYTE, ptr::null());
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as GLint);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as GLint);
            gl::BindTexture(gl::TEXTURE_2D, 0);
            gl::FramebufferTexture2D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, color_attachment, 0);
            gl_assert_no_error!();

            let mut depth_stencil_attachment = 0;
            gl::GenRenderbuffers(1, &mut depth_stencil_attachment);
            gl::BindRenderbuffer(gl::RENDERBUFFER, depth_stencil_attachment);
            gl::RenderbufferStorage(gl::RENDERBUFFER, gl::DEPTH24_STENCIL8, width, height);
            gl::BindRenderbuffer(gl::RENDERBUFFER, 0);
            gl::FramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_STENCIL_ATTACHMENT, gl::RENDERBUFFER, depth_stencil_attachment);
            gl_assert_no_error!();

            assert_eq!(gl::CheckFramebufferStatus(gl::FRAMEBUFFER), gl::FRAMEBUFFER_COMPLETE);

            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
        }
        Framebuffer { handle, width, height }
    }

    pub fn width(&self) -> i32 {
        self.width
    }

    pub fn height(&self) -> i32 {
        self.height
    }

    pub fn bind(&self) {
        unsafe { gl::BindFramebuffer(gl::FRAMEBUFFER, self.handle); }
    }

    pub fn unbind(&self) {
        unsafe { gl::BindFramebuffer(gl::FRAMEBUFFER, 0); }
    }

    pub fn read_pixels(&self) -> Vec<u8> {
        let mut bytes = vec![0; (self.width * self.height * 4) as usize];
        unsafe {
            gl::BindFramebuffer(gl::FRAMEBUFFER, self.handle);
            gl::ReadPixels(0, 0, self.width, self.height, gl::RGBA, gl::UNSIGNED_BYTE, &mut bytes[0] as *mut u8 as *mut GLvoid);
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
        }
        bytes
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe { gl::DeleteFramebuffers(1, &self.handle) }
    }
}