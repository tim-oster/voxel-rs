#![allow(dead_code)]

use std::ptr;

use gl::types::{GLint, GLuint};
use image::{DynamicImage, GenericImageView};

use crate::gl_assert_no_error;

pub struct Framebuffer {
    handle: GLuint,
    width: i32,
    height: i32,
}

/// Framebuffer is a wrapper around a OpenGL framebuffer object. It attaches color, depth & stencil
/// buffer for the given resolution. No multi-sampling is applied.
impl Framebuffer {
    pub fn new(width: i32, height: i32) -> Self {
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
        Self { handle, width, height }
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

    #[allow(clippy::unused_self)]
    pub fn unbind(&self) {
        unsafe { gl::BindFramebuffer(gl::FRAMEBUFFER, 0); }
    }

    #[allow(clippy::unused_self)]
    pub fn clear(&self, r: f32, g: f32, b: f32, a: f32) {
        unsafe {
            gl::ClearColor(r, g, b, a);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);
        }
    }

    pub fn read_pixels(&self) -> Vec<u8> {
        let mut bytes = vec![0; (self.width * self.height * 4) as usize];
        unsafe {
            gl::BindFramebuffer(gl::FRAMEBUFFER, self.handle);
            gl::ReadPixels(0, 0, self.width, self.height, gl::RGBA, gl::UNSIGNED_BYTE, ptr::addr_of_mut!(bytes[0]).cast());
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
        }
        bytes
    }

    pub fn as_image(&self) -> DynamicImage {
        let pixels = self.read_pixels();
        let image = image::RgbaImage::from_raw(self.width as u32, self.height as u32, pixels).unwrap();
        DynamicImage::ImageRgba8(image).flipv()
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe { gl::DeleteFramebuffers(1, &self.handle) }
    }
}

pub fn diff_images(lhs: &DynamicImage, rhs: &DynamicImage) -> f64 {
    // source: https://rosettacode.org/wiki/Percentage_difference_between_images#Rust
    fn diff_rgba3(rgba1: image::Rgba<u8>, rgba2: image::Rgba<u8>) -> i32 {
        (rgba1[0] as i32 - rgba2[0] as i32).abs()
            + (rgba1[1] as i32 - rgba2[1] as i32).abs()
            + (rgba1[2] as i32 - rgba2[2] as i32).abs()
    }

    let mut accum = 0;
    let zipper = lhs.pixels().zip(rhs.pixels());
    for (pixel1, pixel2) in zipper {
        accum += diff_rgba3(pixel1.2, pixel2.2);
    }
    accum as f64 / (255.0 * 3.0 * (lhs.width() * lhs.height()) as f64)
}
