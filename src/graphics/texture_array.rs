use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;

use gl::types::*;
use image::{GenericImageView, ImageError};

#[derive(Debug)]
pub enum TextureArrayError {
    ImageError(ImageError),
    Other(String),
}

impl From<ImageError> for TextureArrayError {
    fn from(err: ImageError) -> Self {
        TextureArrayError::ImageError(err)
    }
}

pub struct TextureArrayBuilder {
    mip_levels: u8,
    textures: HashMap<String, u32>,
    paths: Vec<String>,
}

impl TextureArrayBuilder {
    pub fn new(mip_levels: u8) -> TextureArrayBuilder {
        TextureArrayBuilder {
            mip_levels,
            textures: Default::default(),
            paths: Vec::new(),
        }
    }

    pub fn add_texture(&mut self, name: &str, path: &str) -> Result<&mut TextureArrayBuilder, TextureArrayError> {
        let name = String::from(name);
        if self.textures.get(&name).is_some() {
            return Err(TextureArrayError::Other(format!("name '{}' is already registered", name)));
        }
        self.textures.insert(name, self.paths.len() as u32);
        self.paths.push(String::from(path));
        Ok(self)
    }

    pub fn build(&self) -> Result<TextureArray, TextureArrayError> {
        let path = Path::new(&self.paths[0]);
        let mut image = image::open(&path)?;

        let textures = self.textures.clone();
        let mut texture = TextureArray::new(
            image.width(),
            image.height(),
            self.paths.len() as u32,
            self.mip_levels,
            textures,
        );

        texture.bind();
        for (i, path) in self.paths.iter().enumerate() {
            if i > 0 {
                // the first image was already loaded to fetch the dimensions of the array
                image = image::open(&path)?.flipv();
            }
            let data = image.to_rgba8().into_raw();
            texture.sub_image_3d(i as u32, image.width(), image.height(), &data);
        }
        if self.mip_levels > 1 {
            texture.generate_mipmaps();
        }
        texture.unbind();

        Ok(texture)
    }
}

pub struct TextureArray {
    gl_id: GLuint,
    textures: HashMap<String, u32>,
}

impl Drop for TextureArray {
    fn drop(&mut self) {
        unsafe { gl::DeleteTextures(1, &self.gl_id); }
    }
}

impl TextureArray {
    fn new(width: u32, height: u32, depth: u32, mip_levels: u8, textures: HashMap<String, u32>) -> TextureArray {
        let mut id = 0;

        unsafe {
            gl::GenTextures(1, &mut id);
            gl::BindTexture(gl::TEXTURE_2D_ARRAY, id);

            gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as GLint);
            gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_WRAP_R, gl::CLAMP_TO_EDGE as GLint);
            gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MIN_FILTER, gl::LINEAR_MIPMAP_LINEAR as GLint);
            gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MAG_FILTER, gl::NEAREST as GLint);

            gl::TexStorage3D(
                gl::TEXTURE_2D_ARRAY,
                mip_levels as GLint,
                gl::RGBA8,
                width as GLint,
                height as GLint,
                depth as GLint,
            );

            gl::BindTexture(gl::TEXTURE_2D_ARRAY, 0);
        }

        TextureArray { gl_id: id, textures }
    }

    pub fn sub_image_3d(&mut self, depth: u32, width: u32, height: u32, data: &Vec<u8>) {
        unsafe {
            gl::TexSubImage3D(
                gl::TEXTURE_2D_ARRAY,
                0,
                0,
                0,
                depth as GLint,
                width as GLint,
                height as GLint,
                1,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                &data[0] as *const u8 as *const c_void,
            );
        }
    }

    pub fn generate_mipmaps(&self) {
        unsafe { gl::GenerateMipmap(gl::TEXTURE_2D_ARRAY); }
    }

    pub fn bind(&self) {
        unsafe { gl::BindTexture(gl::TEXTURE_2D_ARRAY, self.gl_id) }
    }

    pub fn unbind(&self) {
        unsafe { gl::BindTexture(gl::TEXTURE_2D_ARRAY, 0) }
    }

    pub fn lookup(&self, name: &str) -> Option<u32> {
        if let Some(index) = self.textures.get(&String::from(name)) {
            return Some(*index);
        }
        None
    }
}
