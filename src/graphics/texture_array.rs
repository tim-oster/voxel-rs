use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;

use gl::types::*;
use image::{DynamicImage, GenericImageView, ImageError};

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

enum ImageContent {
    File(String),
    RGB8(u32, u32, Vec<u8>),
}

pub struct TextureArrayBuilder {
    mip_levels: u8,
    textures: HashMap<String, u32>,
    content: Vec<ImageContent>,
}

impl TextureArrayBuilder {
    pub fn new(mip_levels: u8) -> TextureArrayBuilder {
        TextureArrayBuilder {
            mip_levels,
            textures: Default::default(),
            content: Vec::new(),
        }
    }

    pub fn add_file(&mut self, name: &str, path: &str) -> Result<&mut TextureArrayBuilder, TextureArrayError> {
        self.register_texture(name)?;
        self.content.push(ImageContent::File(String::from(path)));
        Ok(self)
    }

    pub fn add_rgba8(&mut self, name: &str, w: u32, h: u32, bytes: Vec<u8>) -> Result<&mut TextureArrayBuilder, TextureArrayError> {
        self.register_texture(name)?;
        self.content.push(ImageContent::RGB8(w, h, bytes));
        Ok(self)
    }

    fn register_texture(&mut self, name: &str) -> Result<(), TextureArrayError> {
        let name = String::from(name);
        if self.textures.get(&name).is_some() {
            return Err(TextureArrayError::Other(format!("name '{}' is already registered", name)));
        }
        self.textures.insert(name, self.content.len() as u32);
        Ok(())
    }

    pub fn build(&self) -> Result<TextureArray, TextureArrayError> {
        let width;
        let height;
        let mut image = None;

        match &self.content[0] {
            ImageContent::File(path) => {
                let path = Path::new(&path);
                image = Some(image::open(&path)?);
                width = image.as_ref().unwrap().width();
                height = image.as_ref().unwrap().height();
            }
            ImageContent::RGB8(w, h, _) => {
                width = *w;
                height = *h;
            }
        }

        let textures = self.textures.clone();
        let mut texture = TextureArray::new(
            width,
            height,
            self.content.len() as u32,
            self.mip_levels,
            textures,
        );

        texture.bind();
        for (i, content) in self.content.iter().enumerate() {
            match content {
                ImageContent::File(path) => {
                    if i > 0 {
                        // the first image was already loaded to fetch the dimensions of the array
                        image = Some(image::open(&path)?.flipv());
                    }
                    let image = image.as_ref().unwrap();
                    let data = image.to_rgba8().into_raw();
                    texture.sub_image_3d(i as u32, image.width(), image.height(), &data);
                }
                ImageContent::RGB8(w, h, bytes) => {
                    texture.sub_image_3d(i as u32, *w, *h, bytes);
                }
            }
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
