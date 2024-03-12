#![allow(dead_code)]

use std::io;

use gl::types::{GLenum, GLint, GLuint};
use image::{GenericImageView, ImageError, ImageFormat};
use rustc_hash::FxHashMap;

use crate::core::assets;
use crate::gl_assert_no_error;
use crate::graphics::resource::Bind;

#[derive(Debug)]
pub enum TextureArrayError {
    Io(io::Error),
    ImageError(ImageError),
    Other(String),
}

impl From<io::Error> for TextureArrayError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<ImageError> for TextureArrayError {
    fn from(err: ImageError) -> Self {
        Self::ImageError(err)
    }
}

enum ImageContent {
    /// File(path)
    File(String),
    /// RGB8(width, height, data)
    RGB8(u32, u32, Vec<u8>),
}

/// `TextureArrayBuilder` allows for combining multiple `ImageContent` values into one texture array.
/// The first `ImageContent` object decides the resolution of the texture array. All other images
/// must have the same dimensions. Adding a new image requires specifying a unique name which
/// can later be used, to lookup the texture's index in the array.
pub struct TextureArrayBuilder {
    mip_levels: u8,
    max_anisotropy: f32,
    textures: FxHashMap<String, u32>,
    content: Vec<ImageContent>,
}

impl TextureArrayBuilder {
    pub fn new(mip_levels: u8, max_anisotropy: f32) -> Self {
        Self {
            mip_levels,
            max_anisotropy,
            textures: FxHashMap::default(),
            content: Vec::new(),
        }
    }

    pub fn add_file(&mut self, name: &str, path: &str) -> Result<&mut Self, TextureArrayError> {
        self.register_texture(name.to_owned())?;
        self.content.push(ImageContent::File(path.to_owned()));
        Ok(self)
    }

    pub fn add_rgba8(&mut self, name: &str, w: u32, h: u32, bytes: Vec<u8>) -> Result<&mut Self, TextureArrayError> {
        let mut bytes = bytes;
        Self::flip_image_v(&mut bytes, w, h, 4);

        self.register_texture(String::from(name))?;
        self.content.push(ImageContent::RGB8(w, h, bytes));
        Ok(self)
    }

    fn register_texture(&mut self, name: String) -> Result<(), TextureArrayError> {
        if self.textures.contains_key(&name) {
            return Err(TextureArrayError::Other(format!("name '{name}' is already registered")));
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
                let data = assets::read(path)?;
                let format = ImageFormat::from_path(path)?;
                image = Some(image::load_from_memory_with_format(&data, format)?.flipv());
                width = image.as_ref().unwrap().width();
                height = image.as_ref().unwrap().height();
            }
            ImageContent::RGB8(w, h, _) => {
                width = *w;
                height = *h;
            }
        }

        let mip_levels = self.mip_levels.min(width.min(height).ilog2() as u8);

        let textures = self.textures.clone();
        let mut texture = TextureArray::new(
            width,
            height,
            self.content.len() as u32,
            mip_levels,
            self.max_anisotropy,
            textures,
        );

        texture.bind();
        for (i, content) in self.content.iter().enumerate() {
            let iw;
            let ih;
            let data;

            match content {
                ImageContent::File(path) => {
                    if i > 0 {
                        // the first image was already loaded to fetch the dimensions of the array
                        let data = assets::read(path)?;
                        let format = ImageFormat::from_path(path)?;
                        image = Some(image::load_from_memory_with_format(&data, format)?.flipv());
                    }

                    let image = image.as_ref().unwrap();
                    iw = image.width();
                    ih = image.height();
                    data = image.as_rgba8().unwrap().as_raw();
                }
                ImageContent::RGB8(w, h, bytes) => {
                    iw = *w;
                    ih = *h;
                    data = bytes;
                }
            }

            assert!(iw == width && ih == height, "image does not match base dimensions: got: {}x{}, base: {}x{}", iw, ih, width, height);
            assert_eq!(data.len(), (iw * ih * 4) as usize);

            texture.sub_image_3d(i as u32, iw, ih, data);
            gl_assert_no_error!();
        }
        if mip_levels > 1 {
            texture.generate_mipmaps();
        }
        texture.unbind();

        Ok(texture)
    }

    fn flip_image_v(data: &mut [u8], w: u32, h: u32, bytes_per_pixel: u8) {
        assert_eq!(data.len(), (w * h * bytes_per_pixel as u32) as usize);

        for y in 0..(h / 2) {
            if y == h / 2 && (h % 2) != 0 {
                // if h is uneven, the most center column of pixels does not have to be flipped
                continue;
            }

            let y1 = (y * w * bytes_per_pixel as u32) as usize;
            let y2 = ((h - 1 - y) * w * bytes_per_pixel as u32) as usize;

            for x in 0..w {
                let y1 = y1 + x as usize * bytes_per_pixel as usize;
                let y2 = y2 + x as usize * bytes_per_pixel as usize;

                for i in 0..(bytes_per_pixel as usize) {
                    data.swap(y1 + i, y2 + i);
                }
            }
        }
    }
}

pub struct TextureArray {
    gl_id: GLuint,
    textures: FxHashMap<String, u32>,
}

impl Drop for TextureArray {
    fn drop(&mut self) {
        unsafe { gl::DeleteTextures(1, &self.gl_id); }
    }
}

impl TextureArray {
    fn new(width: u32, height: u32, depth: u32, mip_levels: u8, max_anisotropy: f32, textures: FxHashMap<String, u32>) -> Self {
        assert!(mip_levels > 0, "mip_levels must at least be 1, but is {}", mip_levels);

        let mut id = 0;

        unsafe {
            gl::GenTextures(1, &mut id);
            gl::BindTexture(gl::TEXTURE_2D_ARRAY, id);

            gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as GLint);
            gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_WRAP_R, gl::CLAMP_TO_EDGE as GLint);
            gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MIN_FILTER, gl::LINEAR_MIPMAP_LINEAR as GLint);
            gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MAG_FILTER, gl::NEAREST as GLint);
            gl_assert_no_error!();

            if max_anisotropy > 1.0 && crate::core::SUPPORTS_GL_ARB_TEXTURE_FILTER_ANISOTROPIC {
                // GL_MAX_TEXTURE_MAX_ANISOTROPY (extension)
                let mut max_value = 1.0;
                gl::GetFloatv(0x84FF as GLenum, &mut max_value);
                gl_assert_no_error!();

                let mut max_anisotropy = max_anisotropy;
                if max_anisotropy > max_value {
                    max_anisotropy = max_value;
                }

                // GL_TEXTURE_MAX_ANISOTROPY (extension)
                gl::TexParameterf(gl::TEXTURE_2D_ARRAY, 0x84FE as GLenum, max_anisotropy);
                gl_assert_no_error!();
            }

            gl::TexStorage3D(
                gl::TEXTURE_2D_ARRAY,
                mip_levels as GLint,
                gl::RGBA8,
                width as GLint,
                height as GLint,
                depth as GLint,
            );
            gl_assert_no_error!();

            gl::BindTexture(gl::TEXTURE_2D_ARRAY, 0);
        }

        Self { gl_id: id, textures }
    }

    #[allow(clippy::unused_self)]
    pub fn sub_image_3d(&mut self, depth: u32, width: u32, height: u32, data: &[u8]) {
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
                std::ptr::addr_of!(data[0]).cast(),
            );
        }
    }

    #[allow(clippy::unused_self)]
    pub fn generate_mipmaps(&self) {
        unsafe { gl::GenerateMipmap(gl::TEXTURE_2D_ARRAY); }
    }

    pub fn lookup(&self, name: &str) -> Option<u32> {
        if let Some(index) = self.textures.get(&String::from(name)) {
            return Some(*index);
        }
        None
    }
}

impl Bind for TextureArray {
    fn bind(&self) {
        unsafe { gl::BindTexture(gl::TEXTURE_2D_ARRAY, self.gl_id) }
    }

    fn unbind(&self) {
        unsafe { gl::BindTexture(gl::TEXTURE_2D_ARRAY, 0) }
    }
}
