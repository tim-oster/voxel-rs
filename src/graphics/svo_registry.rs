use crate::graphics::buffer;
use crate::graphics::buffer::Buffer;
use crate::graphics::resource::Resource;
use crate::graphics::texture_array::{TextureArray, TextureArrayBuilder, TextureArrayError};
use crate::world::chunk::BlockId;

#[derive(Clone)]
struct Texture {
    name: String,
    path: String,
}

struct MaterialEntry {
    block: BlockId,
    material: Material,
}

pub struct Material {
    specular_pow: f32,
    specular_strength: f32,
    tex_top: Option<String>,
    tex_side: Option<String>,
    tex_bottom: Option<String>,
    tex_top_normal: Option<String>,
    tex_side_normal: Option<String>,
    tex_bottom_normal: Option<String>,
}

#[repr(C)]
#[derive(Clone, Default)]
pub(super) struct MaterialInstance {
    pub specular_pow: f32,
    pub specular_strength: f32,
    pub tex_top: i32,
    pub tex_side: i32,
    pub tex_bottom: i32,
    pub tex_top_normal: i32,
    pub tex_side_normal: i32,
    pub tex_bottom_normal: i32,
}

impl Material {
    pub fn new() -> Material {
        Material {
            specular_pow: 0.0,
            specular_strength: 0.0,
            tex_top: None,
            tex_side: None,
            tex_bottom: None,
            tex_top_normal: None,
            tex_side_normal: None,
            tex_bottom_normal: None,
        }
    }

    /// specular set specular material properties for this material.
    pub fn specular(mut self, pow: f32, strength: f32) -> Material {
        self.specular_pow = pow;
        self.specular_strength = strength;
        self
    }

    /// all_sides applies the same texture to all sides of the material.
    pub fn all_sides(self, name: &'static str) -> Material {
        self.top(name).side(name).bottom(name)
    }

    pub fn top(mut self, name: &'static str) -> Material {
        self.tex_top = Some(String::from(name));
        self
    }

    pub fn side(mut self, name: &'static str) -> Material {
        self.tex_side = Some(String::from(name));
        self
    }

    pub fn bottom(mut self, name: &'static str) -> Material {
        self.tex_bottom = Some(String::from(name));
        self
    }

    /// with_normals adds normal textures to all sides a texture was set for. The path is the
    /// same as the side's texture with a "_normal" suffix.
    pub fn with_normals(mut self) -> Material {
        if let Some(tex) = &self.tex_top {
            self.tex_top_normal = Some(tex.clone() + "_normal");
        }
        if let Some(tex) = &self.tex_side {
            self.tex_side_normal = Some(tex.clone() + "_normal");
        }
        if let Some(tex) = &self.tex_bottom {
            self.tex_bottom_normal = Some(tex.clone() + "_normal");
        }
        self
    }
}

pub struct VoxelRegistry {
    textures: Vec<Texture>,
    materials: Vec<MaterialEntry>,
}

impl VoxelRegistry {
    pub fn new() -> VoxelRegistry {
        VoxelRegistry {
            materials: Vec::new(),
            textures: Vec::new(),
        }
    }

    pub fn add_texture(&mut self, name: &'static str, path: &'static str) -> &mut VoxelRegistry {
        self.textures.push(Texture { name: String::from(name), path: String::from(path) });
        self
    }

    pub fn add_material(&mut self, block: BlockId, material: Material) -> &mut VoxelRegistry {
        self.materials.push(MaterialEntry { block, material });
        self
    }

    pub(super) fn build_texture_array(&self) -> Result<Resource<TextureArray, TextureArrayError>, TextureArrayError> {
        let textures = self.textures.clone();
        Resource::new(
            move || {
                let mut builder = TextureArrayBuilder::new(1); // TODO
                for tex in &textures {
                    builder.add_file(&tex.name, &tex.path)?;
                }
                builder.build()
            }
        )
    }

    pub(super) fn build_material_buffer(&self, tex_array: &TextureArray) -> Buffer<MaterialInstance> {
        fn lookup(array: &TextureArray, name: Option<&String>) -> i32 {
            name.map_or(
                0,
                |name| array.lookup(name).unwrap_or(0) as i32,
            )
        }

        let max_block_id = self.materials.iter()
            .max_by(|lhs, rhs| lhs.block.cmp(&rhs.block))
            .unwrap()
            .block;

        let mut materials = vec![Default::default(); max_block_id as usize + 1];

        for entry in &self.materials {
            let mat = &entry.material;
            materials[entry.block as usize] = MaterialInstance {
                specular_pow: mat.specular_pow,
                specular_strength: mat.specular_strength,
                tex_top: lookup(tex_array, mat.tex_top.as_ref()),
                tex_side: lookup(tex_array, mat.tex_side.as_ref()),
                tex_bottom: lookup(tex_array, mat.tex_bottom.as_ref()),
                tex_top_normal: lookup(tex_array, mat.tex_top_normal.as_ref()),
                tex_side_normal: lookup(tex_array, mat.tex_side_normal.as_ref()),
                tex_bottom_normal: lookup(tex_array, mat.tex_bottom_normal.as_ref()),
            };
        }

        Buffer::new(materials, buffer::STATIC_READ)
    }
}
