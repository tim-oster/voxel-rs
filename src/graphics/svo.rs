use std::cell::RefCell;
use std::ops::Deref;

use cgmath::{EuclideanSpace, Matrix4, Point3, Vector3};

use crate::{graphics, world};
use crate::graphics::{buffer, resource, ShaderProgram, TextureArray, TextureArrayError};
use crate::graphics::buffer::{Buffer, MappedBuffer};
use crate::graphics::consts::shader_buffer_indices;
use crate::graphics::fence::Fence;
use crate::graphics::resource::Resource;
use crate::graphics::screen_quad::ScreenQuad;
use crate::graphics::shader::ShaderError;
use crate::graphics::svo_picker::{PickerBatch, PickerResult, PickerTask};
use crate::world::chunk::BlockId;
use crate::world::svo::SerializedChunk;

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

    pub fn specular(mut self, pow: f32, strength: f32) -> Material {
        self.specular_pow = pow;
        self.specular_strength = strength;
        self
    }

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

pub struct ContentRegistry {
    textures: Vec<Texture>,
    materials: Vec<MaterialEntry>,
}

impl ContentRegistry {
    pub fn new() -> ContentRegistry {
        ContentRegistry {
            materials: Vec::new(),
            textures: Vec::new(),
        }
    }

    pub fn add_texture(&mut self, name: &'static str, path: &'static str) -> &mut ContentRegistry {
        self.textures.push(Texture { name: String::from(name), path: String::from(path) });
        self
    }

    pub fn add_material(&mut self, block: BlockId, material: Material) -> &mut ContentRegistry {
        self.materials.push(MaterialEntry { block, material });
        self
    }
}

#[repr(C)]
#[derive(Clone)]
pub(crate) struct MaterialInstance {
    pub specular_pow: f32,
    pub specular_strength: f32,
    pub tex_top: i32,
    pub tex_side: i32,
    pub tex_bottom: i32,
    pub tex_top_normal: i32,
    pub tex_side_normal: i32,
    pub tex_bottom_normal: i32,
}

pub struct Svo {
    screen_quad: ScreenQuad,

    tex_array: Resource<TextureArray, TextureArrayError>,
    world_shader: Resource<ShaderProgram, ShaderError>,
    _material_buffer: Buffer<MaterialInstance>,
    world_buffer: MappedBuffer<u32>,
    render_fence: RefCell<Fence>,

    picker_shader: Resource<ShaderProgram, ShaderError>,
    picker_in_buffer: MappedBuffer<PickerTask>,
    picker_out_buffer: MappedBuffer<PickerResult>,
    picker_fence: RefCell<Fence>,

    stats: Stats,
}

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub size_bytes: f32,
    pub depth: u32,
}

impl Svo {
    pub fn new(registry: ContentRegistry) -> Svo {
        let tex_array = Resource::new(
            Svo::build_texture_array(&registry),
        ).unwrap();

        let world_shader = Resource::new(
            || graphics::ShaderProgramBuilder::new().load_shader_bundle("assets/shaders/world.glsl")?.build()
        ).unwrap();

        let picker_shader = Resource::new(
            || graphics::ShaderProgramBuilder::new().load_shader_bundle("assets/shaders/picker.glsl")?.build()
        ).unwrap();

        let material_buffer = Svo::build_material_buffer(&registry, tex_array.deref());
        material_buffer.bind_as_storage_buffer(shader_buffer_indices::MATERIALS);

        let world_buffer = MappedBuffer::<u32>::new(1000 * 1024 * 1024); // 1000 MB
        world_buffer.bind_as_storage_buffer(shader_buffer_indices::WORLD);

        let picker_in_buffer = MappedBuffer::<PickerTask>::new(50);
        picker_in_buffer.bind_as_storage_buffer(shader_buffer_indices::PICKER_IN);

        let picker_out_buffer = MappedBuffer::<PickerResult>::new(50);
        picker_out_buffer.bind_as_storage_buffer(shader_buffer_indices::PICKER_OUT);

        Svo {
            tex_array,
            world_shader,
            screen_quad: ScreenQuad::new(),
            _material_buffer: material_buffer,
            render_fence: RefCell::new(Fence::new()),
            world_buffer,
            picker_shader,
            picker_in_buffer,
            picker_out_buffer,
            picker_fence: RefCell::new(Fence::new()),
            stats: Stats { size_bytes: 0.0, depth: 0 },
        }
    }

    fn build_texture_array(registry: &ContentRegistry) -> impl resource::Constructor<TextureArray, TextureArrayError> {
        let textures = registry.textures.clone();
        move || {
            let mut builder = graphics::TextureArrayBuilder::new(4);
            for tex in &textures {
                builder.add_file(&tex.name, &tex.path)?;
            }
            builder.build()
        }
    }

    fn build_material_buffer(registry: &ContentRegistry, tex_array: &TextureArray) -> Buffer<MaterialInstance> {
        fn lookup(array: &TextureArray, name: Option<&String>) -> i32 {
            if let Some(name) = name {
                return array.lookup(name).unwrap_or(0) as i32;
            }
            0
        }

        let max_block_id = registry.materials.iter()
            .max_by(|lhs, rhs| lhs.block.cmp(&rhs.block))
            .unwrap()
            .block;
        let mut materials = vec![MaterialInstance {
            specular_pow: 0.0,
            specular_strength: 0.0,
            tex_top: -1,
            tex_side: -1,
            tex_bottom: -1,
            tex_top_normal: -1,
            tex_side_normal: -1,
            tex_bottom_normal: -1,
        }; max_block_id as usize + 1];

        for entry in &registry.materials {
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

    pub fn reload_resources(&mut self) {
        if let Err(e) = self.tex_array.reload() {
            println!("error reloading texture array: {:?}", e);
        }
        if let Err(e) = self.world_shader.reload() {
            println!("error reloading world shader: {:?}", e);
        }
        if let Err(e) = self.picker_shader.reload() {
            println!("error reloading picker shader: {:?}", e);
        }
    }

    pub fn update(&mut self, svo: &mut world::svo::Svo<SerializedChunk>) {
        unsafe {
            let max_depth_exp = (-(svo.depth() as f32)).exp2();
            self.world_buffer.write(max_depth_exp.to_bits());

            // wait for last draw call to finish so that updates and draws do not race and produce temporary "holes" in the world
            // TODO does this issue still occur if new memory blobs are written first and after that related pointers are updated?
            self.render_fence.borrow().wait();
            svo.write_changes_to(self.world_buffer.offset(1));

            self.stats = Stats {
                size_bytes: svo.size_in_bytes() as f32 / 1024f32 / 1024f32,
                depth: svo.depth(),
            };
        }
    }

    pub fn get_stats(&self) -> Stats {
        self.stats
    }
}

pub struct RenderParams {
    pub ambient_intensity: f32,
    pub light_dir: Vector3<f32>,
    pub cam_pos: Point3<f32>,
    pub view_mat: Matrix4<f32>,
    pub fov_y_rad: f32,
    pub aspect_ratio: f32,
    pub selected_block_svo_space: Option<Point3<f32>>,
}

impl Svo {
    pub fn render(&self, params: RenderParams) {
        self.world_shader.bind();

        self.world_shader.set_f32("u_ambient", params.ambient_intensity);
        self.world_shader.set_f32vec3("u_light_dir", &params.light_dir);
        self.world_shader.set_f32vec3("u_cam_pos", &params.cam_pos.to_vec());
        self.world_shader.set_f32mat4("u_view", &params.view_mat);
        self.world_shader.set_f32("u_fovy", params.fov_y_rad);
        self.world_shader.set_f32("u_aspect", params.aspect_ratio);
        self.world_shader.set_texture("u_texture", 0, &self.tex_array);

        let mut selected_block = Vector3::new(999.0, 999.0, 999.0);
        if let Some(pos) = params.selected_block_svo_space {
            selected_block = pos.to_vec();
        }
        self.world_shader.set_f32vec3("u_highlight_pos", &selected_block);

        self.screen_quad.render();
        self.world_shader.unbind();
        self.render_fence.borrow_mut().place();
    }
}

impl Svo {
    pub fn raycast(&self, batch: PickerBatch) {
        self.picker_shader.bind();

        let in_data = self.picker_in_buffer.as_slice_mut();
        batch.serialize_tasks(in_data);

        unsafe {
            gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
            gl::DispatchCompute(50, 1, 1);

            // memory barrier + sync fence necessary to ensure that persistently mapped buffer changes
            // are loaded from the server
            gl::MemoryBarrier(gl::CLIENT_MAPPED_BUFFER_BARRIER_BIT);
        }

        self.picker_fence.borrow_mut().place();
        self.picker_fence.borrow().wait();

        self.picker_shader.unbind();

        let out_data = self.picker_out_buffer.as_slice();
        batch.deserialize_results(out_data);
    }
}

// TODO write tests