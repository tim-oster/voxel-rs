use std::cell::RefCell;
use std::ops::Deref;
use std::ptr;

use cgmath::{EuclideanSpace, Matrix4, Point3, SquareMatrix, Vector3};

use crate::graphics::buffer::{Buffer, MappedBuffer};
use crate::graphics::fence::Fence;
use crate::graphics::framebuffer::Framebuffer;
use crate::graphics::resource::Resource;
use crate::graphics::screen_quad::ScreenQuad;
use crate::graphics::shader::{ShaderError, ShaderProgram, ShaderProgramBuilder};
use crate::graphics::svo_picker::{PickerBatch, PickerBatchResult, PickerResult, PickerTask};
use crate::graphics::svo_registry::{MaterialInstance, VoxelRegistry};
use crate::graphics::texture_array::{TextureArray, TextureArrayError};
use crate::world::hds::WorldSvo;

#[derive(Debug, Copy, Clone)]
pub enum SvoType {
    Esvo,
    Csvo,
}

#[derive(Debug, Copy, Clone)]
pub struct SvoTypeProperties {
    pub name: &'static str,
    pub shader_type_define: &'static str,
}

impl Deref for SvoType {
    type Target = SvoTypeProperties;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Esvo => &SvoTypeProperties { name: "ESVO", shader_type_define: "1" },
            Self::Csvo => &SvoTypeProperties { name: "CSVO", shader_type_define: "2" },
        }
    }
}

/// Buffer indices are constants for all buffer ids used in the SVO shaders.
#[allow(dead_code)]
pub mod buffer_indices {
    pub const WORLD: u32 = 0;
    pub const MATERIALS: u32 = 2;
    pub const PICKER_OUT: u32 = 1;
    pub const PICKER_IN: u32 = 3;
    pub const DEBUG_IN: u32 = 11;
    pub const DEBUG_OUT: u32 = 12;
}

/// Svo can be used to render an SVO of [`SerializedChunk`]. It is initialised
/// with a `VoxelRegistry` with textures and materials to render the actual chunks.
///
/// Note that all coordinates passed must be in SVO coordinate space (\[0;size\] along all axes).
pub struct Svo {
    tex_array: Resource<TextureArray, TextureArrayError>,
    material_buffer: Buffer<MaterialInstance>,
    world_shader: Resource<ShaderProgram, ShaderError>,
    world_buffer: MappedBuffer<u8>,
    // screen_quad is used to render a full-screen quad on which the per-pixel raytracer for the SVO
    // is executed
    screen_quad: ScreenQuad,
    // render_fence synchronizes changes to the mapped world buffer with the renderer
    render_fence: RefCell<Fence>,

    picker_shader: Resource<ShaderProgram, ShaderError>,
    picker_in_buffer: MappedBuffer<PickerTask>,
    picker_out_buffer: MappedBuffer<PickerResult>,
    picker_fence: RefCell<Fence>,

    stats: Stats,
}

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// `used_bytes` is the amount of bytes that is used by of the mapped buffer.
    pub used_bytes: usize,
    /// `capacity_bytes` is the full size of the mapped buffer on both CPU & GPU.
    pub capacity_bytes: usize,
    /// depth is the number of octant divisions the SVO has, until the leaf node is encoded.
    pub depth: u8,
}

pub struct RenderParams {
    /// `ambient_intensity` is the amount of ambient light present in the scene.
    pub ambient_intensity: f32,
    /// `light_dir` indicates in which direction sun light shines in the scene.
    pub light_dir: Vector3<f32>,
    /// `cam_pos` is the eye position from which the scene is rendered.
    pub cam_pos: Point3<f32>,
    /// `cam_fwd` is the look at direction of the camera.
    pub cam_fwd: Vector3<f32>,
    /// `cam_up` is the up vector of the camera.
    pub cam_up: Vector3<f32>,
    /// `fov_y_rad` is the vertical field of view in radians.
    pub fov_y_rad: f32,
    /// `aspect_ratio` is `width / height` of the screen's resolution.
    pub aspect_ratio: f32,
    /// `selected_voxel` is the position of the voxel to be highlighted.
    pub selected_voxel: Option<Point3<f32>>,
    /// `render_shadows` enables secondary ray casting to check for sunlight occlusion.
    pub render_shadows: bool,
    /// `shadow_distance` defines the maximum distance to the primary hit, until which secondary rays are cast.
    pub shadow_distance: f32,
}

impl Svo {
    pub fn new(registry: &VoxelRegistry, typ: SvoType) -> Self {
        let svo_type = typ.shader_type_define;

        let tex_array = registry.build_texture_array().unwrap();
        let material_buffer = registry.build_material_buffer(&tex_array);

        let world_shader = Resource::new(
            || ShaderProgramBuilder::new()
                .with_define("SVO_TYPE", svo_type)
                .load_shader_bundle("assets/shaders/world.glsl")?
                .build()
        ).unwrap();

        let picker_shader = Resource::new(
            || ShaderProgramBuilder::new()
                .with_define("SVO_TYPE", svo_type)
                .load_shader_bundle("assets/shaders/picker.glsl")?
                .build()
        ).unwrap();

        let instance = Self {
            tex_array,
            material_buffer,
            world_shader,
            world_buffer: MappedBuffer::new(300 * 1000 * 1000), // 300 MB
            screen_quad: ScreenQuad::new(),
            render_fence: RefCell::new(Fence::new()),

            picker_shader,
            picker_in_buffer: MappedBuffer::<PickerTask>::new(100),
            picker_out_buffer: MappedBuffer::<PickerResult>::new(100),
            picker_fence: RefCell::new(Fence::new()),

            stats: Stats { used_bytes: 0, capacity_bytes: 0, depth: 0 },
        };

        // default bind after instantiation
        instance.bind_buffers_globally();

        instance
    }

    pub fn bind_buffers_globally(&self) {
        self.material_buffer.bind_as_storage_buffer(buffer_indices::MATERIALS);
        self.world_buffer.bind_as_storage_buffer(buffer_indices::WORLD);
        self.picker_in_buffer.bind_as_storage_buffer(buffer_indices::PICKER_IN);
        self.picker_out_buffer.bind_as_storage_buffer(buffer_indices::PICKER_OUT);
    }

    pub fn reload_resources(&mut self) {
        if let Err(e) = self.tex_array.reload() {
            println!("error reloading texture array: {e:?}");
        }
        if let Err(e) = self.world_shader.reload() {
            println!("error reloading world shader: {e:?}");
        }
        if let Err(e) = self.picker_shader.reload() {
            println!("error reloading picker shader: {e:?}");
        }
    }

    /// Writes all changes from the given `svo` to the GPU buffer.
    pub fn update<T: WorldSvo<U> + ?Sized, U>(&mut self, svo: &mut T) {
        unsafe {
            let max_depth_exp = (-(svo.depth() as f32)).exp2();
            let max_depth_exp_bytes = max_depth_exp.to_bits().to_le_bytes();
            ptr::copy(max_depth_exp_bytes.as_ptr(), self.world_buffer.cast(), max_depth_exp_bytes.len());

            // wait for last draw call to finish so that updates and draws do not race and produce temporary "holes" in the world
            self.render_fence.borrow().wait();

            let len = self.world_buffer.len() - 1;
            svo.write_changes_to(self.world_buffer.offset(4), len, true);

            self.stats = Stats {
                used_bytes: svo.size_in_bytes(),
                capacity_bytes: self.world_buffer.size_in_bytes(),
                depth: svo.depth(),
            };
        }
    }

    pub fn get_stats(&self) -> Stats {
        self.stats
    }

    /// Draws a full-screen quad on which the raytracing shader is executed.
    pub fn render(&self, params: &RenderParams, target: &Framebuffer) {
        let view_mat = Matrix4::look_to_rh(params.cam_pos, params.cam_fwd, params.cam_up).invert().unwrap();

        self.world_shader.bind();

        self.world_shader.set_f32("u_ambient", params.ambient_intensity);
        self.world_shader.set_f32vec3("u_light_dir", &params.light_dir);
        self.world_shader.set_f32vec3("u_cam_pos", &params.cam_pos.to_vec());
        self.world_shader.set_f32mat4("u_view", &view_mat);
        self.world_shader.set_f32("u_fovy", params.fov_y_rad);
        self.world_shader.set_f32("u_aspect", params.aspect_ratio);
        self.world_shader.set_texture("u_texture", 0, &self.tex_array);
        self.world_shader.set_i32("u_render_shadows", params.render_shadows as i32);
        self.world_shader.set_f32("u_shadow_distance", params.shadow_distance);

        let mut selected_block = Vector3::new(f32::NAN, f32::NAN, f32::NAN);
        if let Some(pos) = params.selected_voxel {
            selected_block = pos.to_vec();
        }
        self.world_shader.set_f32vec3("u_highlight_pos", &selected_block);

        unsafe {
            let (width, height) = (target.width(), target.height());

            gl::BindImageTexture(0, target.color_attachment(), 0, gl::FALSE, 0, gl::WRITE_ONLY, gl::RGBA32F);
            gl::DispatchCompute((width / 32 + 1) as u32, (height / 32 + 1) as u32, 1);
            gl::MemoryBarrier(gl::SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }

        self.world_shader.unbind();

        // place a fence to allow for waiting on the current frame to be rendered
        self.render_fence.borrow_mut().place();
    }

    /// Uploads the given `batch` to the GPU and runs a compute shader on it to calculate
    /// SVO interceptions without rendering anything.
    pub fn raycast(&self, batch: &PickerBatch, result: &mut PickerBatchResult) {
        self.picker_shader.bind();

        let in_data = self.picker_in_buffer.as_slice_mut();
        let task_count = batch.serialize_tasks(in_data);

        unsafe {
            gl::DispatchCompute(task_count as u32, 1, 1);

            // memory barrier is not required because buffer is mapped with gl::MAP_COHERENT_BIT
            // gl::MemoryBarrier(gl::CLIENT_MAPPED_BUFFER_BARRIER_BIT);
        }

        // sync fence necessary to ensure that persistently mapped buffer changes are loaded from the server
        // (https://www.khronos.org/opengl/wiki/Buffer_Object#Persistent_mapping)
        self.picker_fence.borrow_mut().place();
        self.picker_fence.borrow().wait();

        self.picker_shader.unbind();

        let out_data = self.picker_out_buffer.as_slice();
        batch.deserialize_results(&out_data[..task_count], result);
    }
}

#[cfg(test)]
mod svo_tests {
    use std::env;
    use std::sync::Arc;

    use cgmath::{InnerSpace, Point3, Vector3};

    use crate::{assert_float_eq, gl_assert_no_error};
    use crate::core::GlContext;
    use crate::graphics::framebuffer::{diff_images, Framebuffer};
    use crate::graphics::macros::assert_vec3_eq;
    use crate::graphics::svo::{RenderParams, Svo, SvoType};
    use crate::graphics::svo_picker::{PickerBatch, PickerBatchResult, RayResult};
    use crate::graphics::svo_registry::{Material, VoxelRegistry};
    use crate::world::chunk::{Chunk, ChunkPos, ChunkStorageAllocator};
    use crate::world::hds::{ChunkBufferPool, csvo, esvo, WorldSvo};
    use crate::world::hds::octree::Position;
    use crate::world::world::BorrowedChunk;

    struct TestCase {
        name: &'static str,
        svo: Svo,
    }

    fn create_test_cases<F>(chunk_builder: F) -> Vec<TestCase>
    where
        F: Fn(&mut Chunk),
    {
        let create_chunk = || {
            let storage_alloc = ChunkStorageAllocator::new();
            let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0), 5, storage_alloc.allocate());
            chunk_builder(&mut chunk);
            chunk
        };
        let create_esvo = || {
            let buffer_alloc = Arc::new(ChunkBufferPool::default());
            let chunk = esvo::SerializedChunk::new(BorrowedChunk::from(create_chunk()), &buffer_alloc);

            let mut svo = esvo::Esvo::<esvo::SerializedChunk>::new();
            svo.set_leaf(Position(0, 0, 0), chunk, true);
            svo.serialize();

            let mut esvo = Svo::new(&create_voxel_registry(), SvoType::Esvo);
            esvo.update(&mut svo);
            esvo
        };
        let create_csvo = || {
            let buffer_alloc = Arc::new(ChunkBufferPool::default());
            let chunk = csvo::SerializedChunk::new(BorrowedChunk::from(create_chunk()), &buffer_alloc);

            let mut svo = csvo::Csvo::new();
            svo.set_leaf(Position(0, 0, 0), chunk, true);
            svo.serialize();

            let mut csvo = Svo::new(&create_voxel_registry(), SvoType::Csvo);
            csvo.update(&mut svo);
            csvo
        };

        vec![
            TestCase { name: "esvo", svo: create_esvo() },
            TestCase { name: "csvo", svo: create_csvo() },
        ]
    }

    fn create_voxel_registry() -> VoxelRegistry {
        let mut registry = VoxelRegistry::new();
        registry
            .add_texture("stone", "assets/textures/stone.png")
            .add_texture("stone_normal", "assets/textures/stone_n.png")
            .add_texture("dirt", "assets/textures/dirt.png")
            .add_texture("dirt_normal", "assets/textures/dirt_n.png")
            .add_texture("grass_side", "assets/textures/grass_side.png")
            .add_texture("grass_side_normal", "assets/textures/grass_side_n.png")
            .add_texture("grass_top", "assets/textures/grass_top.png")
            .add_texture("grass_top_normal", "assets/textures/grass_top_n.png")
            .add_material(0, Material::new())
            .add_material(1, Material::new().specular(70.0, 0.4).all_sides("stone").with_normals())
            .add_material(2, Material::new().specular(14.0, 0.4).top("grass_top").side("grass_side").bottom("dirt").with_normals());
        registry
    }

    /// Tests if rendering of a demo chunks works correctly. Voxels are textured and lighting is
    /// applied. Result is stored in an image and compared against a reference image.
    #[test]
    fn render() {
        let (width, height) = (640, 490);
        let _context = GlContext::new_headless(width, height); // do not drop context

        let cases = create_test_cases(|chunk| {
            for x in 0..5 {
                for z in 0..5 {
                    chunk.set_block(x, 0, z, 1);
                }
            }

            chunk.set_block(1, 1, 1, 2);
            chunk.set_block(3, 1, 1, 2);
            chunk.set_block(1, 3, 1, 2);
            chunk.set_block(3, 3, 1, 2);

            chunk.set_block(1, 1, 3, 2);
            chunk.set_block(3, 1, 3, 2);
            chunk.set_block(1, 3, 3, 2);
            chunk.set_block(3, 3, 3, 2);
        });
        for tc in cases {
            tc.svo.bind_buffers_globally();

            let fb = Framebuffer::new(width as i32, height as i32, false, false);
            fb.bind();
            fb.clear(0.0, 0.0, 0.0, 1.0);

            let cam_pos = Point3::new(2.5, 2.5, 7.5);
            tc.svo.render(&RenderParams {
                ambient_intensity: 0.3,
                light_dir: Vector3::new(-1.0, -1.0, -1.0).normalize(),
                cam_pos,
                cam_fwd: -Vector3::unit_z(),
                cam_up: Vector3::unit_y(),
                fov_y_rad: 72.0f32.to_radians(),
                aspect_ratio: width as f32 / height as f32,
                selected_voxel: Some(Point3::new(1.0, 1.0, 3.0)),
                render_shadows: true,
                shadow_distance: 500.0,
            }, &fb);

            fb.unbind();
            gl_assert_no_error!();

            let actual = fb.as_image();
            actual.save_with_format("assets/tests/graphics_svo_render_actual.png", image::ImageFormat::Png).unwrap();

            let expected = image::open("assets/tests/graphics_svo_render_expected.png").unwrap();
            let diff_percent = diff_images(&actual, &expected);
            let threshold = env::var("TEST_SVO_RENDER_THRESHOLD").map_or(0.001, |x| x.parse::<f64>().unwrap());

            println!("test case {}", tc.name);
            assert!(diff_percent < threshold, "difference for test case {}: {:.5} < {:.5}", tc.name, diff_percent, threshold);
            println!("passed");
        }
    }

    /// Tests if multiple raycasts return the expected results.
    #[test]
    fn raycast() {
        let _context = GlContext::new_headless(1, 1); // do not drop context

        let cases = create_test_cases(|chunk| {
            chunk.set_block(0, 0, 0, 1);
            chunk.set_block(1, 0, 0, 1);
        });
        for tc in cases {
            tc.svo.bind_buffers_globally();

            let mut batch = PickerBatch::new();
            batch.add_ray(Point3::new(0.5, 1.5, 0.5), Vector3::new(0.0, -1.0, 0.0), 1.0);
            batch.add_ray(Point3::new(0.5, 0.5, 0.5), Vector3::new(1.0, 0.0, 0.0), 1.0);
            batch.add_ray(Point3::new(0.5, 0.5, -2.0), Vector3::new(0.0, 0.0, 1.0), 1.0);

            let mut result = PickerBatchResult::new();
            tc.svo.raycast(&mut batch, &mut result);

            gl_assert_no_error!();

            println!("test case {}", tc.name);
            assert_eq!(result, PickerBatchResult {
                rays: vec![
                    RayResult {
                        dst: assert_float_eq!(result.rays[0].dst, 0.5, 0.0001),
                        inside_voxel: false,
                        pos: assert_vec3_eq!(result.rays[0].pos, Point3::new(0.5, 1.0, 0.5), 0.0001),
                        normal: Vector3::new(0.0, 1.0, 0.0),
                    },
                    RayResult {
                        dst: assert_float_eq!(result.rays[1].dst, 0.5, 0.0001),
                        inside_voxel: true,
                        pos: assert_vec3_eq!(result.rays[1].pos, Point3::new(1.0, 0.5, 0.5), 0.0001),
                        normal: Vector3::new(-1.0, 0.0, 0.0),
                    },
                    RayResult {
                        dst: -1.0,
                        inside_voxel: false,
                        pos: Point3::new(0.0, 0.0, 0.0),
                        normal: Vector3::new(0.0, 0.0, 0.0),
                    },
                ],
                aabbs: vec![],
            });
            println!("passed");
        }
    }
}
