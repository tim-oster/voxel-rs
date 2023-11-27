use std::cell::RefCell;

use cgmath::{EuclideanSpace, Matrix4, Point3, Vector3};

use crate::graphics::buffer::{Buffer, MappedBuffer};
use crate::graphics::fence::Fence;
use crate::graphics::resource::Resource;
use crate::graphics::screen_quad::ScreenQuad;
use crate::graphics::shader::{ShaderError, ShaderProgram, ShaderProgramBuilder};
use crate::graphics::svo_picker::{PickerBatch, PickerBatchResult, PickerResult, PickerTask};
use crate::graphics::svo_registry::{MaterialInstance, VoxelRegistry};
use crate::graphics::texture_array::{TextureArray, TextureArrayError};
use crate::world;
use crate::world::chunk::{BlockPos, ChunkPos};
use crate::world::svo::SerializedChunk;

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
/// with a VoxelRegistry with textures and materials to render the actual chunks.
///
/// All coordinates passed are transformed from the actual SVO coordinate space to the rendering
/// space. The converting [`CoordSpace`] can be set with `Svo::update`.
pub struct Svo {
    tex_array: Resource<TextureArray, TextureArrayError>,
    // _material_buffer needs to be stored to drop it together with all other resources
    _material_buffer: Buffer<MaterialInstance>,
    world_shader: Resource<ShaderProgram, ShaderError>,
    world_buffer: MappedBuffer<u32>,
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
    coord_space: Option<CoordSpace>,
}

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// size_bytes is the size of the CPU side SVO buffer that is synced to the GPU.
    pub size_bytes: usize,
    /// depth is the number of octant divisions the SVO has, until the leaf node is encoded.
    pub depth: u32,
}

pub struct RenderParams {
    /// ambient_intensity is the amount of ambient light present in the scene.
    pub ambient_intensity: f32,
    /// light_dir indicates in which direction sun light shines in the scene.
    pub light_dir: Vector3<f32>,
    /// cam_pos is the eye position from which the scene is rendered.
    pub cam_pos: Point3<f32>,
    /// view_mat is the camera view matrix.
    pub view_mat: Matrix4<f32>,
    /// fov_y_rad is the vertical field of view in radians.
    pub fov_y_rad: f32,
    /// aspect_ratio is `width / height` of the screen's resolution.
    pub aspect_ratio: f32,
    /// selected_voxel is the position, in SVO-space, of the voxel to be highlighted. It is
    /// transformed using the `coord_space` passed in [`Svo::update`].
    pub selected_voxel: Option<Point3<f32>>,
}

impl Svo {
    pub fn new(registry: VoxelRegistry) -> Svo {
        let tex_array = registry.build_texture_array().unwrap();
        let material_buffer = registry.build_material_buffer(&tex_array);
        material_buffer.bind_as_storage_buffer(buffer_indices::MATERIALS);

        let world_shader = Resource::new(
            || ShaderProgramBuilder::new().load_shader_bundle("assets/shaders/world.glsl")?.build()
        ).unwrap();

        let world_buffer = MappedBuffer::<u32>::new(1000 * 1024 * 1024); // 1000 MB
        world_buffer.bind_as_storage_buffer(buffer_indices::WORLD);

        let picker_shader = Resource::new(
            || ShaderProgramBuilder::new().load_shader_bundle("assets/shaders/picker.glsl")?.build()
        ).unwrap();

        let picker_in_buffer = MappedBuffer::<PickerTask>::new(100);
        picker_in_buffer.bind_as_storage_buffer(buffer_indices::PICKER_IN);

        let picker_out_buffer = MappedBuffer::<PickerResult>::new(100);
        picker_out_buffer.bind_as_storage_buffer(buffer_indices::PICKER_OUT);

        Svo {
            tex_array,
            _material_buffer: material_buffer,
            world_shader,
            world_buffer,
            screen_quad: ScreenQuad::new(),
            render_fence: RefCell::new(Fence::new()),

            picker_shader,
            picker_in_buffer,
            picker_out_buffer,
            picker_fence: RefCell::new(Fence::new()),

            stats: Stats { size_bytes: 0, depth: 0 },
            coord_space: None,
        }
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

    /// update will write all changes from the given `svo` to the GPU buffer. It will also update
    /// the internal `coord_space` used for SVO <-> GPU space conversions.
    pub fn update(&mut self, svo: &world::svo::Svo<SerializedChunk>, coord_space: Option<CoordSpace>) {
        self.coord_space = coord_space;

        unsafe {
            let max_depth_exp = (-(svo.depth() as f32)).exp2();
            self.world_buffer.write(max_depth_exp.to_bits());

            // wait for last draw call to finish so that updates and draws do not race and produce temporary "holes" in the world
            self.render_fence.borrow().wait();
            svo.write_changes_to(self.world_buffer.offset(1));

            self.stats = Stats {
                size_bytes: svo.size_in_bytes(),
                depth: svo.depth(),
            };
        }
    }

    pub fn get_stats(&self) -> Stats {
        self.stats
    }

    /// render will draw a full-screen quad on which the raytracing shader is executed.
    pub fn render(&self, params: RenderParams) {
        self.world_shader.bind();

        self.world_shader.set_f32("u_ambient", params.ambient_intensity);
        self.world_shader.set_f32vec3("u_light_dir", &params.light_dir);
        self.world_shader.set_f32vec3("u_cam_pos", &params.cam_pos.to_vec());
        self.world_shader.set_f32mat4("u_view", &params.view_mat);
        self.world_shader.set_f32("u_fovy", params.fov_y_rad);
        self.world_shader.set_f32("u_aspect", params.aspect_ratio);
        self.world_shader.set_texture("u_texture", 0, &self.tex_array);

        let mut selected_block = Vector3::new(f32::NAN, f32::NAN, f32::NAN);
        if let Some(mut pos) = params.selected_voxel {
            if let Some(cs) = self.coord_space {
                pos = cs.cnv_into_space(pos);
            }
            selected_block = pos.to_vec();
        }
        self.world_shader.set_f32vec3("u_highlight_pos", &selected_block);

        self.screen_quad.render();
        self.world_shader.unbind();

        // place a fence to allow for waiting on the current frame to be rendered
        self.render_fence.borrow_mut().place();
    }

    /// raycast uploads the given `batch` to the GPU and runs a compute shader on it to calculate
    /// SVO interceptions without rendering anything.
    pub fn raycast(&self, batch: PickerBatch) -> PickerBatchResult {
        self.picker_shader.bind();

        let in_data = self.picker_in_buffer.as_slice_mut();
        let task_count = batch.serialize_tasks(in_data, self.coord_space);

        unsafe {
            gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
            gl::DispatchCompute(task_count as u32, 1, 1);

            // memory barrier + sync fence necessary to ensure that persistently mapped buffer changes
            // are loaded from the server (https://www.khronos.org/opengl/wiki/Buffer_Object#Persistent_mapping)
            gl::MemoryBarrier(gl::CLIENT_MAPPED_BUFFER_BARRIER_BIT);
        }

        self.picker_fence.borrow_mut().place();
        self.picker_fence.borrow().wait();

        self.picker_shader.unbind();

        let out_data = self.picker_out_buffer.as_slice();
        batch.deserialize_results(&out_data[..task_count], self.coord_space)
    }
}

#[cfg(test)]
mod svo_tests {
    use std::sync::Arc;

    use cgmath::{InnerSpace, Matrix4, Point3, SquareMatrix, Vector3};

    use crate::{assert_float_eq, gl_assert_no_error, world};
    use crate::core::GlContext;
    use crate::graphics::framebuffer::{diff_images, Framebuffer};
    use crate::graphics::svo::{RenderParams, Svo};
    use crate::graphics::svo_picker::{PickerBatch, PickerBatchResult, RayResult};
    use crate::graphics::svo_registry::{Material, VoxelRegistry};
    use crate::world::allocator::Allocator;
    use crate::world::chunk::{Chunk, ChunkPos, ChunkStorage};
    use crate::world::octree::Position;
    use crate::world::svo::SerializedChunk;

    fn create_world_svo<F>(builder: F) -> world::svo::Svo<SerializedChunk>
        where F: FnOnce(&mut Chunk) {
        let allocator = Allocator::new(
            Box::new(|| ChunkStorage::with_size(32f32.log2() as u32)),
            Some(Box::new(|storage| storage.reset())),
        );
        let allocator = Arc::new(allocator);

        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0), allocator);
        builder(&mut chunk);

        let chunk = SerializedChunk::new(chunk.pos, chunk.get_storage().unwrap(), 5);
        let mut svo = world::svo::Svo::<SerializedChunk>::new();
        svo.set(Position(0, 0, 0), Some(chunk));
        svo.serialize();
        svo
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
    /// applied. Result is stored in an image an compared against a reference image.
    #[test]
    fn render() {
        let (width, height) = (640, 490);
        let _context = GlContext::new_headless(width, height); // do not drop context
        let world_svo = create_world_svo(|chunk| {
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

        let mut svo = Svo::new(create_voxel_registry());
        svo.update(&world_svo, None);

        let fb = Framebuffer::new(width as i32, height as i32);

        fb.bind();
        fb.clear(0.0, 0.0, 0.0, 1.0);

        let cam_pos = Point3::new(2.5, 2.5, 7.5);
        svo.render(RenderParams {
            ambient_intensity: 0.3,
            light_dir: Vector3::new(-1.0, -1.0, -1.0).normalize(),
            cam_pos,
            view_mat: Matrix4::look_to_rh(cam_pos, -Vector3::unit_z(), Vector3::unit_y()).invert().unwrap(),
            fov_y_rad: 72.0f32.to_radians(),
            aspect_ratio: width as f32 / height as f32,
            selected_voxel: Some(Point3::new(1.0, 1.0, 3.0)),
        });
        fb.unbind();
        gl_assert_no_error!();

        let actual = fb.as_image();
        actual.save_with_format("assets/tests/graphics_svo_render_actual.png", image::ImageFormat::Png).unwrap();

        let expected = image::open("assets/tests/graphics_svo_render_expected.png").unwrap();
        let diff_percent = diff_images(&actual, &expected);
        assert!(diff_percent < 0.001, "difference: {:.5} < 0.001", diff_percent);
    }

    /// Tests if multiple raycasts return the expected results.
    #[test]
    fn raycast() {
        let _context = GlContext::new_headless(1, 1); // do not drop context
        let world_svo = create_world_svo(|chunk| {
            chunk.set_block(0, 0, 0, 1);
            chunk.set_block(1, 0, 0, 1);
        });

        let mut svo = Svo::new(create_voxel_registry());
        svo.update(&world_svo, None);

        let mut batch = PickerBatch::new();
        batch.add_ray(Point3::new(0.5, 1.5, 0.5), Vector3::new(0.0, -1.0, 0.0), 1.0);
        batch.add_ray(Point3::new(0.5, 0.5, 0.5), Vector3::new(1.0, 0.0, 0.0), 1.0);
        batch.add_ray(Point3::new(0.5, 0.5, -2.0), Vector3::new(0.0, 0.0, 1.0), 1.0);

        let result = svo.raycast(batch);
        gl_assert_no_error!();
        assert_eq!(result, PickerBatchResult {
            rays: vec![
                RayResult {
                    dst: 0.5,
                    inside_block: false,
                    pos: Point3::new(0.5, assert_float_eq!(result.rays[0].pos.y, 1.0), 0.5),
                    normal: Vector3::new(0.0, 1.0, 0.0),
                },
                RayResult {
                    dst: 0.5,
                    inside_block: true,
                    pos: Point3::new(assert_float_eq!(result.rays[1].pos.x, 1.0), 0.5, 0.5),
                    normal: Vector3::new(-1.0, 0.0, 0.0),
                },
                RayResult {
                    dst: -1.0,
                    inside_block: false,
                    pos: Point3::new(0.0, 0.0, 0.0),
                    normal: Vector3::new(0.0, 0.0, 0.0),
                },
            ],
            aabbs: vec![],
        });
    }
}

#[derive(Copy, Clone)]
pub struct CoordSpace {
    pub center: ChunkPos,
    pub dst: u32,
}

pub type CoordSpacePos = Point3<f32>;

#[allow(dead_code)]
impl CoordSpace {
    pub fn new(center: ChunkPos, dst: u32) -> CoordSpace {
        CoordSpace { center, dst }
    }

    pub fn cnv_into_space(&self, pos: Point3<f32>) -> CoordSpacePos {
        let mut block_pos = BlockPos::from(pos);
        let delta = block_pos.chunk - self.center;

        let rd = self.dst as i32;
        block_pos.chunk.x = rd + delta.x;
        block_pos.chunk.y = rd + delta.y;
        block_pos.chunk.z = rd + delta.z;

        block_pos.to_point()
    }

    pub fn cnv_out_of_space(&self, pos: CoordSpacePos) -> Point3<f32> {
        let mut block_pos = BlockPos::from(pos);

        let rd = self.dst as i32;
        let delta = block_pos.chunk - ChunkPos::new(rd, rd, rd);

        block_pos.chunk.x = self.center.x + delta.x;
        block_pos.chunk.y = self.center.y + delta.y;
        block_pos.chunk.z = self.center.z + delta.z;

        block_pos.to_point()
    }
}

#[cfg(test)]
mod coord_space_tests {
    use cgmath::Point3;

    use crate::graphics::svo::CoordSpace;
    use crate::world::chunk::ChunkPos;

    /// Test transformation for positive coordinates.
    #[test]
    fn coord_space_positive() {
        let cs = CoordSpace {
            center: ChunkPos::new(4, 5, 12),
            dst: 2,
        };

        let world_pos = Point3::new(32.0 * 5.0 + 16.25, 32.0 * 3.0 + 4.25, 32.0 * 10.0 + 20.5);
        let svo_pos = cs.cnv_into_space(world_pos);
        assert_eq!(svo_pos, Point3::new(32.0 * 3.0 + 16.25, 32.0 * 0.0 + 4.25, 32.0 * 0.0 + 20.5));

        let cnv_back = cs.cnv_out_of_space(svo_pos);
        assert_eq!(cnv_back, world_pos);
    }

    /// Test transformation for negative coordinates.
    #[test]
    fn coord_space_negative() {
        let cs = CoordSpace {
            center: ChunkPos::new(-1, -1, -1),
            dst: 2,
        };

        let world_pos = Point3::new(-16.25, -4.25, -20.5);
        let svo_pos = cs.cnv_into_space(world_pos);
        assert_eq!(svo_pos, Point3::new(32.0 * 2.0 + 15.75, 32.0 * 2.0 + 27.75, 32.0 * 2.0 + 11.5));

        let cnv_back = cs.cnv_out_of_space(svo_pos);
        assert_eq!(cnv_back, world_pos);
    }
}
