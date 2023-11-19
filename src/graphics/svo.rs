use std::cell::RefCell;

use cgmath::{EuclideanSpace, Matrix4, Point3, Vector3};

use crate::{graphics, world};
use crate::graphics::{ShaderProgram, TextureArray, TextureArrayError};
use crate::graphics::buffer::{Buffer, MappedBuffer};
use crate::graphics::consts::shader_buffer_indices;
use crate::graphics::fence::Fence;
use crate::graphics::resource::Resource;
use crate::graphics::screen_quad::ScreenQuad;
use crate::graphics::shader::ShaderError;
use crate::graphics::svo_picker::{PickerBatch, PickerBatchResult, PickerResult, PickerTask};
use crate::graphics::svo_registry::{MaterialInstance, VoxelRegistry};
use crate::world::chunk::{BlockPos, ChunkPos};
use crate::world::svo::SerializedChunk;

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
    /// selected_block is the position, in SVO-space, of the block to be highlighted. It is
    /// transformed using the `coord_space` passed in [`Svo::update`].
    pub selected_block: Option<Point3<f32>>,
}

impl Svo {
    pub fn new(registry: VoxelRegistry) -> Svo {
        let tex_array = registry.build_texture_array().unwrap();
        let material_buffer = registry.build_material_buffer(&tex_array);
        material_buffer.bind_as_storage_buffer(shader_buffer_indices::MATERIALS);

        let world_shader = Resource::new(
            || graphics::ShaderProgramBuilder::new().load_shader_bundle("assets/shaders/world.glsl")?.build()
        ).unwrap();

        let world_buffer = MappedBuffer::<u32>::new(1000 * 1024 * 1024); // 1000 MB
        world_buffer.bind_as_storage_buffer(shader_buffer_indices::WORLD);

        let picker_shader = Resource::new(
            || graphics::ShaderProgramBuilder::new().load_shader_bundle("assets/shaders/picker.glsl")?.build()
        ).unwrap();

        let picker_in_buffer = MappedBuffer::<PickerTask>::new(100);
        picker_in_buffer.bind_as_storage_buffer(shader_buffer_indices::PICKER_IN);

        let picker_out_buffer = MappedBuffer::<PickerResult>::new(100);
        picker_out_buffer.bind_as_storage_buffer(shader_buffer_indices::PICKER_OUT);

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
    pub fn update(&mut self, svo: &world::svo::Svo<SerializedChunk>, coord_space: CoordSpace) {
        self.coord_space = Some(coord_space);

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
        if let Some(mut pos) = params.selected_block {
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

// TODO write render tests

#[derive(Copy, Clone)]
pub struct CoordSpace {
    pub center: ChunkPos,
    pub dst: u32,
}

pub type CoordSpacePos = Point3<f32>;

impl CoordSpace {
    pub fn new(center: ChunkPos, dst: u32) -> CoordSpace {
        CoordSpace { center, dst }
    }

    pub fn cnv_into_space(&self, pos: Point3<f32>) -> CoordSpacePos {
        let mut block_pos = BlockPos::from(pos);
        let delta = block_pos.chunk - self.center;

        let rd = self.dst as i32;
        block_pos.chunk.x = rd + delta.x;
        // block_pos.chunk.y = rd + delta.y;
        block_pos.chunk.z = rd + delta.z;

        block_pos.to_point()
    }

    pub fn cnv_out_of_space(&self, pos: CoordSpacePos) -> Point3<f32> {
        let mut block_pos = BlockPos::from(pos);

        let rd = self.dst as i32;
        // TODO y=rd is ignored here
        let delta = block_pos.chunk - ChunkPos::new(rd, 0/*rd*/, rd);

        block_pos.chunk.x = self.center.x + delta.x;
        //block_pos.chunk.y = self.center.y + delta.y;
        block_pos.chunk.z = self.center.z + delta.z;

        block_pos.to_point()
    }
}

#[cfg(test)]
mod coord_space_tests {
    use cgmath::Point3;

    use crate::graphics::svo::CoordSpace;
    use crate::world::chunk::ChunkPos;

    #[test]
    fn coord_space_positive() {
        let cs = CoordSpace {
            center: ChunkPos::new(4, 5, 12),
            dst: 2,
        };

        // TODO
        // let world_pos = Point3::new(32.0 * 5.0 + 16.25, 32.0 * 2.0 + 4.25, 32.0 * 10.0 + 20.5);
        // let svo_pos = cs.cnv_into_space(world_pos);
        // assert_eq!(svo_pos, Point3::new(32.0 * 3.0 + 16.25, 32.0 * -1.0 + 4.25, 32.0 * 0.0 + 20.5));

        let world_pos = Point3::new(32.0 * 5.0 + 16.25, 32.0 * 2.0 + 4.2, 32.0 * 10.0 + 20.5);
        let svo_pos = cs.cnv_into_space(world_pos);
        assert_eq!(svo_pos, Point3::new(32.0 * 3.0 + 16.25, 32.0 * 2.0 + 4.2, 32.0 * 0.0 + 20.5));

        let cnv_back = cs.cnv_out_of_space(svo_pos);
        assert_eq!(cnv_back, world_pos);
    }

    #[test]
    fn coord_space_negative() {
        let cs = CoordSpace {
            center: ChunkPos::new(-1, -1, -1),
            dst: 2,
        };

        // TODO
        // let world_pos = Point3::new(-16.25, 4.25, -20.5);
        // let svo_pos = cs.cnv_into_space(world_pos);
        // assert_eq!(svo_pos, Point3::new(32.0 * 2.0 + 15.75, 32.0 * 3.0 + 4.25, 32.0 * 2.0 + 11.5));

        let world_pos = Point3::new(-16.25, 4.2, -20.5);
        let svo_pos = cs.cnv_into_space(world_pos);
        assert_eq!(svo_pos, Point3::new(32.0 * 2.0 + 15.75, 4.2, 32.0 * 2.0 + 11.5));

        let cnv_back = cs.cnv_out_of_space(svo_pos);
        assert_eq!(cnv_back, world_pos);
    }
}
