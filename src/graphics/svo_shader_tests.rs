#[cfg(test)]
mod tests {
    use std::ptr;
    use std::sync::Arc;

    use cgmath::{InnerSpace, Point2, Point3, Vector3, Vector4};

    use crate::assert_float_eq;
    use crate::core::GlContext;
    use crate::graphics::buffer;
    use crate::graphics::buffer::{Buffer, MappedBuffer};
    use crate::graphics::macros::{AlignedBool, AlignedPoint2, AlignedPoint3, AlignedVec3, AlignedVec4};
    use crate::graphics::macros::{assert_vec2_eq, assert_vec3_eq, assert_vec4_eq};
    use crate::graphics::resource::Resource;
    use crate::graphics::shader::{ShaderError, ShaderProgram, ShaderProgramBuilder};
    use crate::graphics::svo::{buffer_indices, SvoType};
    use crate::graphics::svo_registry::MaterialInstance;
    use crate::graphics::texture_array::{TextureArray, TextureArrayBuilder, TextureArrayError};
    use crate::world::chunk::{Chunk, ChunkPos, ChunkStorageAllocator};
    use crate::world::hds::{ChunkBufferPool, csvo, esvo, WorldSvo};
    use crate::world::hds::octree::Position;
    use crate::world::world::BorrowedChunk;

    #[repr(C)]
    struct BufferIn {
        max_dst: f32,
        cast_translucent: AlignedBool,
        pos: AlignedPoint3<f32>,
        dir: AlignedVec3<f32>,
    }

    #[repr(C)]
    struct BufferOut {
        result: OctreeResult,
        stack_ptr: i32,
        stack: [StackFrame; 100],
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq)]
    struct OctreeResult {
        t: f32,
        value: u32,
        face_id: i32,
        pos: AlignedPoint3<f32>,
        uv: AlignedPoint2<f32>,
        color: AlignedVec4<f32>,
        inside_voxel: AlignedBool,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq)]
    struct StackFrame {
        t_min: f32,
        ptr: u32,
        idx: u32,
        parent_octant_idx: u32,
        scale: i32,
        is_child: AlignedBool,
        is_leaf: AlignedBool,
        crossed_boundary: AlignedBool,
        next_ptr: u32,
    }

    trait SvoWrapper<> {
        fn depth(&self) -> u8;
        unsafe fn write_changes_to(&mut self, dst: *mut u8, dst_len: usize, reset: bool);
    }
    impl<T: esvo::Serializable> SvoWrapper for esvo::Esvo<T> {
        fn depth(&self) -> u8 { WorldSvo::depth(self) }
        unsafe fn write_changes_to(&mut self, dst: *mut u8, dst_len: usize, reset: bool) { WorldSvo::<T>::write_changes_to(self, dst, dst_len, reset); }
    }
    impl SvoWrapper for csvo::Csvo {
        fn depth(&self) -> u8 { WorldSvo::depth(self) }
        unsafe fn write_changes_to(&mut self, dst: *mut u8, dst_len: usize, reset: bool) { WorldSvo::write_changes_to(self, dst, dst_len, reset); }
    }

    fn create_test_world<F>(svo_type: SvoType, svo_pos: Position, builder: F) -> MappedBuffer<u8>
    where
        F: FnOnce(&mut Chunk),
    {
        let storage_alloc = ChunkStorageAllocator::new();
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0), 5, storage_alloc.allocate());
        builder(&mut chunk);
        chunk.storage.as_mut().unwrap().compact();

        let mut svo: Box<dyn SvoWrapper> = match svo_type {
            SvoType::Esvo => {
                let buffer_alloc = Arc::new(ChunkBufferPool::default());
                let chunk = esvo::SerializedChunk::new(BorrowedChunk::from(chunk), &buffer_alloc);
                let mut svo = esvo::Esvo::<esvo::SerializedChunk>::new();
                svo.set_leaf(svo_pos, chunk, true);
                svo.serialize();
                Box::new(svo)
            }
            SvoType::Csvo => {
                let buffer_alloc = Arc::new(ChunkBufferPool::default());
                let chunk = csvo::SerializedChunk::new(BorrowedChunk::from(chunk), &buffer_alloc);
                let mut svo = csvo::Csvo::new();
                svo.set_leaf(svo_pos, chunk, true);
                svo.serialize();
                Box::new(svo)
            }
        };

        let world_buffer = MappedBuffer::<u8>::new(100 * 1000 * 1000);
        unsafe {
            let max_depth_exp = (-(svo.depth() as f32)).exp2();
            let max_depth_bytes = max_depth_exp.to_bits().to_le_bytes();
            ptr::copy(max_depth_bytes.as_ptr(), world_buffer.cast(), max_depth_bytes.len());

            svo.write_changes_to(world_buffer.offset(4), world_buffer.len() - 1, true);
        }
        world_buffer
    }

    fn create_test_materials() -> (Buffer<MaterialInstance>, Resource<TextureArray, TextureArrayError>) {
        let tex_array = Resource::new(
            || TextureArrayBuilder::new(1, 0.0)
                .add_rgba8("full", 4, 4, vec![
                    255, 000, 000, 255, /**/ 255, 000, 000, 255, /**/ 255, 000, 000, 255, /**/ 255, 000, 000, 255,
                    255, 000, 000, 255, /**/ 255, 000, 000, 255, /**/ 255, 000, 000, 255, /**/ 255, 000, 000, 255,
                    255, 000, 000, 255, /**/ 255, 000, 000, 255, /**/ 255, 000, 000, 255, /**/ 255, 000, 000, 255,
                    255, 000, 000, 255, /**/ 255, 000, 000, 255, /**/ 255, 000, 000, 255, /**/ 255, 000, 000, 255,
                ])?
                .add_rgba8("coords", 4, 4, vec![
                    // 255 * 0.2 = 51
                    000, 153, 000, 255, /**/ 051, 153, 000, 255, /**/ 102, 153, 000, 255, /**/ 153, 153, 000, 255,
                    000, 102, 000, 255, /**/ 051, 102, 000, 255, /**/ 102, 102, 000, 255, /**/ 153, 102, 000, 255,
                    000, 051, 000, 255, /**/ 051, 051, 000, 255, /**/ 102, 051, 000, 255, /**/ 153, 051, 000, 255,
                    000, 000, 000, 255, /**/ 051, 000, 000, 255, /**/ 102, 000, 000, 255, /**/ 153, 000, 000, 255,
                ])?
                .add_rgba8("transparent_1", 4, 4, vec![
                    000, 000, 000, 000, /**/ 000, 000, 000, 000, /**/ 255, 000, 000, 255, /**/ 255, 000, 000, 255,
                    000, 000, 000, 000, /**/ 000, 000, 000, 000, /**/ 255, 000, 000, 255, /**/ 255, 000, 000, 255,
                    000, 000, 000, 000, /**/ 000, 000, 000, 000, /**/ 255, 000, 000, 255, /**/ 255, 000, 000, 255,
                    000, 000, 000, 000, /**/ 000, 000, 000, 000, /**/ 255, 000, 000, 255, /**/ 255, 000, 000, 255,
                ])?
                .add_rgba8("transparent_2", 4, 4, vec![
                    000, 000, 000, 000, /**/ 000, 000, 000, 000, /**/ 000, 255, 000, 255, /**/ 000, 255, 000, 255,
                    000, 000, 000, 000, /**/ 000, 000, 000, 000, /**/ 000, 255, 000, 255, /**/ 000, 255, 000, 255,
                    000, 000, 000, 000, /**/ 000, 000, 000, 000, /**/ 000, 255, 000, 255, /**/ 000, 255, 000, 255,
                    000, 000, 000, 000, /**/ 000, 000, 000, 000, /**/ 000, 255, 000, 255, /**/ 000, 255, 000, 255,
                ])?
                .build()
        ).unwrap();

        let material_buffer = Buffer::new(vec![
            MaterialInstance { // empty
                specular_pow: 0.0,
                specular_strength: 0.0,
                tex_top: -1,
                tex_side: -1,
                tex_bottom: -1,
                tex_top_normal: -1,
                tex_side_normal: -1,
                tex_bottom_normal: -1,
            },
            MaterialInstance { // full
                specular_pow: 0.0,
                specular_strength: 0.0,
                tex_top: tex_array.lookup("full").unwrap() as i32,
                tex_side: tex_array.lookup("full").unwrap() as i32,
                tex_bottom: tex_array.lookup("full").unwrap() as i32,
                tex_top_normal: -1,
                tex_side_normal: -1,
                tex_bottom_normal: -1,
            },
            MaterialInstance { // coords
                specular_pow: 0.0,
                specular_strength: 0.0,
                tex_top: tex_array.lookup("coords").unwrap() as i32,
                tex_side: tex_array.lookup("coords").unwrap() as i32,
                tex_bottom: tex_array.lookup("coords").unwrap() as i32,
                tex_top_normal: -1,
                tex_side_normal: -1,
                tex_bottom_normal: -1,
            },
            MaterialInstance { // transparent_1
                specular_pow: 0.0,
                specular_strength: 0.0,
                tex_top: tex_array.lookup("transparent_1").unwrap() as i32,
                tex_side: tex_array.lookup("transparent_1").unwrap() as i32,
                tex_bottom: tex_array.lookup("transparent_1").unwrap() as i32,
                tex_top_normal: -1,
                tex_side_normal: -1,
                tex_bottom_normal: -1,
            },
            MaterialInstance { // transparent_2
                specular_pow: 0.0,
                specular_strength: 0.0,
                tex_top: tex_array.lookup("transparent_2").unwrap() as i32,
                tex_side: tex_array.lookup("transparent_2").unwrap() as i32,
                tex_bottom: tex_array.lookup("transparent_2").unwrap() as i32,
                tex_top_normal: -1,
                tex_side_normal: -1,
                tex_bottom_normal: -1,
            },
        ], buffer::STATIC_READ);

        (material_buffer, tex_array)
    }

    struct TestSetup {
        _context: GlContext,
        _world_buffer: MappedBuffer<u8>,
        _material_buffer: Buffer<MaterialInstance>,
        _tex_array: Resource<TextureArray, TextureArrayError>,
        shader: Resource<ShaderProgram, ShaderError>,
    }

    fn setup_test<F>(svo_type: SvoType, svo_pos: Option<Position>, world_builder: F) -> TestSetup
    where
        F: FnOnce(&mut Chunk),
    {
        let context = GlContext::new_headless(640, 490);

        let svo_pos = svo_pos.unwrap_or(Position(0, 0, 0));
        let world_buffer = create_test_world(svo_type, svo_pos, world_builder);
        world_buffer.bind_as_storage_buffer(buffer_indices::WORLD);

        let shader = Resource::new(
            move || ShaderProgramBuilder::new().with_define("SVO_TYPE", svo_type.shader_type_define).load_shader_bundle("assets/shaders/svo.test.glsl")?.build()
        ).unwrap();

        let (material_buffer, tex_array) = create_test_materials();
        material_buffer.bind_as_storage_buffer(buffer_indices::MATERIALS);
        shader.bind();
        shader.set_texture("u_texture", 0, &tex_array);
        shader.unbind();

        TestSetup {
            _context: context,
            _world_buffer: world_buffer,
            _material_buffer: material_buffer,
            _tex_array: tex_array,
            shader,
        }
    }

    fn cast_ray(shader: &Resource<ShaderProgram, ShaderError>, pos: Point3<f32>, dir: Vector3<f32>, max_dst: f32, cast_translucent: bool) -> BufferOut {
        let buffer_in = Buffer::new(vec![BufferIn {
            max_dst,
            cast_translucent: AlignedBool::from(cast_translucent),
            pos: AlignedPoint3(pos),
            dir: AlignedVec3(dir.normalize()),
        }], buffer::STATIC_READ);
        buffer_in.bind_as_storage_buffer(buffer_indices::DEBUG_IN);

        let mut buffer_out = Buffer::new(vec![BufferOut {
            result: OctreeResult {
                t: 0.0,
                value: 0,
                face_id: 0,
                pos: AlignedPoint3::new(0.0, 0.0, 0.0),
                uv: AlignedPoint2::new(0.0, 0.0),
                color: AlignedVec4::new(0.0, 0.0, 0.0, 0.0),
                inside_voxel: AlignedBool::from(false),
            },
            stack_ptr: 0,
            stack: [StackFrame {
                t_min: 0.0,
                ptr: 0,
                idx: 0,
                parent_octant_idx: 0,
                scale: 0,
                is_child: AlignedBool::from(false),
                is_leaf: AlignedBool::from(false),
                crossed_boundary: AlignedBool::from(false),
                next_ptr: 0,
            }; 100],
        }], buffer::STATIC_DRAW | buffer::STATIC_READ);
        buffer_out.bind_as_storage_buffer(buffer_indices::DEBUG_OUT);

        unsafe {
            shader.bind();
            gl::DispatchCompute(1, 1, 1);
            gl::MemoryBarrier(gl::ALL_BARRIER_BITS);
            shader.unbind();
        }

        buffer_out.pull_data();
        buffer_out.take().remove(0)
    }

    mod esvo_tests {
        use crate::graphics::svo_shader_tests::tests::*;

        /// Tests if the shader properly traverses through the octree by checking each step the shader
        /// takes. The goal is to verify that the algorithm is capable of performing the basic
        /// PUSH, ADVANCE & POP mechanisms necessary to step through the SVO memory.
        #[test]
        fn shader_svo_traversal() {
            // set up a world with one block at the "end" of the chunk to make sure that the algorithm
            // has to step through many empty children
            let setup = setup_test(SvoType::Esvo, None, |chunk| chunk.set_block(31, 0, 0, 1));
            let buffer_out = cast_ray(
                &setup.shader,
                Point3::new(0.0, 0.5, 0.5),
                Vector3::new(1.0, 0.0, 0.0),
                32.0,
                false,
            );

            println!("total stack frames: {}", buffer_out.stack_ptr + 1);
            for i in 0..=buffer_out.stack_ptr {
                println!("f{}: {:?}", i, buffer_out.stack[i as usize]);
            }
            println!("\n{:?}", buffer_out.result);

            assert_eq!(buffer_out.stack_ptr, 10);
            assert_eq!(&buffer_out.stack[..11], &[
                StackFrame { t_min: 0.0, ptr: 0, idx: 0, parent_octant_idx: 0, scale: 22, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 0.0, ptr: 65, idx: 0, parent_octant_idx: 0, scale: 21, is_child: AlignedBool(0), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 16.0, ptr: 65, idx: 1, parent_octant_idx: 0, scale: 21, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 16.0, ptr: 5, idx: 0, parent_octant_idx: 1, scale: 20, is_child: AlignedBool(0), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 24.0, ptr: 5, idx: 1, parent_octant_idx: 1, scale: 20, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 24.0, ptr: 17, idx: 0, parent_octant_idx: 1, scale: 19, is_child: AlignedBool(0), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 28.0, ptr: 17, idx: 1, parent_octant_idx: 1, scale: 19, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 28.0, ptr: 29, idx: 0, parent_octant_idx: 1, scale: 18, is_child: AlignedBool(0), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 30.0, ptr: 29, idx: 1, parent_octant_idx: 1, scale: 18, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 30.0, ptr: 41, idx: 0, parent_octant_idx: 1, scale: 17, is_child: AlignedBool(0), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 31.0, ptr: 41, idx: 1, parent_octant_idx: 1, scale: 17, is_child: AlignedBool(1), is_leaf: AlignedBool(1), crossed_boundary: AlignedBool(0), next_ptr: 0 },
            ]);
            assert_eq!(buffer_out.result, OctreeResult {
                t: 31.0,
                value: 1,
                face_id: 0,
                pos: assert_vec3_eq!(buffer_out.result.pos, AlignedPoint3::new(31.000008, 0.5, 0.5)),
                uv: AlignedPoint2::new(0.5, 0.5),
                color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                inside_voxel: AlignedBool::from(false),
            });
        }

        /// Tests if casting along every axis yields the expected result. Per axis, it checks for both
        /// positive and negative direction as well as being inside and outside the root octant. The
        /// final cases combine all axis into a diagonal raycast.
        #[test]
        fn cast_inside_outside_all_axes() {
            // place blocks at 30 instead of 31 to have space for inside casts for every axis
            let setup = setup_test(SvoType::Esvo, None, |chunk| {
                chunk.set_block(30, 0, 0, 1);
                chunk.set_block(0, 30, 0, 1);
                chunk.set_block(0, 0, 30, 1);
                chunk.set_block(30, 30, 30, 1);
            });

            struct TestCase {
                name: &'static str,
                pos: Point3<f32>,
                dir: Vector3<f32>,
                expected: OctreeResult,
            }
            let cases = vec![
                TestCase {
                    name: "x pos",
                    pos: Point3::new(0.5, 0.5, 0.5),
                    dir: Vector3::new(1.0, 0.0, 0.0),
                    expected: OctreeResult {
                        t: 29.5,
                        value: 1,
                        face_id: 0,
                        pos: AlignedPoint3::new(30.000008, 0.5, 0.5),
                        uv: AlignedPoint2::new(0.5, 0.5),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
                TestCase {
                    name: "x neg",
                    pos: Point3::new(31.5, 0.5, 0.5),
                    dir: Vector3::new(-1.0, 0.0, 0.0),
                    expected: OctreeResult {
                        t: 0.5,
                        value: 1,
                        face_id: 1,
                        pos: AlignedPoint3::new(30.999992, 0.5, 0.5),
                        uv: AlignedPoint2::new(0.5, 0.5),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
                TestCase {
                    name: "y pos",
                    pos: Point3::new(0.5, 0.5, 0.5),
                    dir: Vector3::new(0.0, 1.0, 0.0),
                    expected: OctreeResult {
                        t: 29.5,
                        value: 1,
                        face_id: 2,
                        pos: AlignedPoint3::new(0.5, 30.000008, 0.5),
                        uv: AlignedPoint2::new(0.5, 0.5),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
                TestCase {
                    name: "y neg",
                    pos: Point3::new(0.5, 31.5, 0.5),
                    dir: Vector3::new(0.0, -1.0, 0.0),
                    expected: OctreeResult {
                        t: 0.5,
                        value: 1,
                        face_id: 3,
                        pos: AlignedPoint3::new(0.5, 30.999992, 0.5),
                        uv: AlignedPoint2::new(0.5, 0.5),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
                TestCase {
                    name: "z pos",
                    pos: Point3::new(0.5, 0.5, 0.5),
                    dir: Vector3::new(0.0, 0.0, 1.0),
                    expected: OctreeResult {
                        t: 29.5,
                        value: 1,
                        face_id: 4,
                        pos: AlignedPoint3::new(0.5, 0.5, 30.000008),
                        uv: AlignedPoint2::new(0.5, 0.5),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
                TestCase {
                    name: "z neg",
                    pos: Point3::new(0.5, 0.5, 31.5),
                    dir: Vector3::new(0.0, 0.0, -1.0),
                    expected: OctreeResult {
                        t: 0.5,
                        value: 1,
                        face_id: 5,
                        pos: AlignedPoint3::new(0.5, 0.5, 30.999992),
                        uv: AlignedPoint2::new(0.5, 0.5),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
                TestCase {
                    name: "diagonal pos",
                    pos: Point3::new(0.6, 0.5, 0.6),
                    dir: Vector3::new(1.0, 1.0, 1.0),
                    expected: OctreeResult {
                        t: 51.095497,
                        value: 1,
                        face_id: 2,
                        pos: AlignedPoint3::new(30.099998, 30.000008, 30.099998),
                        uv: AlignedPoint2::new(0.099998474, 0.9000015),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
                TestCase {
                    name: "diagonal neg",
                    pos: Point3::new(31.4, 31.5, 31.4),
                    dir: Vector3::new(-1.0, -1.0, -1.0),
                    expected: OctreeResult {
                        t: 0.86602306,
                        value: 1,
                        face_id: 3,
                        pos: AlignedPoint3::new(30.900002, 30.999992, 30.900002),
                        uv: AlignedPoint2::new(0.9000015, 0.9000015),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
            ];
            let derive_expected = |actual: &OctreeResult, expected: &OctreeResult| OctreeResult {
                t: assert_float_eq!(actual.t, expected.t),
                value: expected.value,
                face_id: expected.face_id,
                pos: assert_vec3_eq!(actual.pos, expected.pos),
                uv: assert_vec2_eq!(actual.uv, expected.uv),
                color: assert_vec4_eq!(actual.color, expected.color),
                inside_voxel: expected.inside_voxel,
            };
            for case in cases {
                let buffer_out = cast_ray(&setup.shader, case.pos, case.dir, 100.0, false);
                assert_eq!(buffer_out.result, derive_expected(&buffer_out.result, &case.expected), "test case \"{}\" inside", case.name);

                let mut case = case;
                case.pos -= case.dir.normalize();
                case.expected.t += 1.0;

                let buffer_out = cast_ray(&setup.shader, case.pos, case.dir, 100.0, false);
                assert_eq!(buffer_out.result, derive_expected(&buffer_out.result, &case.expected), "test case \"{}\" outside", case.name);
            }
        }

        /// Tests if uv calculation and texture lookup works properly on every side of a voxel. This
        /// uses a special texture that maps a simple 2d coordinate system with a resolution of 4x4
        /// onto each side along the uv space.
        #[test]
        fn uv_coords_on_all_sides() {
            let setup = setup_test(SvoType::Esvo, None, |chunk| chunk.set_block(0, 0, 0, 2));

            struct TestCase {
                pos: Point3<f32>,
                dir: Vector3<f32>,
                expected_uv: Point2<f32>,
                expected_color: Vector4<f32>,
            }
            let cases = vec![
                // pos z
                TestCase {
                    pos: Point3::new(0.1, 0.1, -0.1),
                    dir: Vector3::new(0.0, 0.0, 1.0),
                    expected_uv: Point2::new(0.1, 0.1),
                    expected_color: Vector4::new(0.0, 0.0, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(0.1, 0.5, -0.1),
                    dir: Vector3::new(0.0, 0.0, 1.0),
                    expected_uv: Point2::new(0.1, 0.5),
                    expected_color: Vector4::new(0.0, 0.4, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(0.5, 0.1, -0.1),
                    dir: Vector3::new(0.0, 0.0, 1.0),
                    expected_uv: Point2::new(0.5, 0.1),
                    expected_color: Vector4::new(0.4, 0.0, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(0.5, 0.5, -0.1),
                    dir: Vector3::new(0.0, 0.0, 1.0),
                    expected_uv: Point2::new(0.5, 0.5),
                    expected_color: Vector4::new(0.4, 0.4, 0.0, 1.0),
                },
                // neg z
                TestCase {
                    pos: Point3::new(0.1, 0.1, 1.1),
                    dir: Vector3::new(0.0, 0.0, -1.0),
                    expected_uv: Point2::new(0.9, 0.1),
                    expected_color: Vector4::new(0.6, 0.0, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(0.1, 0.5, 1.1),
                    dir: Vector3::new(0.0, 0.0, -1.0),
                    expected_uv: Point2::new(0.9, 0.5),
                    expected_color: Vector4::new(0.6, 0.4, 0.0, 1.0),
                },
                // pos x
                TestCase {
                    pos: Point3::new(-0.1, 0.1, 0.1),
                    dir: Vector3::new(1.0, 0.0, 0.0),
                    expected_uv: Point2::new(0.9, 0.1),
                    expected_color: Vector4::new(0.6, 0.0, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(-0.1, 0.5, 0.1),
                    dir: Vector3::new(1.0, 0.0, 0.0),
                    expected_uv: Point2::new(0.9, 0.5),
                    expected_color: Vector4::new(0.6, 0.4, 0.0, 1.0),
                },
                // neg x
                TestCase {
                    pos: Point3::new(1.1, 0.1, 0.1),
                    dir: Vector3::new(-1.0, 0.0, 0.0),
                    expected_uv: Point2::new(0.1, 0.1),
                    expected_color: Vector4::new(0.0, 0.0, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(1.1, 0.5, 0.1),
                    dir: Vector3::new(-1.0, 0.0, 0.0),
                    expected_uv: Point2::new(0.1, 0.5),
                    expected_color: Vector4::new(0.0, 0.4, 0.0, 1.0),
                },
                // pos y
                TestCase {
                    pos: Point3::new(0.1, -0.1, 0.1),
                    dir: Vector3::new(0.0, 1.0, 0.0),
                    expected_uv: Point2::new(0.1, 0.9),
                    expected_color: Vector4::new(0.0, 0.6, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(0.1, -0.1, 0.5),
                    dir: Vector3::new(0.0, 1.0, 0.0),
                    expected_uv: Point2::new(0.1, 0.5),
                    expected_color: Vector4::new(0.0, 0.4, 0.0, 1.0),
                },
                // neg y
                TestCase {
                    pos: Point3::new(0.1, 1.1, 0.1),
                    dir: Vector3::new(0.0, -1.0, 0.0),
                    expected_uv: Point2::new(0.1, 0.1),
                    expected_color: Vector4::new(0.0, 0.0, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(0.1, 1.1, 0.5),
                    dir: Vector3::new(0.0, -1.0, 0.0),
                    expected_uv: Point2::new(0.1, 0.5),
                    expected_color: Vector4::new(0.0, 0.4, 0.0, 1.0),
                },
            ];
            for (i, case) in cases.iter().enumerate() {
                let buffer_out = cast_ray(&setup.shader, case.pos, case.dir, 32.0, false);
                let case_name = format!("#{} (pos={:?}, dir={:?})", i, case.pos, case.dir);

                println!("{case_name}");
                assert_vec2_eq!(buffer_out.result.uv, case.expected_uv);
                assert_vec4_eq!(buffer_out.result.color, case.expected_color);
            }
        }

        /// Tests if translucency is properly accounted for during ray casting. Assert that identical,
        /// adjacent voxels are skipped and make sure that `cast_translucent` flag is respected.
        #[test]
        fn casting_against_translucent_leafs() {
            // This setup has to small rows of adjacent blocks. The first row consists of the same
            // blocks while the second one has two different kinds. Both kinds have textures that
            // are transparent on the left half and opaque on the right half. This allows testing
            // the algorithm's behaviour when casting diagonally through them.
            let setup = setup_test(SvoType::Esvo, None, |chunk| {
                chunk.set_block(0, 0, 0, 3);
                chunk.set_block(0, 0, 1, 3);
                chunk.set_block(5, 0, 0, 3);
                chunk.set_block(5, 0, 1, 4);
            });

            let dir = Point3::new(0.75, 0.5, 1.0) - Point3::new(0.25, 0.5, -0.1);

            // do not cast translucent
            let buffer_out = cast_ray(&setup.shader, Point3::new(0.25, 0.5, -0.1), dir, 32.0, false);
            assert_eq!(buffer_out.result, OctreeResult {
                t: assert_float_eq!(buffer_out.result.t, 0.1, 0.01),
                value: 3,
                face_id: 4,
                pos: assert_vec3_eq!(buffer_out.result.pos, AlignedPoint3::new(0.295, 0.5, 0.0), 0.01),
                uv: assert_vec2_eq!(buffer_out.result.uv, AlignedPoint2::new(0.295, 0.5), 0.01),
                color: AlignedVec4::new(0.0, 0.0, 0.0, 0.0),
                inside_voxel: AlignedBool::from(false),
            }, "do not cast translucent");

            // cast translucent with adjacent identical
            let buffer_out = cast_ray(&setup.shader, Point3::new(0.25, 0.5, -0.1), dir, 32.0, true);
            assert_eq!(buffer_out.result, OctreeResult {
                t: -1.0,
                value: 0,
                face_id: 0,
                pos: AlignedPoint3::new(0.0, 0.0, 0.0),
                uv: AlignedPoint2::new(0.0, 0.0),
                color: AlignedVec4::new(0.0, 0.0, 0.0, 0.0),
                inside_voxel: AlignedBool::from(false),
            }, "cast translucent with adjacent identical");

            // cast translucent with adjacent different
            let buffer_out = cast_ray(&setup.shader, Point3::new(5.25, 0.5, -0.1), dir, 32.0, true);
            assert_eq!(buffer_out.result, OctreeResult {
                t: assert_float_eq!(buffer_out.result.t, 1.2, 0.01),
                value: 4,
                face_id: 4,
                pos: assert_vec3_eq!(buffer_out.result.pos, AlignedPoint3::new(5.75, 0.5, 1.0), 0.01),
                uv: assert_vec2_eq!(buffer_out.result.uv, AlignedPoint2::new(0.75, 0.5), 0.01),
                color: AlignedVec4::new(0.0, 1.0, 0.0, 1.0),
                inside_voxel: AlignedBool::from(false),
            }, "cast translucent with adjacent different");
        }

        /// Tests if the algorithm correctly detects starting from within a voxel and that it does not
        /// intersect with that voxel.
        #[test]
        fn detect_inside_leaf_voxel() {
            let setup = setup_test(SvoType::Esvo, None, |chunk| chunk.set_block(0, 0, 0, 1));

            // inside block
            let buffer_out = cast_ray(
                &setup.shader,
                Point3::new(0.5, 0.5, 0.5),
                Vector3::new(1.0, 0.0, 0.0),
                32.0,
                false,
            );
            assert_eq!(buffer_out.result, OctreeResult {
                t: -1.0,
                value: 0,
                face_id: 0,
                pos: AlignedPoint3::new(0.0, 0.0, 0.0),
                uv: AlignedPoint2::new(0.0, 0.0),
                color: AlignedVec4::new(0.0, 0.0, 0.0, 0.0),
                inside_voxel: AlignedBool::from(true),
            }, "inside block");

            // outside block
            let buffer_out = cast_ray(
                &setup.shader,
                Point3::new(-0.5, 0.5, 0.5),
                Vector3::new(1.0, 0.0, 0.0),
                32.0,
                false,
            );
            assert_eq!(buffer_out.result, OctreeResult {
                t: 0.5,
                value: 1,
                face_id: 0,
                pos: assert_vec3_eq!(buffer_out.result.pos, AlignedPoint3::new(8e-6, 0.5, 0.5)),
                uv: AlignedPoint2::new(0.5, 0.5),
                color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                inside_voxel: AlignedBool::from(false),
            }, "outside block");
        }

        /// Tests large coordinates at the upper end of the SVO bounds. This caused problems before due
        /// to division by zero errors inside the traversal algorithm because the epsilon values were
        /// not properly applied on axes of the direction vector with a value of 0.0.
        #[test]
        fn check_at_higher_coordinates() {
            let setup = setup_test(SvoType::Esvo, Some(Position(15, 15, 15)), |chunk| {
                for x in 0..32 {
                    for z in 0..32 {
                        for y in 0..5 {
                            chunk.set_block(x, y, z, 1);
                        }
                    }
                }
            });
            let buffer_out = cast_ray(
                &setup.shader,
                Point3::new(484.9203, 485.95938, 493.8467),
                Vector3::new(0.0, -1.0, 0.0),
                10.0,
                false,
            );

            println!("total stack frames: {}", buffer_out.stack_ptr + 1);
            for i in 0..=buffer_out.stack_ptr {
                println!("f{}: {:?}", i, buffer_out.stack[i as usize]);
            }
            println!("\n{:?}", buffer_out.result);

            assert_eq!(buffer_out.stack_ptr, 9);
            assert_eq!(&buffer_out.stack[..10], &[
                StackFrame { t_min: 0.0, ptr: 0, idx: 7, parent_octant_idx: 0, scale: 22, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 0.0, ptr: 11009, idx: 7, parent_octant_idx: 7, scale: 21, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 0.0, ptr: 11057, idx: 7, parent_octant_idx: 7, scale: 20, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 0.0, ptr: 11069, idx: 7, parent_octant_idx: 7, scale: 19, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 0.0, ptr: 11081, idx: 0, parent_octant_idx: 7, scale: 18, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 0.0, ptr: 5, idx: 4, parent_octant_idx: 0, scale: 17, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 0.0, ptr: 17, idx: 7, parent_octant_idx: 4, scale: 16, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 0.0, ptr: 1397, idx: 0, parent_octant_idx: 7, scale: 15, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: 0.0, ptr: 2021, idx: 6, parent_octant_idx: 0, scale: 14, is_child: AlignedBool(0), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 0 },
                StackFrame { t_min: assert_float_eq!(buffer_out.stack[9].t_min, 0.9593506), ptr: 2021, idx: 4, parent_octant_idx: 0, scale: 14, is_child: AlignedBool(1), is_leaf: AlignedBool(1), crossed_boundary: AlignedBool(0), next_ptr: 0 },
            ]);
            assert_eq!(buffer_out.result, OctreeResult {
                t: assert_float_eq!(buffer_out.result.t, 0.9593506),
                value: 1,
                face_id: 3,
                pos: assert_vec3_eq!(buffer_out.result.pos, AlignedPoint3::new(484.9203, 484.99994, 493.84668)),
                uv: assert_vec2_eq!(buffer_out.result.uv,AlignedPoint2::new(0.9202881, 0.8466797)),
                color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                inside_voxel: AlignedBool::from(false),
            });
        }
    }

    mod csvo_tests {
        use crate::graphics::svo_shader_tests::tests::*;

        /// Tests if the shader properly traverses through the octree by checking each step the shader
        /// takes. The goal is to verify that the algorithm is capable of performing the basic
        /// PUSH, ADVANCE & POP mechanisms necessary to step through the SVO memory.
        #[test]
        fn shader_svo_traversal() {
            // set up a world with one block at the "end" of the chunk to make sure that the algorithm
            // has to step through many empty children
            let setup = setup_test(SvoType::Csvo, None, |chunk| chunk.set_block(31, 0, 0, 1));
            let buffer_out = cast_ray(
                &setup.shader,
                Point3::new(0.0, 0.5, 0.5),
                Vector3::new(1.0, 0.0, 0.0),
                32.0,
                false,
            );

            println!("total stack frames: {}", buffer_out.stack_ptr + 1);
            for i in 0..=buffer_out.stack_ptr {
                println!("f{}: {:?}", i, buffer_out.stack[i as usize]);
            }
            println!("\n{:?}", buffer_out.result);

            assert_eq!(buffer_out.stack_ptr, 10);
            assert_eq!(&buffer_out.stack[..11], &[
                StackFrame { t_min: 0.0, ptr: 21, idx: 0, parent_octant_idx: 6, scale: 22, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(1), next_ptr: 0 },
                StackFrame { t_min: 0.0, ptr: 9, idx: 0, parent_octant_idx: 5, scale: 21, is_child: AlignedBool(0), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 4294967295 },
                StackFrame { t_min: 16.0, ptr: 9, idx: 1, parent_octant_idx: 5, scale: 21, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 12 },
                StackFrame { t_min: 16.0, ptr: 12, idx: 0, parent_octant_idx: 4, scale: 20, is_child: AlignedBool(0), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 4294967295 },
                StackFrame { t_min: 24.0, ptr: 12, idx: 1, parent_octant_idx: 4, scale: 20, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 15 },
                StackFrame { t_min: 24.0, ptr: 15, idx: 0, parent_octant_idx: 3, scale: 19, is_child: AlignedBool(0), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 4294967295 },
                StackFrame { t_min: 28.0, ptr: 15, idx: 1, parent_octant_idx: 3, scale: 19, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 17 },
                StackFrame { t_min: 28.0, ptr: 17, idx: 0, parent_octant_idx: 2, scale: 18, is_child: AlignedBool(0), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 4294967295 },
                StackFrame { t_min: 30.0, ptr: 17, idx: 1, parent_octant_idx: 2, scale: 18, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 20 },
                StackFrame { t_min: 30.0, ptr: 20, idx: 0, parent_octant_idx: 1, scale: 17, is_child: AlignedBool(0), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 4294967295 },
                StackFrame { t_min: 31.0, ptr: 20, idx: 1, parent_octant_idx: 1, scale: 17, is_child: AlignedBool(1), is_leaf: AlignedBool(1), crossed_boundary: AlignedBool(0), next_ptr: 23 },
            ]);
            assert_eq!(buffer_out.result, OctreeResult {
                t: 31.0,
                value: 1,
                face_id: 0,
                pos: assert_vec3_eq!(buffer_out.result.pos, AlignedPoint3::new(31.000008, 0.5, 0.5)),
                uv: AlignedPoint2::new(0.5, 0.5),
                color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                inside_voxel: AlignedBool::from(false),
            });
        }

        /// Tests if casting along every axis yields the expected result. Per axis, it checks for both
        /// positive and negative direction as well as being inside and outside the root octant. The
        /// final cases combine all axis into a diagonal raycast.
        #[test]
        fn cast_inside_outside_all_axes() {
            // place blocks at 30 instead of 31 to have space for inside casts for every axis
            let setup = setup_test(SvoType::Csvo, None, |chunk| {
                chunk.set_block(30, 0, 0, 1);
                chunk.set_block(0, 30, 0, 1);
                chunk.set_block(0, 0, 30, 1);
                chunk.set_block(30, 30, 30, 1);
            });

            struct TestCase {
                name: &'static str,
                pos: Point3<f32>,
                dir: Vector3<f32>,
                expected: OctreeResult,
            }
            let cases = vec![
                TestCase {
                    name: "x pos",
                    pos: Point3::new(0.5, 0.5, 0.5),
                    dir: Vector3::new(1.0, 0.0, 0.0),
                    expected: OctreeResult {
                        t: 29.5,
                        value: 1,
                        face_id: 0,
                        pos: AlignedPoint3::new(30.000008, 0.5, 0.5),
                        uv: AlignedPoint2::new(0.5, 0.5),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
                TestCase {
                    name: "x neg",
                    pos: Point3::new(31.5, 0.5, 0.5),
                    dir: Vector3::new(-1.0, 0.0, 0.0),
                    expected: OctreeResult {
                        t: 0.5,
                        value: 1,
                        face_id: 1,
                        pos: AlignedPoint3::new(30.999992, 0.5, 0.5),
                        uv: AlignedPoint2::new(0.5, 0.5),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
                TestCase {
                    name: "y pos",
                    pos: Point3::new(0.5, 0.5, 0.5),
                    dir: Vector3::new(0.0, 1.0, 0.0),
                    expected: OctreeResult {
                        t: 29.5,
                        value: 1,
                        face_id: 2,
                        pos: AlignedPoint3::new(0.5, 30.000008, 0.5),
                        uv: AlignedPoint2::new(0.5, 0.5),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
                TestCase {
                    name: "y neg",
                    pos: Point3::new(0.5, 31.5, 0.5),
                    dir: Vector3::new(0.0, -1.0, 0.0),
                    expected: OctreeResult {
                        t: 0.5,
                        value: 1,
                        face_id: 3,
                        pos: AlignedPoint3::new(0.5, 30.999992, 0.5),
                        uv: AlignedPoint2::new(0.5, 0.5),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
                TestCase {
                    name: "z pos",
                    pos: Point3::new(0.5, 0.5, 0.5),
                    dir: Vector3::new(0.0, 0.0, 1.0),
                    expected: OctreeResult {
                        t: 29.5,
                        value: 1,
                        face_id: 4,
                        pos: AlignedPoint3::new(0.5, 0.5, 30.000008),
                        uv: AlignedPoint2::new(0.5, 0.5),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
                TestCase {
                    name: "z neg",
                    pos: Point3::new(0.5, 0.5, 31.5),
                    dir: Vector3::new(0.0, 0.0, -1.0),
                    expected: OctreeResult {
                        t: 0.5,
                        value: 1,
                        face_id: 5,
                        pos: AlignedPoint3::new(0.5, 0.5, 30.999992),
                        uv: AlignedPoint2::new(0.5, 0.5),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
                TestCase {
                    name: "diagonal pos",
                    pos: Point3::new(0.6, 0.5, 0.6),
                    dir: Vector3::new(1.0, 1.0, 1.0),
                    expected: OctreeResult {
                        t: 51.095497,
                        value: 1,
                        face_id: 2,
                        pos: AlignedPoint3::new(30.099998, 30.000008, 30.099998),
                        uv: AlignedPoint2::new(0.099998474, 0.9000015),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
                TestCase {
                    name: "diagonal neg",
                    pos: Point3::new(31.4, 31.5, 31.4),
                    dir: Vector3::new(-1.0, -1.0, -1.0),
                    expected: OctreeResult {
                        t: 0.86602306,
                        value: 1,
                        face_id: 3,
                        pos: AlignedPoint3::new(30.900002, 30.999992, 30.900002),
                        uv: AlignedPoint2::new(0.9000015, 0.9000015),
                        color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                        inside_voxel: AlignedBool::from(false),
                    },
                },
            ];
            let derive_expected = |actual: &OctreeResult, expected: &OctreeResult| OctreeResult {
                t: assert_float_eq!(actual.t, expected.t),
                value: expected.value,
                face_id: expected.face_id,
                pos: assert_vec3_eq!(actual.pos, expected.pos),
                uv: assert_vec2_eq!(actual.uv, expected.uv),
                color: assert_vec4_eq!(actual.color, expected.color),
                inside_voxel: expected.inside_voxel,
            };
            for case in cases {
                let buffer_out = cast_ray(&setup.shader, case.pos, case.dir, 100.0, false);
                assert_eq!(buffer_out.result, derive_expected(&buffer_out.result, &case.expected), "test case \"{}\" inside", case.name);

                let mut case = case;
                case.pos -= case.dir.normalize();
                case.expected.t += 1.0;

                let buffer_out = cast_ray(&setup.shader, case.pos, case.dir, 100.0, false);
                assert_eq!(buffer_out.result, derive_expected(&buffer_out.result, &case.expected), "test case \"{}\" outside", case.name);
            }
        }

        /// Tests if uv calculation and texture lookup works properly on every side of a voxel. This
        /// uses a special texture that maps a simple 2d coordinate system with a resolution of 4x4
        /// onto each side along the uv space.
        #[test]
        fn uv_coords_on_all_sides() {
            let setup = setup_test(SvoType::Csvo, None, |chunk| chunk.set_block(0, 0, 0, 2));

            struct TestCase {
                pos: Point3<f32>,
                dir: Vector3<f32>,
                expected_uv: Point2<f32>,
                expected_color: Vector4<f32>,
            }
            let cases = vec![
                // pos z
                TestCase {
                    pos: Point3::new(0.1, 0.1, -0.1),
                    dir: Vector3::new(0.0, 0.0, 1.0),
                    expected_uv: Point2::new(0.1, 0.1),
                    expected_color: Vector4::new(0.0, 0.0, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(0.1, 0.5, -0.1),
                    dir: Vector3::new(0.0, 0.0, 1.0),
                    expected_uv: Point2::new(0.1, 0.5),
                    expected_color: Vector4::new(0.0, 0.4, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(0.5, 0.1, -0.1),
                    dir: Vector3::new(0.0, 0.0, 1.0),
                    expected_uv: Point2::new(0.5, 0.1),
                    expected_color: Vector4::new(0.4, 0.0, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(0.5, 0.5, -0.1),
                    dir: Vector3::new(0.0, 0.0, 1.0),
                    expected_uv: Point2::new(0.5, 0.5),
                    expected_color: Vector4::new(0.4, 0.4, 0.0, 1.0),
                },
                // neg z
                TestCase {
                    pos: Point3::new(0.1, 0.1, 1.1),
                    dir: Vector3::new(0.0, 0.0, -1.0),
                    expected_uv: Point2::new(0.9, 0.1),
                    expected_color: Vector4::new(0.6, 0.0, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(0.1, 0.5, 1.1),
                    dir: Vector3::new(0.0, 0.0, -1.0),
                    expected_uv: Point2::new(0.9, 0.5),
                    expected_color: Vector4::new(0.6, 0.4, 0.0, 1.0),
                },
                // pos x
                TestCase {
                    pos: Point3::new(-0.1, 0.1, 0.1),
                    dir: Vector3::new(1.0, 0.0, 0.0),
                    expected_uv: Point2::new(0.9, 0.1),
                    expected_color: Vector4::new(0.6, 0.0, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(-0.1, 0.5, 0.1),
                    dir: Vector3::new(1.0, 0.0, 0.0),
                    expected_uv: Point2::new(0.9, 0.5),
                    expected_color: Vector4::new(0.6, 0.4, 0.0, 1.0),
                },
                // neg x
                TestCase {
                    pos: Point3::new(1.1, 0.1, 0.1),
                    dir: Vector3::new(-1.0, 0.0, 0.0),
                    expected_uv: Point2::new(0.1, 0.1),
                    expected_color: Vector4::new(0.0, 0.0, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(1.1, 0.5, 0.1),
                    dir: Vector3::new(-1.0, 0.0, 0.0),
                    expected_uv: Point2::new(0.1, 0.5),
                    expected_color: Vector4::new(0.0, 0.4, 0.0, 1.0),
                },
                // pos y
                TestCase {
                    pos: Point3::new(0.1, -0.1, 0.1),
                    dir: Vector3::new(0.0, 1.0, 0.0),
                    expected_uv: Point2::new(0.1, 0.9),
                    expected_color: Vector4::new(0.0, 0.6, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(0.1, -0.1, 0.5),
                    dir: Vector3::new(0.0, 1.0, 0.0),
                    expected_uv: Point2::new(0.1, 0.5),
                    expected_color: Vector4::new(0.0, 0.4, 0.0, 1.0),
                },
                // neg y
                TestCase {
                    pos: Point3::new(0.1, 1.1, 0.1),
                    dir: Vector3::new(0.0, -1.0, 0.0),
                    expected_uv: Point2::new(0.1, 0.1),
                    expected_color: Vector4::new(0.0, 0.0, 0.0, 1.0),
                },
                TestCase {
                    pos: Point3::new(0.1, 1.1, 0.5),
                    dir: Vector3::new(0.0, -1.0, 0.0),
                    expected_uv: Point2::new(0.1, 0.5),
                    expected_color: Vector4::new(0.0, 0.4, 0.0, 1.0),
                },
            ];
            for (i, case) in cases.iter().enumerate() {
                let buffer_out = cast_ray(&setup.shader, case.pos, case.dir, 32.0, false);
                let case_name = format!("#{} (pos={:?}, dir={:?})", i, case.pos, case.dir);

                println!("{case_name}");
                assert_vec2_eq!(buffer_out.result.uv, case.expected_uv);
                assert_vec4_eq!(buffer_out.result.color, case.expected_color);
            }
        }

        /// Tests if translucency is properly accounted for during ray casting. Assert that identical,
        /// adjacent voxels are skipped and make sure that `cast_translucent` flag is respected.
        #[test]
        fn casting_against_translucent_leafs() {
            // This setup has to small rows of adjacent blocks. The first row consists of the same
            // blocks while the second one has two different kinds. Both kinds have textures that
            // are transparent on the left half and opaque on the right half. This allows testing
            // the algorithm's behaviour when casting diagonally through them.
            let setup = setup_test(SvoType::Csvo, None, |chunk| {
                chunk.set_block(0, 0, 0, 3);
                chunk.set_block(0, 0, 1, 3);
                chunk.set_block(5, 0, 0, 3);
                chunk.set_block(5, 0, 1, 4);
            });

            let dir = Point3::new(0.75, 0.5, 1.0) - Point3::new(0.25, 0.5, -0.1);

            // do not cast translucent
            let buffer_out = cast_ray(&setup.shader, Point3::new(0.25, 0.5, -0.1), dir, 32.0, false);
            assert_eq!(buffer_out.result, OctreeResult {
                t: assert_float_eq!(buffer_out.result.t, 0.1, 0.01),
                value: 3,
                face_id: 4,
                pos: assert_vec3_eq!(buffer_out.result.pos, AlignedPoint3::new(0.295, 0.5, 0.0), 0.01),
                uv: assert_vec2_eq!(buffer_out.result.uv, AlignedPoint2::new(0.295, 0.5), 0.01),
                color: AlignedVec4::new(0.0, 0.0, 0.0, 0.0),
                inside_voxel: AlignedBool::from(false),
            }, "do not cast translucent");

            // cast translucent with adjacent identical
            let buffer_out = cast_ray(&setup.shader, Point3::new(0.25, 0.5, -0.1), dir, 32.0, true);
            assert_eq!(buffer_out.result, OctreeResult {
                t: -1.0,
                value: 0,
                face_id: 0,
                pos: AlignedPoint3::new(0.0, 0.0, 0.0),
                uv: AlignedPoint2::new(0.0, 0.0),
                color: AlignedVec4::new(0.0, 0.0, 0.0, 0.0),
                inside_voxel: AlignedBool::from(false),
            }, "cast translucent with adjacent identical");

            // cast translucent with adjacent different
            let buffer_out = cast_ray(&setup.shader, Point3::new(5.25, 0.5, -0.1), dir, 32.0, true);
            assert_eq!(buffer_out.result, OctreeResult {
                t: assert_float_eq!(buffer_out.result.t, 1.2, 0.01),
                value: 4,
                face_id: 4,
                pos: assert_vec3_eq!(buffer_out.result.pos, AlignedPoint3::new(5.75, 0.5, 1.0), 0.01),
                uv: assert_vec2_eq!(buffer_out.result.uv, AlignedPoint2::new(0.75, 0.5), 0.01),
                color: AlignedVec4::new(0.0, 1.0, 0.0, 1.0),
                inside_voxel: AlignedBool::from(false),
            }, "cast translucent with adjacent different");
        }

        /// Tests if the algorithm correctly detects starting from within a voxel and that it does not
        /// intersect with that voxel.
        #[test]
        fn detect_inside_leaf_voxel() {
            let setup = setup_test(SvoType::Csvo, None, |chunk| chunk.set_block(0, 0, 0, 1));

            // inside block
            let buffer_out = cast_ray(
                &setup.shader,
                Point3::new(0.5, 0.5, 0.5),
                Vector3::new(1.0, 0.0, 0.0),
                32.0,
                false,
            );
            assert_eq!(buffer_out.result, OctreeResult {
                t: -1.0,
                value: 0,
                face_id: 0,
                pos: AlignedPoint3::new(0.0, 0.0, 0.0),
                uv: AlignedPoint2::new(0.0, 0.0),
                color: AlignedVec4::new(0.0, 0.0, 0.0, 0.0),
                inside_voxel: AlignedBool::from(true),
            }, "inside block");

            // outside block
            let buffer_out = cast_ray(
                &setup.shader,
                Point3::new(-0.5, 0.5, 0.5),
                Vector3::new(1.0, 0.0, 0.0),
                32.0,
                false,
            );
            assert_eq!(buffer_out.result, OctreeResult {
                t: 0.5,
                value: 1,
                face_id: 0,
                pos: assert_vec3_eq!(buffer_out.result.pos, AlignedPoint3::new(8e-6, 0.5, 0.5)),
                uv: AlignedPoint2::new(0.5, 0.5),
                color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                inside_voxel: AlignedBool::from(false),
            }, "outside block");
        }

        /// Tests large coordinates at the upper end of the SVO bounds. This caused problems before due
        /// to division by zero errors inside the traversal algorithm because the epsilon values were
        /// not properly applied on axes of the direction vector with a value of 0.0.
        #[test]
        fn check_at_higher_coordinates() {
            let setup = setup_test(SvoType::Csvo, Some(Position(15, 15, 15)), |chunk| {
                for x in 0..32 {
                    for z in 0..32 {
                        for y in 0..5 {
                            chunk.set_block(x, y, z, 1);
                        }
                    }
                }
            });
            let buffer_out = cast_ray(
                &setup.shader,
                Point3::new(484.9203, 485.95938, 493.8467),
                Vector3::new(0.0, -1.0, 0.0),
                10.0,
                false,
            );

            println!("total stack frames: {}", buffer_out.stack_ptr + 1);
            for i in 0..=buffer_out.stack_ptr {
                println!("f{}: {:?}", i, buffer_out.stack[i as usize]);
            }
            println!("\n{:?}", buffer_out.result);

            assert_eq!(buffer_out.stack_ptr, 9);
            assert_eq!(&buffer_out.stack[..10], &[
                StackFrame { t_min: 0.0, ptr: 21814, idx: 7, parent_octant_idx: 9, scale: 22, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 21826 },
                StackFrame { t_min: 0.0, ptr: 21826, idx: 7, parent_octant_idx: 8, scale: 21, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 21829 },
                StackFrame { t_min: 0.0, ptr: 21829, idx: 7, parent_octant_idx: 7, scale: 20, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 21832 },
                StackFrame { t_min: 0.0, ptr: 21832, idx: 7, parent_octant_idx: 6, scale: 19, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(1), next_ptr: 0 },
                StackFrame { t_min: 0.0, ptr: 20485, idx: 0, parent_octant_idx: 5, scale: 18, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 20494 },
                StackFrame { t_min: 0.0, ptr: 20494, idx: 4, parent_octant_idx: 4, scale: 17, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 20662 },
                StackFrame { t_min: 0.0, ptr: 20662, idx: 7, parent_octant_idx: 3, scale: 16, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 20736 },
                StackFrame { t_min: 0.0, ptr: 20736, idx: 0, parent_octant_idx: 2, scale: 15, is_child: AlignedBool(1), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 20739 },
                StackFrame { t_min: 0.0, ptr: 20739, idx: 6, parent_octant_idx: 1, scale: 14, is_child: AlignedBool(0), is_leaf: AlignedBool(0), crossed_boundary: AlignedBool(0), next_ptr: 4294967295 },
                StackFrame { t_min: 0.9593506, ptr: 20739, idx: 4, parent_octant_idx: 1, scale: 14, is_child: AlignedBool(1), is_leaf: AlignedBool(1), crossed_boundary: AlignedBool(0), next_ptr: 20744 },
            ]);
            assert_eq!(buffer_out.result, OctreeResult {
                t: assert_float_eq!(buffer_out.result.t, 0.9593506),
                value: 1,
                face_id: 3,
                pos: assert_vec3_eq!(buffer_out.result.pos, AlignedPoint3::new(484.9203, 484.99994, 493.84668)),
                uv: assert_vec2_eq!(buffer_out.result.uv,AlignedPoint2::new(0.9202881, 0.8466797)),
                color: AlignedVec4::new(1.0, 0.0, 0.0, 1.0),
                inside_voxel: AlignedBool::from(false),
            });
        }
    }

    mod esvo_benchmarks {
        use test::Bencher;

        use cgmath::{Point3, Vector3};

        use crate::graphics::svo::SvoType;
        use crate::graphics::svo_shader_tests::tests::{cast_ray, setup_test};

        #[bench]
        fn hitting_nothing(b: &mut Bencher) {
            let setup = setup_test(SvoType::Esvo, None, |chunk| {
                for x in 0..32 {
                    chunk.set_block(x, 0, 0, 1);
                }
                for y in 0..32 {
                    chunk.set_block(0, y, 0, 1);
                }
                for z in 0..32 {
                    chunk.set_block(0, 0, z, 1);
                }
            });

            b.iter(|| {
                cast_ray(&setup.shader, Point3::new(0.0, 1.5, 1.5), Vector3::new(1.0, 0.0, 0.0), 32.0, false)
            });
        }

        #[bench]
        fn hitting_opaque_voxel(b: &mut Bencher) {
            let setup = setup_test(SvoType::Esvo, None, |chunk| chunk.set_block(31, 0, 0, 1));

            b.iter(|| {
                cast_ray(&setup.shader, Point3::new(0.0, 0.5, 0.5), Vector3::new(1.0, 0.0, 0.0), 32.0, false)
            });
        }

        #[bench]
        fn hitting_transparent_voxels(b: &mut Bencher) {
            let setup = setup_test(SvoType::Esvo, None, |chunk| {
                for x in 0..6 {
                    chunk.set_block(x, 0, 0, 3);
                }
                chunk.set_block(6, 0, 0, 1);
            });

            b.iter(|| {
                cast_ray(&setup.shader, Point3::new(-0.1, 0.25, 0.75), Vector3::new(1.0, 0.0, 0.0), 32.0, true)
            });
        }
    }

    mod csvo_benchmarks {
        use test::Bencher;

        use cgmath::{Point3, Vector3};

        use crate::graphics::svo::SvoType;
        use crate::graphics::svo_shader_tests::tests::{cast_ray, setup_test};

        #[bench]
        fn hitting_nothing(b: &mut Bencher) {
            let setup = setup_test(SvoType::Csvo, None, |chunk| {
                for x in 0..32 {
                    chunk.set_block(x, 0, 0, 1);
                }
                for y in 0..32 {
                    chunk.set_block(0, y, 0, 1);
                }
                for z in 0..32 {
                    chunk.set_block(0, 0, z, 1);
                }
            });

            b.iter(|| {
                cast_ray(&setup.shader, Point3::new(0.0, 1.5, 1.5), Vector3::new(1.0, 0.0, 0.0), 32.0, false)
            });
        }

        #[bench]
        fn hitting_opaque_voxel(b: &mut Bencher) {
            let setup = setup_test(SvoType::Csvo, None, |chunk| chunk.set_block(31, 0, 0, 1));

            b.iter(|| {
                cast_ray(&setup.shader, Point3::new(0.0, 0.5, 0.5), Vector3::new(1.0, 0.0, 0.0), 32.0, false)
            });
        }

        #[bench]
        fn hitting_transparent_voxels(b: &mut Bencher) {
            let setup = setup_test(SvoType::Csvo, None, |chunk| {
                for x in 0..6 {
                    chunk.set_block(x, 0, 0, 3);
                }
                chunk.set_block(6, 0, 0, 1);
            });

            b.iter(|| {
                cast_ray(&setup.shader, Point3::new(-0.1, 0.25, 0.75), Vector3::new(1.0, 0.0, 0.0), 32.0, true)
            });
        }
    }

    mod csvo_bit_magic_tests {
        use std::ops::Sub;
        use std::slice;

        fn read_u8(buffer: &[u32], ptr: usize) -> u8 {
            let index = ptr / 4;
            let modulo = ptr % 4;

            let v0 = buffer[index] as usize >> (modulo * 8) & 0xff;
            v0 as u8
        }

        fn read_u16(buffer: &[u32], ptr: usize) -> u16 {
            (read_u32(buffer, ptr) & 0xffff) as u16
        }

        fn read_u32(buffer: &[u32], ptr: usize) -> u32 {
            let index = ptr / 4;
            let modulo = ptr % 4;

            let l_shift = (4 - modulo) as u32 * 8;
            let mask = unsafe { 1_usize.unchecked_shl(l_shift) - 1 };
            let v0 = buffer[index] as usize >> (modulo * 8) & mask;
            let v1 = (buffer[index + 1] as usize) << l_shift & !mask;

            (v0 | v1) as u32
        }

        fn read_next_ptr(buffer: &[u32], ptr: usize, depth: u8, idx: u8) -> (Option<u32>, bool) {
            if depth > 2 {
                // internal nodes
                let header_mask = read_u16(buffer, ptr);
                let child_mask = header_mask >> (idx * 2) & 3;

                if child_mask == 0 {
                    return (None, false);
                }

                let offset_mask = (1 << (idx * 2)) - 1;
                let preceding_mask = header_mask & offset_mask;

                let mut offset = 0;
                offset += (1 << ((preceding_mask >> (0 * 2)) & 3)) >> 1;
                offset += (1 << ((preceding_mask >> (1 * 2)) & 3)) >> 1;
                offset += (1 << ((preceding_mask >> (2 * 2)) & 3)) >> 1;
                offset += (1 << ((preceding_mask >> (3 * 2)) & 3)) >> 1;
                offset += (1 << ((preceding_mask >> (4 * 2)) & 3)) >> 1;
                offset += (1 << ((preceding_mask >> (5 * 2)) & 3)) >> 1;
                offset += (1 << ((preceding_mask >> (6 * 2)) & 3)) >> 1;
                offset += (1 << ((preceding_mask >> (7 * 2)) & 3)) >> 1;

                let mut ptr_bytes = 0;
                ptr_bytes += (1 << ((header_mask >> (0 * 2)) & 3)) >> 1;
                ptr_bytes += (1 << ((header_mask >> (1 * 2)) & 3)) >> 1;
                ptr_bytes += (1 << ((header_mask >> (2 * 2)) & 3)) >> 1;
                ptr_bytes += (1 << ((header_mask >> (3 * 2)) & 3)) >> 1;
                ptr_bytes += (1 << ((header_mask >> (4 * 2)) & 3)) >> 1;
                ptr_bytes += (1 << ((header_mask >> (5 * 2)) & 3)) >> 1;
                ptr_bytes += (1 << ((header_mask >> (6 * 2)) & 3)) >> 1;
                ptr_bytes += (1 << ((header_mask >> (7 * 2)) & 3)) >> 1;

                let mut ptr_offset = read_u32(buffer, ptr + 2 + offset);
                ptr_offset &= unsafe { 1usize.unchecked_shl((1 << (child_mask - 1)) * 8) - 1 } as u32;

                if (ptr_offset & (1 << 31)) != 0 {
                    // absolute pointer
                    return (Some(ptr_offset ^ (1 << 31)), true);
                }

                return (Some((ptr + 2 + ptr_bytes) as u32 + ptr_offset), false);
            }

            let header_mask = read_u8(buffer, ptr);
            let child_mask = header_mask >> idx & 1;

            if child_mask == 0 {
                return (None, false);
            }

            let offset_mask = (1 << idx as u16) - 1;
            let offset = u8::count_ones(header_mask & offset_mask);

            if depth == 2 {
                // pre-leaf nodes
                let ptr_bytes = u8::count_ones(header_mask);
                let ptr_offset = read_u8(buffer, ptr + 1 + offset as usize) as u32;
                return (Some(ptr as u32 + 1 + ptr_bytes + ptr_offset), false);
            }

            // leaf nodes
            return (Some(ptr as u32 + 1 + offset), false);
        }

        fn read_leaf(buffer: &[u32], material_section_ptr: usize, pre_leaf_ptr: usize, ptr: usize, idx: u8) -> u32 {
            let material_section_offset = read_u16(buffer, pre_leaf_ptr + 1) as usize;

            let leaf_index = (ptr - (pre_leaf_ptr + 3)) as isize;
            let bit_mark = leaf_index * 8 + idx as isize;

            let mask = unsafe { 1_usize.unchecked_shl(bit_mark.min(32) as u32) - 1 } as u32;
            let v0 = read_u32(buffer, pre_leaf_ptr + 3) & mask;
            let mask = unsafe { 1_usize.unchecked_shl(bit_mark.sub(32).max(0) as u32) - 1 } as u32;
            let v1 = read_u32(buffer, pre_leaf_ptr + 3 + 4) & mask;

            let preceding_leaves = (u32::count_ones(v0) + u32::count_ones(v1)) as usize;

            read_u32(buffer, material_section_ptr + material_section_offset + preceding_leaves * 4)
        }

        #[test]
        fn test_read_scalars() {
            let data = vec![
                0b00001000_00000100_00000010_00000001,
                0b00001000_00000100_00000010_00000001,
                0,
            ];

            assert_eq!(read_u8(&data, 0), 1);
            assert_eq!(read_u8(&data, 1), 2);
            assert_eq!(read_u8(&data, 2), 4);
            assert_eq!(read_u8(&data, 3), 8);
            assert_eq!(read_u8(&data, 4), 1);

            assert_eq!(read_u16(&data, 0), 1 | (2 << 8));
            assert_eq!(read_u16(&data, 1), 2 | (4 << 8));
            assert_eq!(read_u16(&data, 2), 4 | (8 << 8));
            assert_eq!(read_u16(&data, 3), 8 | (1 << 8));
            assert_eq!(read_u16(&data, 4), 1 | (2 << 8));

            assert_eq!(read_u32(&data, 0), data[0]);
            assert_eq!(read_u32(&data, 1), 0b00000001_00001000_00000100_00000010);
            assert_eq!(read_u32(&data, 2), 0b00000010_00000001_00001000_00000100);
            assert_eq!(read_u32(&data, 3), 0b00000100_00000010_00000001_00001000);
            assert_eq!(read_u32(&data, 4), data[1]);

            let data = vec![1, 2, 3];
            assert_eq!(read_u32(&data, 0), 1);
            assert_eq!(read_u32(&data, 4), 2);
        }

        #[test]
        fn test_read_next_ptr() {
            // leaf nodes
            let data = vec![
                0b00000100_00000100_00000100_10001010,
            ];
            assert_eq!(read_next_ptr(&data, 0, 1, 0), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 1, 1), (Some(1), false));
            assert_eq!(read_next_ptr(&data, 0, 1, 2), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 1, 3), (Some(2), false));
            assert_eq!(read_next_ptr(&data, 0, 1, 4), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 1, 5), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 1, 6), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 1, 7), (Some(3), false));

            // pre-leaf nodes
            let data = vec![
                0b00001100_00000100_00000000_10001010,
            ];
            assert_eq!(read_next_ptr(&data, 0, 2, 0), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 2, 1), (Some(4), false));
            assert_eq!(read_next_ptr(&data, 0, 2, 2), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 2, 3), (Some(8), false));
            assert_eq!(read_next_ptr(&data, 0, 2, 4), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 2, 5), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 2, 6), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 2, 7), (Some(16), false));

            // internal nodes with relative pointers
            let data = vec![
                // 2nd/1  _ 1st    _ mask/2 _ mask/1
                0b00000001_00000000_11000000_10000100,
                // 3rd/3  _ 3rd/2  _ 3rd/1  _ 2nd/2
                0b00000100_00000010_00000001_00000010,
                //        _        _        _ 3rd/4
                0b00000000_00000000_00000000_00001000,
            ];
            assert_eq!(read_next_ptr(&data, 0, 3, 0), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 3, 1), (Some(9), false));
            assert_eq!(read_next_ptr(&data, 0, 3, 2), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 3, 3), (Some(9 + 0b00000010_00000001), false));
            assert_eq!(read_next_ptr(&data, 0, 3, 4), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 3, 5), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 3, 6), (None, false));
            assert_eq!(read_next_ptr(&data, 0, 3, 7), (Some(9 + 0b00001000_00000100_00000010_00000001), false));

            // internal nodes with absolute pointers
            let data = vec![
                // 1st/2  _ 1st/1  _ mask/2 _ mask/1
                0b00000010_00000001_00000000_00000011,
                //        _        _ 1st/4  _ 1st/3
                0b00000000_00000000_10001000_00000100,
            ];
            assert_eq!(read_next_ptr(&data, 0, 3, 0), (Some(0b00001000_00000100_00000010_00000001), true));
        }

        #[test]
        fn test_read_leaf() {
            fn to_u32_vec(data: Vec<u8>) -> Vec<u32> {
                let len = data.len().checked_div(4).unwrap();
                let ptr: *const u32 = data.as_slice().as_ptr().cast();
                unsafe { slice::from_raw_parts(ptr, len) }.to_vec()
            }

            let data = to_u32_vec(vec![
                // material section start
                0, 0, 0, 0, // arbitrary filler
                // material subsection
                11, 0, 0, 0,
                22, 0, 0, 0,
                33, 0, 0, 0,
                44, 0, 0, 0,
                55, 0, 0, 0,
                // pre-leaf node
                0b10101110, // header mask
                4, 0, // material section offset
                0b00000010, // leaf 1
                0b00000100, // leaf 2
                0b00001000, // leaf 3
                0b00010000, // leaf 4
                0b00100000, // leaf 5
                0, 0, 0, // padding to full u32
                0, 0, 0, 0, // additional u32 for padding
            ]);

            assert_eq!(read_leaf(&data, 0, 24, 27, 1), 11);
            assert_eq!(read_leaf(&data, 0, 24, 28, 2), 22);
            assert_eq!(read_leaf(&data, 0, 24, 29, 3), 33);
            assert_eq!(read_leaf(&data, 0, 24, 30, 4), 44);
            assert_eq!(read_leaf(&data, 0, 24, 31, 5), 55);
        }
    }
}
