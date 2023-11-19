#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cgmath::{InnerSpace, Point2, Point3, Vector3, Vector4};

    use crate::{assert_float_eq, ChunkPos, graphics, Position};
    use crate::chunk::ChunkStorage;
    use crate::core::{Config, GlContext};
    use crate::graphics::{buffer, ShaderProgram, TextureArray, TextureArrayError};
    use crate::graphics::buffer::{Buffer, MappedBuffer};
    use crate::graphics::consts::shader_buffer_indices;
    use crate::graphics::resource::Resource;
    use crate::graphics::shader::ShaderError;
    use crate::graphics::macros::{AlignedBool, AlignedPoint2, AlignedPoint3, AlignedVec3, AlignedVec4};
    use crate::graphics::svo_registry::MaterialInstance;
    use crate::world::allocator::Allocator;
    use crate::world::chunk::Chunk;
    use crate::world::svo::{SerializedChunk, Svo};

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
        inside_block: AlignedBool,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq)]
    struct StackFrame {
        t_min: f32,
        ptr: i32,
        idx: i32,
        parent_octant_idx: i32,
        scale: i32,
        is_child: AlignedBool,
        is_leaf: AlignedBool,
    }

    fn create_test_world<F>(builder: F) -> MappedBuffer<u32>
        where F: FnOnce(&mut Chunk) {
        // TODO refactor world & svo setup logic once rest of codebase has been refactored
        let allocator = Allocator::new(
            Box::new(|| ChunkStorage::with_size(32f32.log2() as u32)),
            Some(Box::new(|storage| storage.reset())),
        );
        let allocator = Arc::new(allocator);

        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0), allocator);
        builder(&mut chunk);

        let chunk = SerializedChunk::new(chunk.pos, chunk.get_storage().unwrap(), 5);
        let mut svo = Svo::<SerializedChunk>::new();
        svo.set(Position(0, 0, 0), Some(chunk));
        svo.serialize();

        let world_buffer = MappedBuffer::<u32>::new(1000 * 1024 * 1024 / 4);
        unsafe {
            let max_depth_exp = (-(svo.depth() as f32)).exp2();
            world_buffer.write(max_depth_exp.to_bits());
            svo.write_changes_to(world_buffer.offset(1));
            svo.reset_changes();
        }
        world_buffer
    }

    fn create_test_materials() -> (Buffer<MaterialInstance>, Resource<TextureArray, TextureArrayError>) {
        let tex_array = Resource::new(
            || graphics::TextureArrayBuilder::new(1)
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
        _world_buffer: MappedBuffer<u32>,
        _material_buffer: Buffer<MaterialInstance>,
        _tex_array: Resource<TextureArray, TextureArrayError>,
        shader: Resource<ShaderProgram, ShaderError>,
    }

    fn setup_test<F>(world_builder: F) -> TestSetup
        where F: FnOnce(&mut Chunk) {
        let context = GlContext::new(Config {
            width: 640,
            height: 490,
            title: "",
            msaa_samples: 0,
            headless: true,
        });

        let world_buffer = create_test_world(world_builder);
        world_buffer.bind_as_storage_buffer(shader_buffer_indices::WORLD);

        let shader = Resource::new(
            || graphics::ShaderProgramBuilder::new().load_shader_bundle("assets/shaders/svo.test.glsl")?.build()
        ).unwrap();

        let (material_buffer, tex_array) = create_test_materials();
        material_buffer.bind_as_storage_buffer(shader_buffer_indices::MATERIALS);
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
        buffer_in.bind_as_storage_buffer(shader_buffer_indices::DEBUG_IN);

        let mut buffer_out = Buffer::new(vec![BufferOut {
            result: OctreeResult {
                t: 0.0,
                value: 0,
                face_id: 0,
                pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)),
                uv: AlignedPoint2(Point2::new(0.0, 0.0)),
                color: AlignedVec4(Vector4::new(0.0, 0.0, 0.0, 0.0)),
                inside_block: AlignedBool::from(false),
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
            }; 100],
        }], buffer::STATIC_DRAW | buffer::STATIC_READ);
        buffer_out.bind_as_storage_buffer(shader_buffer_indices::DEBUG_OUT);

        unsafe {
            shader.bind();
            gl::DispatchCompute(1, 1, 1);
            gl::MemoryBarrier(gl::ALL_BARRIER_BITS);
            shader.unbind();
        }

        buffer_out.pull_data();
        buffer_out.take().remove(0)
    }

    /// Tests if the shader properly traverses through the octree by checking each step the shader
    /// takes. The goal is to verify that the algorithm is capable of performing the basic
    /// PUSH, ADVANCE & POP mechanisms necessary to step through the SVO memory.
    #[test]
    fn shader_svo_traversal() {
        // setup a world with one block at the "end" of the chunk to make sure that the algorithm
        // has to step through many empty children
        let setup = setup_test(|chunk| chunk.set_block(31, 0, 0, 1));
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

        assert_eq!(buffer_out.stack_ptr, 18);
        assert_eq!(&buffer_out.stack[..19], &[
            StackFrame { t_min: 0.0, ptr: 0, idx: 1, parent_octant_idx: 0, scale: 22, is_child: AlignedBool::from(true), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 0.0, ptr: 113, idx: 1, parent_octant_idx: 0, scale: 21, is_child: AlignedBool::from(true), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 0.0, ptr: 5, idx: 1, parent_octant_idx: 0, scale: 20, is_child: AlignedBool::from(true), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 0.0, ptr: 17, idx: 1, parent_octant_idx: 0, scale: 19, is_child: AlignedBool::from(true), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 0.0, ptr: 29, idx: 1, parent_octant_idx: 0, scale: 18, is_child: AlignedBool::from(true), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 0.0, ptr: 41, idx: 1, parent_octant_idx: 0, scale: 17, is_child: AlignedBool::from(false), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 1.0, ptr: 41, idx: 0, parent_octant_idx: 0, scale: 17, is_child: AlignedBool::from(false), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 2.0, ptr: 29, idx: 0, parent_octant_idx: 0, scale: 18, is_child: AlignedBool::from(false), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 4.0, ptr: 17, idx: 0, parent_octant_idx: 0, scale: 19, is_child: AlignedBool::from(false), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 8.0, ptr: 5, idx: 0, parent_octant_idx: 0, scale: 20, is_child: AlignedBool::from(false), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 16.0, ptr: 113, idx: 0, parent_octant_idx: 0, scale: 21, is_child: AlignedBool::from(true), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 16.0, ptr: 5, idx: 1, parent_octant_idx: 1, scale: 20, is_child: AlignedBool::from(false), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 24.0, ptr: 5, idx: 0, parent_octant_idx: 1, scale: 20, is_child: AlignedBool::from(true), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 24.0, ptr: 65, idx: 1, parent_octant_idx: 1, scale: 19, is_child: AlignedBool::from(false), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 28.0, ptr: 65, idx: 0, parent_octant_idx: 1, scale: 19, is_child: AlignedBool::from(true), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 28.0, ptr: 77, idx: 1, parent_octant_idx: 1, scale: 18, is_child: AlignedBool::from(false), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 30.0, ptr: 77, idx: 0, parent_octant_idx: 1, scale: 18, is_child: AlignedBool::from(true), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 30.0, ptr: 89, idx: 1, parent_octant_idx: 1, scale: 17, is_child: AlignedBool::from(false), is_leaf: AlignedBool::from(false) },
            StackFrame { t_min: 31.0, ptr: 89, idx: 0, parent_octant_idx: 1, scale: 17, is_child: AlignedBool::from(true), is_leaf: AlignedBool::from(true) },
        ]);
        assert_eq!(buffer_out.result, OctreeResult {
            t: 31.0,
            value: 1,
            face_id: 0,
            pos: AlignedPoint3(Point3::new(31.000008, 0.5, 0.5)),
            uv: AlignedPoint2(Point2::new(0.5, 0.5)),
            color: AlignedVec4(Vector4::new(1.0, 0.0, 0.0, 1.0)),
            inside_block: AlignedBool::from(false),
        });
    }

    /// Tests if casting along every axis yields the expected result. Per axis, it checks for both
    /// positive and negative direction as well as being inside and outside the root octant. The
    /// final cases combine all axis into a diagonal raycast.
    #[test]
    fn cast_inside_outside_all_axes() {
        // place blocks at 30 instead of 31 to have space for inside casts for every axis
        let setup = setup_test(|chunk| {
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
                    pos: AlignedPoint3(Point3::new(30.000008, 0.5, 0.5)),
                    uv: AlignedPoint2(Point2::new(0.5, 0.5)),
                    color: AlignedVec4(Vector4::new(1.0, 0.0, 0.0, 1.0)),
                    inside_block: AlignedBool::from(false),
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
                    pos: AlignedPoint3(Point3::new(30.999992, 0.5, 0.5)),
                    uv: AlignedPoint2(Point2::new(0.5, 0.5)),
                    color: AlignedVec4(Vector4::new(1.0, 0.0, 0.0, 1.0)),
                    inside_block: AlignedBool::from(false),
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
                    pos: AlignedPoint3(Point3::new(0.5, 30.000008, 0.5)),
                    uv: AlignedPoint2(Point2::new(0.5, 0.5)),
                    color: AlignedVec4(Vector4::new(1.0, 0.0, 0.0, 1.0)),
                    inside_block: AlignedBool::from(false),
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
                    pos: AlignedPoint3(Point3::new(0.5, 30.999992, 0.5)),
                    uv: AlignedPoint2(Point2::new(0.5, 0.5)),
                    color: AlignedVec4(Vector4::new(1.0, 0.0, 0.0, 1.0)),
                    inside_block: AlignedBool::from(false),
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
                    pos: AlignedPoint3(Point3::new(0.5, 0.5, 30.000008)),
                    uv: AlignedPoint2(Point2::new(0.5, 0.5)),
                    color: AlignedVec4(Vector4::new(1.0, 0.0, 0.0, 1.0)),
                    inside_block: AlignedBool::from(false),
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
                    pos: AlignedPoint3(Point3::new(0.5, 0.5, 30.999992)),
                    uv: AlignedPoint2(Point2::new(0.5, 0.5)),
                    color: AlignedVec4(Vector4::new(1.0, 0.0, 0.0, 1.0)),
                    inside_block: AlignedBool::from(false),
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
                    pos: AlignedPoint3(Point3::new(30.099998, 30.000008, 30.099998)),
                    uv: AlignedPoint2(Point2::new(0.099998474, 0.9000015)),
                    color: AlignedVec4(Vector4::new(1.0, 0.0, 0.0, 1.0)),
                    inside_block: AlignedBool::from(false),
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
                    pos: AlignedPoint3(Point3::new(30.900002, 30.999992, 30.900002)),
                    uv: AlignedPoint2(Point2::new(0.9000015, 0.9000015)),
                    color: AlignedVec4(Vector4::new(1.0, 0.0, 0.0, 1.0)),
                    inside_block: AlignedBool::from(false),
                },
            },
        ];
        for case in cases {
            let buffer_out = cast_ray(&setup.shader, case.pos, case.dir, 100.0, false);
            assert_eq!(buffer_out.result, case.expected, "test case \"{}\" inside", case.name);

            let mut case = case;
            case.pos -= case.dir.normalize();
            case.expected.t += 1.0;

            let buffer_out = cast_ray(&setup.shader, case.pos, case.dir, 100.0, false);
            assert_eq!(buffer_out.result, case.expected, "test case \"{}\" outside", case.name);
        }
    }

    /// Tests if uv calculation and texture lookup works properly on every side of a voxel. This
    /// uses a special texture that maps a simple 2d coordinate system with a resolution of 4x4
    /// onto each side along the uv space.
    #[test]
    fn uv_coords_on_all_sides() {
        let setup = setup_test(|chunk| chunk.set_block(0, 0, 0, 2));

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
            assert_eq!(buffer_out.result.uv.0, Point2::new(
                assert_float_eq!(buffer_out.result.uv.x, case.expected_uv.x),
                assert_float_eq!(buffer_out.result.uv.y, case.expected_uv.y),
            ), "{}", case_name);
            assert_eq!(buffer_out.result.color.0, case.expected_color, "{}", case_name);
        }
    }

    /// Tests if translucency is properly accounted for during ray casting. Assert that identical,
    /// adjacent voxels are skipped and make sure that cast_translucent flag is respected.
    #[test]
    fn casting_against_translucent_leafs() {
        // This setup has to small rows of adjacent blocks. The first row consists of the same
        // blocks while the second one has two different kinds. Both kinds have textures that
        // are transparent on the left half and opaque on the right half. This allows testing
        // the algorithm's behaviour when casting diagonally through them.
        let setup = setup_test(|chunk| {
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
            pos: AlignedPoint3(Point3::new(
                assert_float_eq!(buffer_out.result.pos.x, 0.295, 0.01),
                assert_float_eq!(buffer_out.result.pos.y, 0.5),
                assert_float_eq!(buffer_out.result.pos.z, 0.0),
            )),
            uv: AlignedPoint2(Point2::new(
                assert_float_eq!(buffer_out.result.uv.x, 0.295, 0.01),
                assert_float_eq!(buffer_out.result.uv.y, 0.5),
            )),
            color: AlignedVec4(Vector4::new(0.0, 0.0, 0.0, 0.0)),
            inside_block: AlignedBool::from(false),
        }, "do not cast translucent");

        // cast translucent with adjacent identical
        let buffer_out = cast_ray(&setup.shader, Point3::new(0.25, 0.5, -0.1), dir, 32.0, true);
        assert_eq!(buffer_out.result, OctreeResult {
            t: -1.0,
            value: 0,
            face_id: 0,
            pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)),
            uv: AlignedPoint2(Point2::new(0.0, 0.0)),
            color: AlignedVec4(Vector4::new(0.0, 0.0, 0.0, 0.0)),
            inside_block: AlignedBool::from(false),
        }, "cast translucent with adjacent identical");

        // cast translucent with adjacent different
        let buffer_out = cast_ray(&setup.shader, Point3::new(5.25, 0.5, -0.1), dir, 32.0, true);
        assert_eq!(buffer_out.result, OctreeResult {
            t: assert_float_eq!(buffer_out.result.t, 1.2, 0.01),
            value: 4,
            face_id: 4,
            pos: AlignedPoint3(Point3::new(
                assert_float_eq!(buffer_out.result.pos.x, 5.75, 0.01),
                assert_float_eq!(buffer_out.result.pos.y, 0.5),
                assert_float_eq!(buffer_out.result.pos.z, 1.0),
            )),
            uv: AlignedPoint2(Point2::new(
                assert_float_eq!(buffer_out.result.uv.x, 0.75, 0.01),
                assert_float_eq!(buffer_out.result.uv.y, 0.5),
            )),
            color: AlignedVec4(Vector4::new(0.0, 1.0, 0.0, 1.0)),
            inside_block: AlignedBool::from(false),
        }, "cast translucent with adjacent different");
    }

    /// Tests if the algorithm correctly detects starting from within a voxel and that it does not
    /// intersect with that voxel.
    #[test]
    fn detect_inside_leaf_voxel() {
        let setup = setup_test(|chunk| chunk.set_block(0, 0, 0, 1));

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
            pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)),
            uv: AlignedPoint2(Point2::new(0.0, 0.0)),
            color: AlignedVec4(Vector4::new(0.0, 0.0, 0.0, 0.0)),
            inside_block: AlignedBool::from(true),
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
            pos: AlignedPoint3(Point3::new(
                assert_float_eq!(buffer_out.result.pos.x, 8e-6),
                0.5,
                0.5,
            )),
            uv: AlignedPoint2(Point2::new(0.5, 0.5)),
            color: AlignedVec4(Vector4::new(1.0, 0.0, 0.0, 1.0)),
            inside_block: AlignedBool::from(false),
        }, "outside block");
    }
}
