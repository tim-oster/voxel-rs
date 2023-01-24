#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cgmath::{InnerSpace, Point2, Point3, Vector3, Vector4};

    use crate::{AlignedPoint3, AlignedVec3, assert_float_eq, Chunk, ChunkPos, graphics, Position, SerializedChunk, Svo};
    use crate::chunk::ChunkStorage;
    use crate::core::{Config, GlContext};
    use crate::graphics::{buffer, ShaderProgram, TextureArray, TextureArrayError};
    use crate::graphics::buffer::{Buffer, MappedBuffer};
    use crate::graphics::resource::Resource;
    use crate::graphics::shader::ShaderError;
    use crate::graphics::svo::Material;
    use crate::graphics::util::{AlignedBool, AlignedPoint2, AlignedVec4};
    use crate::world::allocator::Allocator;

    #[repr(C)]
    struct BufferIn {
        max_dst: f32,
        pos: AlignedPoint3<f32>,
        dir: AlignedVec3<f32>,
        cast_translucent: AlignedBool,
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
        }
        world_buffer
    }

    fn create_test_materials() -> (Buffer<Material>, Resource<TextureArray, TextureArrayError>) {
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
            Material { // empty
                specular_pow: 0.0,
                specular_strength: 0.0,
                tex_top: -1,
                tex_side: -1,
                tex_bottom: -1,
                tex_top_normal: -1,
                tex_side_normal: -1,
                tex_bottom_normal: -1,
            },
            Material { // full
                specular_pow: 0.0,
                specular_strength: 0.0,
                tex_top: tex_array.lookup("full").unwrap() as i32,
                tex_side: tex_array.lookup("full").unwrap() as i32,
                tex_bottom: tex_array.lookup("full").unwrap() as i32,
                tex_top_normal: -1,
                tex_side_normal: -1,
                tex_bottom_normal: -1,
            },
            Material { // coords
                specular_pow: 0.0,
                specular_strength: 0.0,
                tex_top: tex_array.lookup("coords").unwrap() as i32,
                tex_side: tex_array.lookup("coords").unwrap() as i32,
                tex_bottom: tex_array.lookup("coords").unwrap() as i32,
                tex_top_normal: -1,
                tex_side_normal: -1,
                tex_bottom_normal: -1,
            },
            Material { // transparent_1
                specular_pow: 0.0,
                specular_strength: 0.0,
                tex_top: tex_array.lookup("transparent_1").unwrap() as i32,
                tex_side: tex_array.lookup("transparent_1").unwrap() as i32,
                tex_bottom: tex_array.lookup("transparent_1").unwrap() as i32,
                tex_top_normal: -1,
                tex_side_normal: -1,
                tex_bottom_normal: -1,
            },
            Material { // transparent_2
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
        context: GlContext,
        world_buffer: MappedBuffer<u32>,
        shader: Resource<ShaderProgram, ShaderError>,
        material_buffer: Buffer<Material>,
        tex_array: Resource<TextureArray, TextureArrayError>,
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
        world_buffer.bind_as_storage_buffer(0);

        let shader = Resource::new(
            || graphics::ShaderProgramBuilder::new().load_shader_bundle("assets/shaders/svo.test.glsl")?.build()
        ).unwrap();

        let (material_buffer, tex_array) = create_test_materials();
        material_buffer.bind_as_storage_buffer(2);
        shader.bind();
        shader.set_texture("u_texture", 0, &tex_array);
        shader.unbind();

        TestSetup { context, world_buffer, shader, material_buffer, tex_array }
    }

    fn cast_ray(shader: &Resource<ShaderProgram, ShaderError>, pos: Point3<f32>, dir: Vector3<f32>, max_dst: f32, cast_translucent: bool) -> BufferOut {
        let new_buffer_in = Buffer::new(vec![BufferIn {
            max_dst,
            pos: AlignedPoint3(pos),
            dir: AlignedVec3(dir.normalize()),
            cast_translucent: AlignedBool(cast_translucent),
        }], buffer::STATIC_READ);
        new_buffer_in.bind_as_storage_buffer(11);

        let mut buffer_out = Buffer::new(vec![BufferOut {
            result: OctreeResult {
                t: 0.0,
                value: 0,
                face_id: 0,
                pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)),
                uv: AlignedPoint2(Point2::new(0.0, 0.0)),
                color: AlignedVec4(Vector4::new(0.0, 0.0, 0.0, 0.0)),
                inside_block: AlignedBool(false),
            },
            stack_ptr: 0,
            stack: [StackFrame {
                t_min: 0.0,
                ptr: 0,
                idx: 0,
                parent_octant_idx: 0,
                scale: 0,
                is_child: AlignedBool(false),
                is_leaf: AlignedBool(false),
            }; 100],
        }], buffer::STATIC_DRAW | buffer::STATIC_READ);
        buffer_out.bind_as_storage_buffer(12);

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
            StackFrame { t_min: 0.0, ptr: 0, idx: 1, parent_octant_idx: 0, scale: 22, is_child: AlignedBool(true), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 0.0, ptr: 113, idx: 1, parent_octant_idx: 0, scale: 21, is_child: AlignedBool(true), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 0.0, ptr: 5, idx: 1, parent_octant_idx: 0, scale: 20, is_child: AlignedBool(true), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 0.0, ptr: 17, idx: 1, parent_octant_idx: 0, scale: 19, is_child: AlignedBool(true), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 0.0, ptr: 29, idx: 1, parent_octant_idx: 0, scale: 18, is_child: AlignedBool(true), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 0.0, ptr: 41, idx: 1, parent_octant_idx: 0, scale: 17, is_child: AlignedBool(false), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 1.0, ptr: 41, idx: 0, parent_octant_idx: 0, scale: 17, is_child: AlignedBool(false), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 2.0, ptr: 29, idx: 0, parent_octant_idx: 0, scale: 18, is_child: AlignedBool(false), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 4.0, ptr: 17, idx: 0, parent_octant_idx: 0, scale: 19, is_child: AlignedBool(false), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 8.0, ptr: 5, idx: 0, parent_octant_idx: 0, scale: 20, is_child: AlignedBool(false), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 16.0, ptr: 113, idx: 0, parent_octant_idx: 0, scale: 21, is_child: AlignedBool(true), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 16.0, ptr: 5, idx: 1, parent_octant_idx: 1, scale: 20, is_child: AlignedBool(false), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 24.0, ptr: 5, idx: 0, parent_octant_idx: 1, scale: 20, is_child: AlignedBool(true), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 24.0, ptr: 65, idx: 1, parent_octant_idx: 1, scale: 19, is_child: AlignedBool(false), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 28.0, ptr: 65, idx: 0, parent_octant_idx: 1, scale: 19, is_child: AlignedBool(true), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 28.0, ptr: 77, idx: 1, parent_octant_idx: 1, scale: 18, is_child: AlignedBool(false), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 30.0, ptr: 77, idx: 0, parent_octant_idx: 1, scale: 18, is_child: AlignedBool(true), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 30.0, ptr: 89, idx: 1, parent_octant_idx: 1, scale: 17, is_child: AlignedBool(false), is_leaf: AlignedBool(false) },
            StackFrame { t_min: 31.0, ptr: 89, idx: 0, parent_octant_idx: 1, scale: 17, is_child: AlignedBool(true), is_leaf: AlignedBool(true) },
        ]);
        assert_eq!(buffer_out.result, OctreeResult {
            t: 31.0,
            value: 1,
            face_id: 0,
            pos: AlignedPoint3(Point3::new(31.000008, 0.5, 0.5)),
            uv: AlignedPoint2(Point2::new(0.5, 0.5)),
            color: AlignedVec4(Vector4::new(1.0, 0.0, 0.0, 1.0)),
            inside_block: AlignedBool(false),
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
                    inside_block: AlignedBool(false),
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
                    inside_block: AlignedBool(false),
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
                    inside_block: AlignedBool(false),
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
                    inside_block: AlignedBool(false),
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
                    inside_block: AlignedBool(false),
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
                    inside_block: AlignedBool(false),
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
                    uv: AlignedPoint2(Point2::new(0.099998474, 0.099998474)),
                    color: AlignedVec4(Vector4::new(1.0, 0.0, 0.0, 1.0)),
                    inside_block: AlignedBool(false),
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
                    uv: AlignedPoint2(Point2::new(0.099998474, 0.9000015)),
                    color: AlignedVec4(Vector4::new(1.0, 0.0, 0.0, 1.0)),
                    inside_block: AlignedBool(false),
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
            pos: Point2<f32>,
            dir: Vector3<f32>,
            expected_uv: Point2<f32>,
            expected_color: Vector4<f32>,
        }
        let cases = vec![
            // pos z
            TestCase {
                pos: Point2::new(0.1, 0.1),
                dir: Vector3::new(0.0, 0.0, 1.0),
                expected_uv: Point2::new(0.1, 0.1),
                expected_color: Vector4::new(0.0, 0.0, 0.0, 1.0),
            },
            TestCase {
                pos: Point2::new(0.1, 0.5),
                dir: Vector3::new(0.0, 0.0, 1.0),
                expected_uv: Point2::new(0.1, 0.5),
                expected_color: Vector4::new(0.0, 0.4, 0.0, 1.0),
            },
            TestCase {
                pos: Point2::new(0.5, 0.1),
                dir: Vector3::new(0.0, 0.0, 1.0),
                expected_uv: Point2::new(0.5, 0.1),
                expected_color: Vector4::new(0.4, 0.0, 0.0, 1.0),
            },
            TestCase {
                pos: Point2::new(0.5, 0.5),
                dir: Vector3::new(0.0, 0.0, 1.0),
                expected_uv: Point2::new(0.5, 0.5),
                expected_color: Vector4::new(0.4, 0.4, 0.0, 1.0),
            },
            // neg z
            // TODO complete for other sides and fix this one
            TestCase {
                pos: Point2::new(0.1, 0.1),
                dir: Vector3::new(0.0, 0.0, -1.0),
                expected_uv: Point2::new(0.1, 0.1),
                expected_color: Vector4::new(0.0, 0.0, 0.0, 1.0),
            },
            TestCase {
                pos: Point2::new(0.1, 0.5),
                dir: Vector3::new(0.0, 0.0, -1.0),
                expected_uv: Point2::new(0.1, 0.1),
                expected_color: Vector4::new(0.0, 0.0, 0.0, 1.0),
            },
            TestCase {
                pos: Point2::new(0.5, 0.5),
                dir: Vector3::new(0.0, 0.0, -1.0),
                expected_uv: Point2::new(0.5, 0.5),
                expected_color: Vector4::new(0.4, 0.4, 0.0, 1.0),
            },
        ];
        for case in cases {
            let buffer_out = cast_ray(
                &setup.shader,
                Point3::new(case.pos.x, case.pos.y, -0.1),
                case.dir,
                32.0,
                false,
            );
            assert_eq!(buffer_out.result, OctreeResult {
                t: assert_float_eq!(buffer_out.result.t, 0.1),
                value: 2,
                face_id: 4,
                pos: AlignedPoint3(Point3::new(
                    assert_float_eq!(buffer_out.result.pos.x, case.pos.x),
                    assert_float_eq!(buffer_out.result.pos.y, case.pos.y),
                    assert_float_eq!(buffer_out.result.pos.z, 0.0),
                )),
                uv: AlignedPoint2(Point2::new(
                    assert_float_eq!(buffer_out.result.uv.x, case.expected_uv.x),
                    assert_float_eq!(buffer_out.result.uv.y, case.expected_uv.y),
                )),
                color: AlignedVec4(case.expected_color),
                inside_block: AlignedBool(false),
            }, "({:?}) for ({}, {})", case.dir, case.pos.x, case.pos.y);
        }
    }

    // TODO finish after testing uv mapping
    // #[test]
    // fn casting_against_translucent_leafs() {
    //     // TODO
    //     // - test that alpha=0 is properly traced in all cases (inside same octant but also across different octants)
    //     //      - are adjacent blocks skipped?
    //     //      - is cast_translucent flag properly taken into account?
    //
    //     // TODO comment
    //     let setup = setup_test(|chunk| {
    //         chunk.set_block(0, 0, 0, 3);
    //         chunk.set_block(0, 0, 1, 3);
    //         chunk.set_block(5, 0, 0, 3);
    //         chunk.set_block(5, 0, 1, 4);
    //     });
    //
    //     let start = Point3::new(0.25, 0.5, -0.1);
    //     let end = Point3::new(0.75, 0.5, 1.0);
    //
    //     // do not cast translucent
    //     let buffer_out = cast_ray(
    //         &setup.shader,
    //         Point3::new(0.25, 0.5, -0.1),
    //         (end-start).normalize(),
    //         32.0,
    //         false,
    //     );
    //     assert_eq!(buffer_out.result, OctreeResult {
    //         t: -1.0,
    //         value: 0,
    //         face_id: 0,
    //         pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)),
    //         uv: AlignedPoint2(Point2::new(0.0, 0.0)),
    //         color: AlignedVec4(Vector4::new(0.0, 0.0, 0.0, 0.0)),
    //         inside_block: AlignedBool(true),
    //     }, "do not cast translucent");
    // }

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
            inside_block: AlignedBool(true),
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
            inside_block: AlignedBool(false),
        }, "outside block");
    }
}
