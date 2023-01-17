#[cfg(test)]
mod tests {
    use std::{mem, ptr};
    use std::ffi::c_void;
    use std::sync::Arc;

    use cgmath::{InnerSpace, Point2, Point3, Vector3, Vector4};
    use gl::types::{GLsizeiptr, GLvoid};
    use glfw::Context;

    use crate::{Chunk, ChunkPos, graphics, Position, SerializedChunk, Svo};
    use crate::chunk::ChunkStorage;
    use crate::world::allocator::Allocator;

    #[repr(align(16))]
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct AlignedVec3<T>(cgmath::Vector3<T>);

    #[repr(align(16))]
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct AlignedVec4<T>(cgmath::Vector4<T>);

    #[repr(align(8))]
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct AlignedPoint2<T>(cgmath::Point2<T>);

    #[repr(align(16))]
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct AlignedPoint3<T>(cgmath::Point3<T>);

    #[repr(align(4))]
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct AlignedBool(bool);

    #[test]
    fn test() {
        let mut context = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
        context.window_hint(glfw::WindowHint::ContextVersion(4, 6));
        context.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));
        context.window_hint(glfw::WindowHint::Visible(false));

        let (mut window, _) = context
            .create_window(640, 490, "", glfw::WindowMode::Windowed)
            .expect("failed to create window");
        window.make_current();

        gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

        let allocator = Allocator::new(
            Box::new(|| ChunkStorage::with_size(32f32.log2() as u32)),
            Some(Box::new(|storage| storage.reset())),
        );
        let allocator = Arc::new(allocator);
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0), allocator);
        chunk.set_block(31, 0, 0, 1);
        let chunk = SerializedChunk::new(chunk.pos, chunk.get_storage().unwrap(), 5);

        let mut svo = Svo::<SerializedChunk>::new();
        svo.set(Position(0, 0, 0), Some(chunk));
        svo.serialize();

        let mut world_ssbo = 0;
        let world_buffer;
        let world_buffer_size = 1000 * 1024 * 1024; // 1000 MB
        unsafe {
            gl::CreateBuffers(1, &mut world_ssbo);

            gl::NamedBufferStorage(
                world_ssbo,
                world_buffer_size,
                ptr::null(),
                gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT,
            );
            world_buffer = gl::MapNamedBufferRange(
                world_ssbo,
                0,
                world_buffer_size,
                gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT,
            ) as *mut u32;

            gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 0, world_ssbo);

            let max_depth_exp = (-(svo.depth() as f32)).exp2();
            world_buffer.write(max_depth_exp.to_bits());
            svo.write_changes_to(world_buffer.offset(1));
        }

        let tex_array = graphics::Resource::new(
            || graphics::TextureArrayBuilder::new(1)
                .add_rgba8("test", 2, 2, vec![
                    255, 0, 0, 255,
                    0, 255, 0, 255,
                    0, 0, 255, 255,
                    255, 255, 0, 255,
                ])?
                .build()
        ).unwrap();

        #[repr(C)]
        struct Material {
            specular_pow: f32,
            specular_strength: f32,
            tex_top: i32,
            tex_side: i32,
            tex_bottom: i32,
            tex_top_normal: i32,
            tex_side_normal: i32,
            tex_bottom_normal: i32,
        }

        let materials = vec![
            Material { // air
                specular_pow: 0.0,
                specular_strength: 0.0,
                tex_top: -1,
                tex_side: -1,
                tex_bottom: -1,
                tex_top_normal: -1,
                tex_side_normal: -1,
                tex_bottom_normal: -1,
            },
            Material { // grass
                specular_pow: 0.0,
                specular_strength: 0.0,
                tex_top: tex_array.lookup("test").unwrap() as i32,
                tex_side: tex_array.lookup("test").unwrap() as i32,
                tex_bottom: tex_array.lookup("test").unwrap() as i32,
                tex_top_normal: -1,
                tex_side_normal: -1,
                tex_bottom_normal: -1,
            },
        ];
        let mut material_ssbo = 0;
        unsafe {
            gl::CreateBuffers(1, &mut material_ssbo);

            gl::NamedBufferData(
                material_ssbo,
                (mem::size_of::<Material>() * materials.len()) as GLsizeiptr,
                &materials[0] as *const Material as *const c_void,
                gl::STATIC_READ,
            );

            gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 2, material_ssbo);
        }

        let shader = graphics::Resource::new(
            || graphics::ShaderProgramBuilder::new().load_shader_bundle("assets/shaders/svo.test.glsl")?.build()
        ).unwrap();

        #[repr(C)]
        struct BufferIn {
            max_dst: f32,
            pos: AlignedPoint3<f32>,
            dir: AlignedVec3<f32>,
            cast_translucent: AlignedBool,
        }
        let buffer_in = BufferIn {
            max_dst: 32.0,
            pos: AlignedPoint3(Point3::new(0.0, 0.01, 0.0)),
            dir: AlignedVec3(Vector3::new(1.0, 0.00001, 0.00001).normalize()),
            cast_translucent: AlignedBool(false),
        };
        let mut buffer_in_ssbo = 0;
        unsafe {
            gl::CreateBuffers(1, &mut buffer_in_ssbo);
            gl::NamedBufferData(
                buffer_in_ssbo,
                mem::size_of::<BufferIn>() as GLsizeiptr,
                &buffer_in as *const BufferIn as *const GLvoid,
                gl::STATIC_READ,
            );
            gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 11, buffer_in_ssbo);
        }

        #[repr(C)]
        #[derive(Copy, Clone, Debug)]
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
        #[repr(C)]
        struct BufferOut {
            result: OctreeResult,
            stack_ptr: i32,
            stack: [StackFrame; 100],
        }
        let mut buffer_out = BufferOut {
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
        };
        let mut buffer_out_ssbo = 0;
        unsafe {
            gl::CreateBuffers(1, &mut buffer_out_ssbo);
            gl::NamedBufferData(
                buffer_out_ssbo,
                mem::size_of::<BufferOut>() as GLsizeiptr,
                &buffer_out as *const BufferOut as *const GLvoid,
                gl::STATIC_DRAW | gl::STATIC_READ,
            );
            gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 12, buffer_out_ssbo);
        }

        unsafe {
            gl::ActiveTexture(gl::TEXTURE0);
            tex_array.bind();
            shader.set_i32("u_texture", 0);

            shader.bind();
            gl::DispatchCompute(1, 1, 1);
            gl::MemoryBarrier(gl::ALL_BARRIER_BITS);
            shader.unbind();

            gl::GetNamedBufferSubData(
                buffer_out_ssbo,
                0,
                mem::size_of::<BufferOut>() as GLsizeiptr,
                &mut buffer_out as *mut BufferOut as *mut GLvoid,
            );
        }

        println!("total stack frames: {}", buffer_out.stack_ptr + 1);
        for i in 0..=buffer_out.stack_ptr {
            println!("f{}: {:?}", i, buffer_out.stack[i as usize]);
        }
        println!("\n{:?}", buffer_out.result);

        window.close();

        // assert last frame to be:
        //  f18: StackFrame { t_min: 31.0, ptr: 89, idx: 6, parent_octant_idx: 1, scale: 17, is_child: AlignedBool(true), is_leaf: AlignedBool(true) }
        assert_eq!(buffer_out.stack_ptr, 18);
        assert_eq!(buffer_out.stack[18], StackFrame {
            t_min: 31.0,
            ptr: 89,
            idx: 6,
            parent_octant_idx: 1,
            scale: 17,
            is_child: AlignedBool(true),
            is_leaf: AlignedBool(true),
        });
    }
}

// TODO test cases:
// - do INTERSECT, PUSH, ADVANCE & POP work properly from inside and outside the octree from various positions & directions?
//      - when INSIDE: does it PUSH to lowest child level first?
//      - ensure that the correct block is hit and correct color, uv and other result data is returned
//      - test with blocks at all possible mask positions to ensure all the bit logic is working correctly
//      - test absolute & relative pointers?
// - test what happens for 0 direction values on at least on axis
// - test that alpha=0 is properly traced in all cases (inside same octant but also across different octants)
//      - are adjacent blocks skipped?
//      - is cast_translucent flag properly taken into account?
// - does inside block detection work?
