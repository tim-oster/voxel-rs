#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cgmath::{InnerSpace, Point2, Point3, Vector3, Vector4};
    use glfw::Context;

    use crate::{AlignedPoint3, AlignedVec3, Chunk, ChunkPos, graphics, Position, SerializedChunk, Svo};
    use crate::chunk::ChunkStorage;
    use crate::core::{Config, GlContext};
    use crate::graphics::buffer;
    use crate::graphics::buffer::{Buffer, MappedBuffer};
    use crate::graphics::types::{AlignedBool, AlignedPoint2, AlignedVec4};
    use crate::world::allocator::Allocator;

    #[test]
    fn test() {
        let context = GlContext::new(Config {
            width: 640,
            height: 490,
            title: "",
            msaa_samples: 0,
            headless: true,
        });

        // TODO refactor world & svo setup logic once rest of codebase has been refactored
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

        let world_buffer = MappedBuffer::<u32>::new(1000 * 1024 * 1024 / 4);
        world_buffer.bind_as_storage_buffer(0);
        unsafe {
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
        let material_buffer = Buffer::new(vec![
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
        ], buffer::STATIC_READ);
        material_buffer.bind_as_storage_buffer(2);

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
        let new_buffer_in = Buffer::new(BufferIn {
            max_dst: 32.0,
            pos: AlignedPoint3(Point3::new(0.0, 0.01, 0.0)),
            dir: AlignedVec3(Vector3::new(1.0, 0.00001, 0.00001).normalize()),
            cast_translucent: AlignedBool(false),
        }, buffer::STATIC_READ);
        new_buffer_in.bind_as_storage_buffer(11);

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
        let mut buffer_out = Buffer::new(BufferOut {
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
        }, buffer::STATIC_DRAW | buffer::STATIC_READ);
        buffer_out.bind_as_storage_buffer(12);

        unsafe {
            gl::ActiveTexture(gl::TEXTURE0);
            tex_array.bind();
            shader.set_i32("u_texture", 0);

            shader.bind();
            gl::DispatchCompute(1, 1, 1);
            gl::MemoryBarrier(gl::ALL_BARRIER_BITS);
            shader.unbind();

            buffer_out.pull_data();
        }

        println!("total stack frames: {}", buffer_out.stack_ptr + 1);
        for i in 0..=buffer_out.stack_ptr {
            println!("f{}: {:?}", i, buffer_out.stack[i as usize]);
        }
        println!("\n{:?}", buffer_out.result);

        context.close();

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
