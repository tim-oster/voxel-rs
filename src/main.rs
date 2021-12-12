extern crate gl;
#[macro_use]
extern crate memoffset;

use std::{mem, ptr};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;

use cgmath;
use cgmath::{Point2, Point3, Vector3};
use cgmath::{ElementWise, EuclideanSpace, InnerSpace};
use gl::types::*;
use image::GenericImageView;
use imgui::{Condition, Window};

mod graphics;
mod core;

unsafe fn gl_check_error_(file: &str, line: u32) -> u32 {
    let mut error_code = gl::GetError();
    while error_code != gl::NO_ERROR {
        let error = match error_code {
            gl::INVALID_ENUM => "INVALID_ENUM",
            gl::INVALID_VALUE => "INVALID_VALUE",
            gl::INVALID_OPERATION => "INVALID_OPERATION",
            gl::STACK_OVERFLOW => "STACK_OVERFLOW",
            gl::STACK_UNDERFLOW => "STACK_UNDERFLOW",
            gl::OUT_OF_MEMORY => "OUT_OF_MEMORY",
            gl::INVALID_FRAMEBUFFER_OPERATION => "INVALID_FRAMEBUFFER_OPERATION",
            _ => "unknown GL error code",
        };

        println!("{} | {} ({})", error, file, line);

        error_code = gl::GetError();
    }
    error_code
}

macro_rules! gl_check_error {
    () => {
        gl_check_error_(file!(), line!())
    };
}

fn main() {
    let svo = build_voxel_model();

    let mut window = core::Window::new(1024, 768, "voxel engine");
    window.request_grab_cursor(true);

    let mut shader = graphics::Resource::new(
        || graphics::ShaderProgram::new_from_files("assets/shaders/shader.vert", "assets/shaders/shader.frag"),
    ).unwrap();

    let (vao, indices_count) = build_vao();
    // let texture = build_texture();

    let mut camera = graphics::Camera::new(72.0, window.get_aspect(), 0.01, 1024.0);
    camera.position = Point3::new(128.0, 80.0, 128.0);

    let cam_speed = 1f32;
    let cam_rot_speed = 0.005f32;
    let mut cam_rot = cgmath::Vector3::new(0.0, -90f32.to_radians(), 0.0);

    let ambient_intensity = 0.3f32;
    let mut light_dir = Vector3::new(-1.0, -1.0, -1.0).normalize();

    let mut use_mouse_input = true;

    let mut ssbo = 0;

    unsafe {
        gl::GenBuffers(1, &mut ssbo);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, ssbo);
        gl::BufferData(gl::SHADER_STORAGE_BUFFER, (svo.descriptors.len() * 4) as GLsizeiptr, &svo.descriptors[0] as *const i32 as *const c_void, gl::STATIC_READ);
        gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 3, ssbo);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);

        gl::Enable(gl::DEPTH_TEST);
        gl::DepthFunc(gl::LESS);

        gl::Enable(gl::CULL_FACE);
        gl::CullFace(gl::BACK);
        gl::FrontFace(gl::CCW);
    }

    while !window.should_close() {
        window.update(|frame| {
            Window::new("Debug")
                .size([300.0, 100.0], Condition::FirstUseEver)
                .build(&frame.ui, || {
                    frame.ui.text(format!(
                        "fps: {}, frame: {:.2}ms, update: {:.2}ms",
                        frame.stats.frames_per_second,
                        frame.stats.avg_frame_time_per_second * 1000.0,
                        frame.stats.avg_update_time_per_second * 1000.0,
                    ));
                    frame.ui.text(format!(
                        "cam pos: ({:.3},{:.3},{:.3})",
                        camera.position.x, camera.position.y, camera.position.z,
                    ));
                    frame.ui.text(format!(
                        "cam fwd: ({:.3},{:.3},{:.3})",
                        camera.forward.x, camera.forward.y, camera.forward.z,
                    ));
                });

            if frame.was_resized {
                camera.update_projection(72.0, frame.get_aspect(), 0.01, 1024.0);
            }
            if frame.input.was_key_pressed(&glfw::Key::Escape) {
                frame.request_close();
            }
            if frame.input.is_key_pressed(&glfw::Key::W) {
                let dir = camera.forward.mul_element_wise(Vector3::new(1.0, 0.0, 1.0)).normalize();
                camera.position += dir * cam_speed * frame.stats.delta_time;
            }
            if frame.input.is_key_pressed(&glfw::Key::S) {
                let dir = camera.forward.mul_element_wise(Vector3::new(1.0, 0.0, 1.0)).normalize();
                camera.position -= dir * cam_speed * frame.stats.delta_time;
            }
            if frame.input.is_key_pressed(&glfw::Key::A) {
                camera.position -= camera.right() * cam_speed * frame.stats.delta_time;
            }
            if frame.input.is_key_pressed(&glfw::Key::D) {
                camera.position += camera.right() * cam_speed * frame.stats.delta_time;
            }
            if frame.input.is_key_pressed(&glfw::Key::Space) {
                camera.position.y += cam_speed * frame.stats.delta_time;
            }
            if frame.input.is_key_pressed(&glfw::Key::LeftShift) {
                camera.position.y -= cam_speed * frame.stats.delta_time;
            }
            if frame.input.was_key_pressed(&glfw::Key::E) {
                light_dir = camera.forward;
            }
            if frame.input.was_key_pressed(&glfw::Key::R) {
                if let Err(err) = shader.reload() {
                    println!("error loading shader: {:?}", err);
                } else {
                    println!("reload shader");
                }
            }
            if frame.input.was_key_pressed(&glfw::Key::T) {
                use_mouse_input = !use_mouse_input;
                frame.request_grab_cursor(use_mouse_input);
            }

            if use_mouse_input {
                let delta = frame.input.get_mouse_delta();
                if delta.x.abs() > 0.01 {
                    cam_rot.y += delta.x * cam_rot_speed * frame.stats.delta_time;
                }
                if delta.y.abs() > 0.01 {
                    cam_rot.x -= delta.y * cam_rot_speed * frame.stats.delta_time;
                }
                camera.set_forward_from_euler(cam_rot);
            }

            unsafe {
                gl::ClearColor(0.0, 0.0, 0.0, 1.0);
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                shader.bind();
                // shader.set_f32mat4("u_projection", camera.get_projection_matrix());
                // shader.set_f32mat4("u_view", &camera.get_view_matrix());
                shader.set_f32("u_ambient", ambient_intensity);
                shader.set_f32vec3("u_light_dir", &light_dir);
                shader.set_f32vec3("u_cam_pos", &camera.position.to_vec());

                shader.set_f32mat4("u_view", &camera.get_camera_to_world_matrix());
                shader.set_f32("u_fovy", 70.0f32.to_radians());
                shader.set_f32("u_aspect", frame.get_aspect());
                shader.set_i32("u_max_depth", svo.max_depth);

                // gl::ActiveTexture(gl::TEXTURE0);
                // gl::BindTexture(gl::TEXTURE_2D, texture);
                // shader.set_i32("u_texture", 0);

                gl::BindVertexArray(vao);
                gl::DrawElements(gl::TRIANGLES, indices_count, gl::UNSIGNED_INT, ptr::null());

                shader.unbind();

                gl_check_error!();
            }
        });
    }
}

fn build_vao() -> (GLuint, i32) {
    unsafe {
        #[repr(C)]
        struct Vertex {
            position: cgmath::Point3<f32>,
            uv: cgmath::Point2<f32>,
            normal: cgmath::Vector3<f32>,
        }

        // let pos = Point3::new(0.0, 0.0, -5.0);
        // let scl = 0.5;
        let vertices = vec![
            // screen quad
            Vertex { position: Point3::new(1.0, 1.0, -1.0), uv: Point2::new(1.0, 1.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            Vertex { position: Point3::new(-1.0, 1.0, -1.0), uv: Point2::new(0.0, 1.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            Vertex { position: Point3::new(-1.0, -1.0, -1.0), uv: Point2::new(0.0, 0.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            Vertex { position: Point3::new(1.0, -1.0, -1.0), uv: Point2::new(1.0, 0.0), normal: Vector3::new(0.0, 0.0, 1.0) },

            // // front
            // Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z + scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z + scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z + scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            // Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z + scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            // // back
            // Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z - scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(0.0, 0.0, -1.0) },
            // Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z - scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(0.0, 0.0, -1.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z - scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(0.0, 0.0, -1.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z - scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(0.0, 0.0, -1.0) },
            // // left
            // Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z + scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(-1.0, 0.0, 0.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z - scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(-1.0, 0.0, 0.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z - scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(-1.0, 0.0, 0.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z + scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(-1.0, 0.0, 0.0) },
            // // right
            // Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z + scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(1.0, 0.0, 0.0) },
            // Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z + scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(1.0, 0.0, 0.0) },
            // Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z - scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(1.0, 0.0, 0.0) },
            // Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z - scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(1.0, 0.0, 0.0) },
            // // top
            // Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z + scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(0.0, 1.0, 0.0) },
            // Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z - scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(0.0, 1.0, 0.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z - scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(0.0, 1.0, 0.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z + scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(0.0, 1.0, 0.0) },
            // // bottom
            // Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z + scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(0.0, -1.0, 0.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z + scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(0.0, -1.0, 0.0) },
            // Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z - scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(0.0, -1.0, 0.0) },
            // Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z - scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(0.0, -1.0, 0.0) },
        ];

        let mut indices = Vec::<i32>::new();
        for i in 0..(vertices.len() / 4) {
            let mut base = vec![0i32, 1, 3, 1, 2, 3];
            for elem in base.iter_mut() {
                *elem += (i as i32) * 4;
            }
            indices.extend(&base);
        }

        let (mut vbo, mut vao, mut ebo) = (0, 0, 0);
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);
        gl::GenBuffers(1, &mut ebo);
        gl::BindVertexArray(vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (vertices.len() * mem::size_of::<Vertex>()) as GLsizeiptr,
            &vertices[0] as *const Vertex as *const c_void,
            gl::STATIC_DRAW,
        );

        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
        gl::BufferData(
            gl::ELEMENT_ARRAY_BUFFER,
            (indices.len() * mem::size_of::<GLint>()) as GLsizeiptr,
            &indices[0] as *const i32 as *const c_void,
            gl::STATIC_DRAW,
        );

        let stride = mem::size_of::<Vertex>() as i32;

        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, stride, offset_of!(Vertex, position) as *const c_void);
        gl::EnableVertexAttribArray(0);

        gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, stride, offset_of!(Vertex, uv) as *const c_void);
        gl::EnableVertexAttribArray(1);

        gl::VertexAttribPointer(2, 3, gl::FLOAT, gl::FALSE, stride, offset_of!(Vertex, normal) as *const c_void);
        gl::EnableVertexAttribArray(2);

        gl::BindBuffer(gl::ARRAY_BUFFER, 0);
        gl::BindVertexArray(0);

        (vao, vertices.len() as i32 / 4 * 6)
    }
}

fn build_texture() -> GLuint {
    unsafe {
        let mut id = 0;
        gl::GenTextures(1, &mut id);
        gl::BindTexture(gl::TEXTURE_2D, id);

        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as GLint);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_R, gl::CLAMP_TO_EDGE as GLint);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as GLint);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as GLint);

        let path = Path::new("assets/textures/grass_top.png");
        let image = image::open(&path).expect("failed to load image");
        let data = image.to_rgba8().into_raw();
        gl::TexImage2D(
            gl::TEXTURE_2D,
            0,
            gl::RGBA8 as i32,
            image.width() as i32,
            image.height() as i32,
            0,
            gl::RGBA,
            gl::UNSIGNED_BYTE,
            &data[0] as *const u8 as *const c_void,
        );
        gl::GenerateMipmap(gl::TEXTURE_2D);

        id
    }
}

#[derive(PartialEq, Eq, Hash)]
struct NodePos {
    x: u8,
    y: u8,
    z: u8,
}

struct TreeNode {
    nodes: [Option<Box<TreeNode>>; 8],
    color: Option<i32>,
}

impl TreeNode {
    fn new() -> Box<TreeNode> {
        Box::new(TreeNode {
            nodes: Default::default(),
            color: None,
        })
    }

    fn build_descriptor(&self) -> i32 {
        if let Some(color) = self.color {
            return color;
        }

        let mut child_mask = 0;
        let mut leaf_mask = 0;

        for (i, child) in self.nodes.iter().enumerate() {
            if child.is_none() {
                continue;
            }
            let child = child.as_ref().unwrap();
            child_mask |= 1 << i;
            if child.color.is_some() {
                leaf_mask |= 1 << i;
            }
        }

        (child_mask << 8) | leaf_mask
    }

    fn build(&self) -> Vec<i32> {
        self.build_with_far_pointer_threshold(0x7fff)
    }

    fn build_with_far_pointer_threshold(&self, far_pointer_threshold: usize) -> Vec<i32> {
        let mut result = Vec::new();
        result.push(self.build_descriptor());
        if ((result[0] >> 8) & 0xff) == 0 {
            return result;
        }

        result[0] |= 1 << 17;

        struct ChildDesc {
            index: usize,
            desc: Vec<i32>,
            far_ptr_index: Option<usize>,
        }
        let mut full_child_descriptors = Vec::new();

        let mut last_child_index = None;
        for child in self.nodes.iter() {
            if child.is_none() {
                // skip leading empty children
                if result.len() > 1 {
                    result.push(0);
                }
                continue;
            }
            let child = child.as_ref().unwrap();
            result.push(child.build_descriptor());
            last_child_index = Some(result.len() - 1);
            if child.color.is_some() {
                continue;
            }
            full_child_descriptors.push(ChildDesc {
                index: result.len() - 1,
                desc: child.build_with_far_pointer_threshold(far_pointer_threshold),
                far_ptr_index: None,
            });
        }
        // strip trailing empty children
        if let Some(index) = last_child_index {
            result.truncate(index + 1);
        }

        full_child_descriptors.sort_by(|a, b| {
            let len_a = a.desc.len();
            let len_b = b.desc.len();
            if len_a == len_b {
                return a.index.partial_cmp(&b.index).unwrap();
            }
            len_a.partial_cmp(&len_b).unwrap()
        });

        let mut space_count = 0;
        let start_index = result.len();
        for child in full_child_descriptors.iter_mut() {
            let offset = (start_index + space_count) - child.index;
            if offset > far_pointer_threshold {
                result.push(0);
                child.far_ptr_index = Some(result.len() - 1);
                space_count += 1;
            }

            space_count += child.desc.len() - 1; // remove descriptor i32 from space
        }

        for child in full_child_descriptors.iter() {
            match child.far_ptr_index {
                None => {
                    let offset = result.len() - child.index;
                    result[child.index] |= (offset as i32) << 17;
                }
                Some(ptr_index) => {
                    let offset = ptr_index - child.index;
                    result[child.index] |= 1 << 16;
                    result[child.index] |= (offset as i32) << 17;

                    let offset = result.len() - child.index;
                    result[ptr_index] = offset as i32;
                }
            }
            result.extend_from_slice(&child.desc[1..]);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use crate::TreeNode;

    #[test]
    fn build_leaf_node() {
        let mut node = TreeNode::new();
        node.color = Some(0xff);

        let desc = node.build();
        assert_eq!(desc, vec![255]);
    }

    #[test]
    fn build_parent_with_only_leaves() {
        let mut parent = TreeNode::new();
        parent.nodes[1] = Some(TreeNode::new());
        parent.nodes[1].as_mut().unwrap().color = Some(0xa);
        parent.nodes[4] = Some(TreeNode::new());
        parent.nodes[4].as_mut().unwrap().color = Some(0xb);

        let desc = parent.build();
        assert_eq!(desc, vec![
            0b00000000_00000010_00010010_00010010,
            0xa,
            0,
            0,
            0xb,
        ]);
    }

    #[test]
    fn build_parent_with_children_with_leaves() {
        let mut child = TreeNode::new();
        child.nodes[3] = Some(TreeNode::new());
        child.nodes[3].as_mut().unwrap().color = Some(0xa);

        let mut root = TreeNode::new();
        root.nodes[1] = Some(child);
        root.nodes[4] = Some(TreeNode::new());
        root.nodes[4].as_mut().unwrap().color = Some(0xb);

        let desc = root.build();
        assert_eq!(desc, vec![
            0b00000000_00000010_00010010_00010000,
            0b00000000_00001000_00001000_00001000,
            0,
            0,
            0xb,
            //
            0xa,
        ]);
    }

    #[test]
    fn build_parent_with_far_pointers() {
        let mut child_a_a = TreeNode::new();
        child_a_a.nodes[3] = Some(TreeNode::new());
        child_a_a.nodes[3].as_mut().unwrap().color = Some(0xa);

        let mut child_a = TreeNode::new();
        child_a.nodes[3] = Some(child_a_a);

        let mut child_b = TreeNode::new();
        child_b.nodes[3] = Some(TreeNode::new());
        child_b.nodes[3].as_mut().unwrap().color = Some(0xb);

        let mut root = TreeNode::new();
        root.nodes[1] = Some(child_a);
        root.nodes[3] = Some(child_b);

        let desc = root.build_with_far_pointer_threshold(1);
        assert_eq!(desc, vec![
            0b00000000_00000010_00001010_00000000,
            0b00000000_00000111_00001000_00000000,
            0,
            0b00000000_00000100_00001000_00001000,
            //
            5,
            //
            0xb,
            //
            0b00000000_00000010_00001000_00001000,
            //
            0xa,
        ]);
    }
}

struct SVO {
    max_depth: i32,
    descriptors: Vec<i32>,
}

fn build_voxel_model() -> SVO {
    println!("loading model");
    let data = dot_vox::load("assets/terrain.vox").unwrap();
    let model = &data.models[0];

    println!("collecting leaves");
    let mut leaves = HashMap::new();
    for voxel in &model.voxels {
        let pos = NodePos { x: voxel.x, y: voxel.z, z: voxel.y };
        let mut leaf = TreeNode::new();
        leaf.color = Some(data.palette[voxel.i as usize] as i32);
        leaves.insert(pos, leaf);
    }

    let mut slot_counts = model.voxels.len() + leaves.len();

    println!("assembling tree");
    let mut input = leaves;
    while input.len() > 1 {
        let output = merge_tree_nodes(input);
        slot_counts += output.len();
        input = output;
    }

    println!("total of {} bytes = {} MB", slot_counts * 4, slot_counts as f32 * 4.0 / 1000.0 / 1000.0);

    println!("building SVO");
    let descriptors = input[input.keys().next().unwrap()].build();
    println!("entries in SVO: {}", descriptors.len());
    println!("SVO size: {} MB", descriptors.len() as f32 * 4.0 / 1000.0 / 1000.0);

    SVO {
        max_depth: (model.size.x.max(model.size.y.max(model.size.z)) as f32).log2().ceil() as i32,
        descriptors,
    }
}

fn merge_tree_nodes(input: HashMap<NodePos, Box<TreeNode>>) -> HashMap<NodePos, Box<TreeNode>> {
    let mut output = HashMap::new();
    for (pos, node) in input.into_iter() {
        let parent_pos = NodePos { x: pos.x / 2, y: pos.y / 2, z: pos.z / 2 };
        let parent_node = output.entry(parent_pos).or_insert_with(|| TreeNode::new());
        let idx = (pos.x % 2 + (pos.y % 2) * 2 + (pos.z % 2) * 4) as usize;
        parent_node.nodes[idx] = Some(node);
    }
    output
}
