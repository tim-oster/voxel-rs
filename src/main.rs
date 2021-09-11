extern crate gl;
extern crate glfw;
#[macro_use]
extern crate memoffset;

use std::{mem, ptr};
use std::collections::HashSet;
use std::ffi::{c_void};
use std::path::Path;

use cgmath;
use cgmath::{ElementWise, EuclideanSpace, InnerSpace, MetricSpace};
use glfw::{Action, Context, Key};
use image::GenericImageView;

use self::gl::types::*;

mod graphics;

type Point3 = cgmath::Point3<f32>;
type Point2 = cgmath::Point2<f32>;
type Vector3 = cgmath::Vector3<f32>;

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
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(4, 6));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));

    let (mut window, events) = glfw
        .create_window(1024, 768, "voxel engine", glfw::WindowMode::Windowed)
        .expect("failed to create window");

    window.make_current();
    window.set_key_polling(true);
    window.set_cursor_pos_polling(true);
    window.set_scroll_polling(true);
    window.set_mouse_button_polling(true);
    window.set_framebuffer_size_polling(true);

    window.set_cursor_mode(glfw::CursorMode::Disabled);

    gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

    let mut last_frame_time = glfw.get_time();

    let mut shader = graphics::Resource::new(
        || graphics::ShaderProgram::new_from_files("assets/shaders/shader.vert", "assets/shaders/shader.frag"),
    ).unwrap();

    let vao = unsafe {
        #[repr(C)]
        struct Vertex {
            position: cgmath::Point3<f32>,
            uv: cgmath::Point2<f32>,
            normal: cgmath::Vector3<f32>,
        }

        let pos = cgmath::Point3::new(0.0, 0.0, -5.0);
        let scl = 0.5;
        let vertices = vec![
            // front
            Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z + scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z + scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z + scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z + scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(0.0, 0.0, 1.0) },
            // back
            Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z - scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(0.0, 0.0, -1.0) },
            Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z - scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(0.0, 0.0, -1.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z - scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(0.0, 0.0, -1.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z - scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(0.0, 0.0, -1.0) },
            // left
            Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z + scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(-1.0, 0.0, 0.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z - scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(-1.0, 0.0, 0.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z - scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(-1.0, 0.0, 0.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z + scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(-1.0, 0.0, 0.0) },
            // right
            Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z + scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(1.0, 0.0, 0.0) },
            Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z + scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(1.0, 0.0, 0.0) },
            Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z - scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(1.0, 0.0, 0.0) },
            Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z - scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(1.0, 0.0, 0.0) },
            // top
            Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z + scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(0.0, 1.0, 0.0) },
            Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z - scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(0.0, 1.0, 0.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z - scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(0.0, 1.0, 0.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z + scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(0.0, 1.0, 0.0) },
            // bottom
            Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z + scl), uv: Point2::new(1.0, 1.0), normal: Vector3::new(0.0, -1.0, 0.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z + scl), uv: Point2::new(0.0, 1.0), normal: Vector3::new(0.0, -1.0, 0.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z - scl), uv: Point2::new(0.0, 0.0), normal: Vector3::new(0.0, -1.0, 0.0) },
            Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z - scl), uv: Point2::new(1.0, 0.0), normal: Vector3::new(0.0, -1.0, 0.0) },
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

        vao
    };

    let texture = unsafe {
        let mut id = 0;
        gl::GenTextures(1, &mut id);
        gl::BindTexture(gl::TEXTURE_2D, id);

        gl::TexParameteri(
            gl::TEXTURE_2D,
            gl::TEXTURE_WRAP_S,
            gl::CLAMP_TO_EDGE as GLint,
        );
        gl::TexParameteri(
            gl::TEXTURE_2D,
            gl::TEXTURE_WRAP_R,
            gl::CLAMP_TO_EDGE as GLint,
        );
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
    };

    let cam_speed = 0.2f32;
    let cam_rot_speed = 0.005f32;
    let mut cam_pos = cgmath::Point3::new(0.0, 0.0, 0.0);
    let mut cam_dir = cgmath::Vector3::new(0.0, 0.0, 0.0);
    let mut cam_rotation = cgmath::Vector3::new(0.0, -90f32.to_radians(), 0.0);
    let cam_up = cgmath::Vector3::new(0.0, 1.0, 0.0);

    let mut key_states = HashSet::new();
    let mut old_key_states = HashSet::new();
    let mut last_mouse_pos = cgmath::Point2::new(0.0, 0.0);
    let mut mouse_delta;

    let ambient_intensity = 0.3f32;
    let mut light_dir = Vector3::new(-1.0, -1.0, -1.0).normalize();

    let mut use_mouse_input = true;

    unsafe {
        gl::Enable(gl::DEPTH_TEST);
        gl::DepthFunc(gl::LESS);

        gl::Enable(gl::CULL_FACE);
        gl::CullFace(gl::BACK);
        gl::FrontFace(gl::CCW);
    }

    while !window.should_close() {
        let current_time = glfw.get_time();
        let delta_time = current_time - last_frame_time;
        last_frame_time = current_time;
        let delta = (delta_time / (1.0 / 60.0)) as f32;

        mouse_delta = cgmath::Vector2::new(0.0, 0.0);

        old_key_states = key_states.clone();
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::FramebufferSize(width, height) => unsafe {
                    gl::Viewport(0, 0, width, height)
                },
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    window.set_should_close(true);
                }
                glfw::WindowEvent::Key(key, _, action, _) => match action {
                    glfw::Action::Press => {
                        key_states.insert(key);
                    }
                    glfw::Action::Release => {
                        key_states.remove(&key);
                    }
                    _ => (),
                },
                glfw::WindowEvent::CursorPos(x, y) => {
                    let new_mouse_pos = cgmath::Point2::new(x as f32, y as f32);
                    if last_mouse_pos.distance2(cgmath::Point2::new(0.0, 0.0)) > 0.0 {
                        mouse_delta = new_mouse_pos - last_mouse_pos;
                    }
                    last_mouse_pos = new_mouse_pos;
                }
                _ => (),
            }
        }

        if key_states.contains(&glfw::Key::W) {
            let dir = cam_dir.mul_element_wise(Vector3::new(1.0, 0.0, 1.0)).normalize();
            cam_pos = cam_pos + dir * cam_speed * delta;
        }
        if key_states.contains(&glfw::Key::S) {
            let dir = cam_dir.mul_element_wise(Vector3::new(1.0, 0.0, 1.0)).normalize();
            cam_pos = cam_pos - dir * cam_speed * delta;
        }
        if key_states.contains(&glfw::Key::A) {
            let right = cam_dir.cross(cam_up);
            cam_pos = cam_pos - right * cam_speed * delta;
        }
        if key_states.contains(&glfw::Key::D) {
            let right = cam_dir.cross(cam_up);
            cam_pos = cam_pos + right * cam_speed * delta;
        }
        if key_states.contains(&glfw::Key::Space) {
            cam_pos.y += cam_speed * delta;
        }
        if key_states.contains(&glfw::Key::LeftShift) {
            cam_pos.y -= cam_speed * delta;
        }
        if key_states.contains(&glfw::Key::E) {
            light_dir = cam_dir;
        }
        if key_states.contains(&glfw::Key::R) && !old_key_states.contains(&glfw::Key::R) {
            if let Err(err) = shader.reload() {
                println!("error loading shader: {:?}", err);
            } else {
                println!("reload shader");
            }
        }
        if key_states.contains(&glfw::Key::T) && !old_key_states.contains(&glfw::Key::T) {
            use_mouse_input = !use_mouse_input;
            if use_mouse_input {
                window.set_cursor_mode(glfw::CursorMode::Disabled);
            } else {
                window.set_cursor_mode(glfw::CursorMode::Normal);
            }
        }

        if use_mouse_input {
            if mouse_delta.x.abs() > 0.01 {
                cam_rotation.y += mouse_delta.x * cam_rot_speed * delta;
            }
            if mouse_delta.y.abs() > 0.01 {
                cam_rotation.x -= mouse_delta.y * cam_rot_speed * delta;
            }
        }
        cam_dir = cgmath::Vector3::new(
            (cam_rotation.y.cos() * cam_rotation.x.cos()) as f32,
            (cam_rotation.x.sin()) as f32,
            (cam_rotation.y.sin() * cam_rotation.x.cos()) as f32,
        ).normalize();

        unsafe {
            gl::ClearColor(0.0, 0.0, 0.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            let projection = {
                let fov = 72.0f32;
                let (width, height) = window.get_size();
                let aspect_ratio = width as f32 / height as f32;
                let zfar = 1024.0;
                let znear = 0.1;
                cgmath::perspective(cgmath::Deg(fov), aspect_ratio, znear, zfar)
            };

            let view = cgmath::Matrix4::look_to_rh(cam_pos, cam_dir, cam_up);

            shader.bind();
            shader.set_f32mat4("u_projection", &projection);
            shader.set_f32mat4("u_view", &view);
            shader.set_f32("u_ambient", ambient_intensity);
            shader.set_f32vec3("u_ligth_dir", &light_dir);
            shader.set_f32vec3("u_cam_pos", &cam_pos.to_vec());

            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, texture);
            shader.set_i32("u_texture", 0);

            gl::BindVertexArray(vao);
            gl::DrawElements(gl::TRIANGLES, 6 * 6, gl::UNSIGNED_INT, ptr::null());

            shader.unbind();

            gl_check_error!();
        }

        window.swap_buffers();
        glfw.poll_events();
    }
}
