#[macro_use]
extern crate memoffset;

extern crate gl;
extern crate glfw;

use self::gl::types::*;
use glfw::{Action, Context, Key};

use std::ffi::{c_void, CString};
use std::{mem, ptr};

use cgmath;
use cgmath::{InnerSpace, Matrix, MetricSpace};
use image::GenericImageView;
use std::collections::HashSet;
use std::path::Path;

type Point3 = cgmath::Point3<f32>;
type Point2 = cgmath::Point2<f32>;

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

    let (shader, vao) = unsafe {
        let vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
        let c_str_vert =
            CString::new(include_bytes!("../assets/shaders/shader.vert").to_vec()).unwrap();
        gl::ShaderSource(vertex_shader, 1, &c_str_vert.as_ptr(), ptr::null());
        gl::CompileShader(vertex_shader);

        let mut success = gl::FALSE as GLint;
        let mut length = 0;
        let mut info_log = [0; 512];
        gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut success);
        if success != gl::TRUE as GLint {
            gl::GetShaderInfoLog(
                vertex_shader,
                512,
                &mut length,
                info_log.as_mut_ptr() as *mut GLchar,
            );
            panic!(
                "error compiling shader: {}",
                String::from_utf8_lossy(&info_log[..(length as usize)])
            );
        }

        let fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
        let c_str_vert =
            CString::new(include_bytes!("../assets/shaders/shader.frag").to_vec()).unwrap();
        gl::ShaderSource(fragment_shader, 1, &c_str_vert.as_ptr(), ptr::null());
        gl::CompileShader(fragment_shader);

        let mut success = gl::FALSE as GLint;
        let mut length = 0;
        let mut info_log = [0; 512];
        gl::GetShaderiv(fragment_shader, gl::COMPILE_STATUS, &mut success);
        if success != gl::TRUE as GLint {
            gl::GetShaderInfoLog(
                fragment_shader,
                512,
                &mut length,
                info_log.as_mut_ptr() as *mut GLchar,
            );
            panic!(
                "error compiling shader: {}",
                String::from_utf8_lossy(&info_log[..(length as usize)])
            );
        }

        let shader_program = gl::CreateProgram();
        gl::AttachShader(shader_program, vertex_shader);
        gl::AttachShader(shader_program, fragment_shader);
        gl::LinkProgram(shader_program);
        let mut success = gl::FALSE as GLint;
        let mut length = 0;
        let mut info_log = [0; 512];
        gl::GetProgramiv(shader_program, gl::LINK_STATUS, &mut success);
        if success != gl::TRUE as GLint {
            gl::GetProgramInfoLog(
                shader_program,
                512,
                &mut length,
                info_log.as_mut_ptr() as *mut GLchar,
            );
            panic!(
                "error linking program: {}",
                String::from_utf8_lossy(&info_log[..(length as usize)])
            );
        }

        gl::DeleteShader(vertex_shader);
        gl::DeleteShader(fragment_shader);

        #[repr(C)]
        struct Vertex {
            position: cgmath::Point3<f32>,
            uv: cgmath::Point2<f32>,
        }

        let pos = cgmath::Point3::new(-1.0, 0.0, -5.0);
        let scl = 0.5;
        let vertices = vec![
            // front
            Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z - scl), uv: Point2::new(1.0, 1.0) },
            Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z - scl), uv: Point2::new(1.0, 0.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z - scl), uv: Point2::new(0.0, 0.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z - scl), uv: Point2::new(0.0, 1.0) },
            // back
            Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z + scl), uv: Point2::new(1.0, 1.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z + scl), uv: Point2::new(0.0, 1.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z + scl), uv: Point2::new(0.0, 0.0) },
            Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z + scl), uv: Point2::new(1.0, 0.0) },
            // left
            Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z + scl), uv: Point2::new(1.0, 1.0) },
            Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z + scl), uv: Point2::new(1.0, 0.0) },
            Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z - scl), uv: Point2::new(0.0, 0.0) },
            Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z - scl), uv: Point2::new(0.0, 1.0) },
            // right
            Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z + scl), uv: Point2::new(1.0, 1.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z - scl), uv: Point2::new(0.0, 1.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z - scl), uv: Point2::new(0.0, 0.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z + scl), uv: Point2::new(1.0, 0.0) },
            // top
            Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z + scl), uv: Point2::new(1.0, 1.0) },
            Vertex { position: Point3::new(pos.x + scl, pos.y + scl, pos.z - scl), uv: Point2::new(1.0, 0.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z - scl), uv: Point2::new(0.0, 0.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y + scl, pos.z + scl), uv: Point2::new(0.0, 1.0) },
            // bottom
            Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z + scl), uv: Point2::new(1.0, 1.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z + scl), uv: Point2::new(0.0, 1.0) },
            Vertex { position: Point3::new(pos.x - scl, pos.y - scl, pos.z - scl), uv: Point2::new(0.0, 0.0) },
            Vertex { position: Point3::new(pos.x + scl, pos.y - scl, pos.z - scl), uv: Point2::new(1.0, 0.0) },
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
        gl::VertexAttribPointer(
            0,
            3,
            gl::FLOAT,
            gl::FALSE,
            stride,
            offset_of!(Vertex, position) as *const c_void,
        );
        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(
            1,
            2,
            gl::FLOAT,
            gl::FALSE,
            stride,
            offset_of!(Vertex, uv) as *const c_void,
        );
        gl::EnableVertexAttribArray(1);

        gl::BindBuffer(gl::ARRAY_BUFFER, 0);
        gl::BindVertexArray(0);

        (shader_program, vao)
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
    let mut last_mouse_pos = cgmath::Point2::new(0.0, 0.0);
    let mut mouse_delta = cgmath::Vector2::new(0.0, 0.0);

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
            cam_pos = cam_pos + cam_dir * cam_speed * delta;
        }
        if key_states.contains(&glfw::Key::S) {
            cam_pos = cam_pos - cam_dir * cam_speed * delta;
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

        if mouse_delta.x.abs() > 0.01 {
            cam_rotation.y += mouse_delta.x * cam_rot_speed * delta;
        }
        if mouse_delta.y.abs() > 0.01 {
            cam_rotation.x -= mouse_delta.y * cam_rot_speed * delta;
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

            gl::UseProgram(shader);

            let uni_name = CString::new("projection").unwrap();
            gl::UniformMatrix4fv(
                gl::GetUniformLocation(shader, uni_name.as_ptr()),
                1,
                gl::FALSE,
                projection.as_ptr(),
            );
            let uni_name = CString::new("view").unwrap();
            gl::UniformMatrix4fv(
                gl::GetUniformLocation(shader, uni_name.as_ptr()),
                1,
                gl::FALSE,
                view.as_ptr(),
            );

            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, texture);
            let uni_name = CString::new("u_texture").unwrap();
            gl::Uniform1i(gl::GetUniformLocation(shader, uni_name.as_ptr()), 0);

            gl::BindVertexArray(vao);
            gl::DrawElements(gl::TRIANGLES, 6 * 6, gl::UNSIGNED_INT, ptr::null());

            gl_check_error!();
        }

        window.swap_buffers();
        glfw.poll_events();
    }
}
