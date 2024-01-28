use std::f32::consts::PI;
use std::ffi::c_int;
use std::ops::Add;

use cgmath::{ElementWise, InnerSpace, Matrix4, SquareMatrix, Vector2, Vector3, Zero};

use crate::core::Frame;
use crate::gamelogic;
use crate::gamelogic::content::blocks;
use crate::graphics::resource::Resource;
use crate::graphics::screen_quad::ScreenQuad;
use crate::graphics::shader::{ShaderError, ShaderProgram, ShaderProgramBuilder};
use crate::graphics::svo_picker::{PickerBatch, PickerBatchResult, RayResult};
use crate::systems::physics::{Entity, Raycaster};
use crate::world::chunk::{BlockId, BlockPos, Chunk};

/// Gameplay handles all user input and uses it to implement the gameplay logic. The in-game UI is
/// also rendered here.
pub struct Gameplay {
    ui_view: Matrix4<f32>,
    crosshair_shader: Resource<ShaderProgram, ShaderError>,
    screen_quad: ScreenQuad,

    is_jumping: bool,
    was_grounded: bool,
    pub looking_at_block: Option<RayResult>,
    selected_block: BlockId,

    look_ray_batch: PickerBatch,
    look_ray_result: PickerBatchResult,
}

impl Gameplay {
    const FLY_SPEED: f32 = 60.0;
    const NORMAL_SPEED: f32 = 9.0;
    const SPRINT_FACTOR: f32 = 1.5;
    const JUMP_SPEED: f32 = 13.0;
    const ROTATION_SPEED: f32 = 0.002;

    pub fn new() -> Gameplay {
        Gameplay {
            ui_view: Matrix4::identity(),
            crosshair_shader: Resource::new(
                || ShaderProgramBuilder::new().load_shader_bundle("assets/shaders/crosshair.glsl")?.build()
            ).unwrap(),
            screen_quad: ScreenQuad::new(),
            is_jumping: false,
            was_grounded: false,
            looking_at_block: None,
            selected_block: blocks::GRASS,
            look_ray_batch: PickerBatch::with_capacity(1),
            look_ray_result: PickerBatchResult::with_capacity(1),
        }
    }

    pub fn update(&mut self, frame: &mut Frame, player: &mut Entity, world: &mut gamelogic::world::World) {
        if frame.input.was_key_pressed(&glfw::Key::Escape) {
            frame.request_close();
        }
        if frame.is_cursor_grabbed() {
            self.handle_mouse_movement(frame, player);
        }

        self.handle_movement(frame, player);
        self.handle_voxel_placement(frame, player, world);
    }

    pub fn handle_window_resize(&mut self, width: i32, height: i32) {
        self.ui_view = cgmath::ortho(0.0, width as f32, height as f32, 0.0, -1.0, 1.0);
    }

    pub fn reload_resources(&mut self) {
        if let Err(e) = self.crosshair_shader.reload() {
            println!("error reloading crosshair shader: {:?}", e);
        }
    }

    fn handle_movement(&mut self, frame: &Frame, player: &mut Entity) {
        let forward = player.get_forward()
            .mul_element_wise(Vector3::new(1.0, 0.0, 1.0))
            .normalize();
        let right = forward.cross(Vector3::unit_y()).normalize();

        let speed = if player.caps.flying {
            Self::FLY_SPEED
        } else if frame.input.is_key_pressed(&glfw::Key::LeftShift) {
            Self::NORMAL_SPEED * Self::SPRINT_FACTOR
        } else {
            Self::NORMAL_SPEED
        };

        let mut impulse = Vector3::new(0.0, 0.0, 0.0);

        if frame.input.is_key_pressed(&glfw::Key::W) {
            let speed = forward * speed;
            impulse += speed;
        }
        if frame.input.is_key_pressed(&glfw::Key::S) {
            let speed = -forward * speed;
            impulse += speed;
        }
        if frame.input.is_key_pressed(&glfw::Key::A) {
            let speed = -right * speed;
            impulse += speed;
        }
        if frame.input.is_key_pressed(&glfw::Key::D) {
            let speed = right * speed;
            impulse += speed;
        }

        if !impulse.is_zero() {
            impulse = impulse.normalize_to(speed);
        }
        player.velocity.x = impulse.x;
        player.velocity.z = impulse.z;

        if frame.input.was_key_pressed(&glfw::Key::F) {
            player.caps.flying = !player.caps.flying;
        }
        if !player.caps.flying {
            let is_grounded = player.get_state().is_grounded;

            if frame.input.is_key_pressed(&glfw::Key::Space) && self.was_grounded {
                if !self.is_jumping {
                    self.is_jumping = true;
                    player.velocity.y = Self::JUMP_SPEED;
                }
            } else if is_grounded {
                self.is_jumping = false;
            }

            self.was_grounded = is_grounded;
        } else {
            self.is_jumping = false;
            self.was_grounded = false;

            player.velocity.y = 0.0;

            if frame.input.is_key_pressed(&glfw::Key::Space) {
                player.velocity.y = speed;
            }
            if frame.input.is_key_pressed(&glfw::Key::LeftShift) {
                player.velocity.y = -speed;
            }
        }
    }

    fn handle_mouse_movement(&mut self, frame: &Frame, player: &mut Entity) {
        let delta = frame.input.get_mouse_delta();
        if delta.x.abs() > 0.01 {
            player.euler_rotation.y += delta.x * Self::ROTATION_SPEED;
        }
        if delta.y.abs() > 0.01 {
            player.euler_rotation.x -= delta.y * Self::ROTATION_SPEED;

            let limit = PI / 2.0 - 0.01;
            player.euler_rotation.x = player.euler_rotation.x.clamp(-limit, limit);
        }
    }

    fn handle_voxel_placement(&mut self, frame: &Frame, player: &Entity, world: &mut gamelogic::world::World) {
        self.look_ray_batch.reset();
        self.look_ray_batch.add_ray(player.position, player.get_forward(), 30.0);

        self.look_ray_result.reset();
        world.world_svo.raycast(&mut self.look_ray_batch, &mut self.look_ray_result);

        let block_result = self.look_ray_result.rays[0];

        if block_result.did_hit() {
            self.looking_at_block = Some(block_result);
        } else {
            self.looking_at_block = None;
        }

        let hot_bar = [blocks::GRASS, blocks::DIRT, blocks::STONE, blocks::STONE_BRICKS, blocks::GLASS];
        for i in 1..=hot_bar.len() {
            let key = glfw::Key::Num1 as c_int + (i - 1) as c_int;
            let key = &key as *const c_int as *const glfw::Key;
            let key = unsafe { &*key };

            if frame.input.was_key_pressed(key) {
                self.selected_block = i as BlockId;
            }
        }

        // removing blocks
        if frame.input.is_button_pressed_once(&glfw::MouseButton::Button1) && block_result.did_hit() {
            let x = block_result.pos.x.floor() as i32;
            let y = block_result.pos.y.floor() as i32;
            let z = block_result.pos.z.floor() as i32;
            world.world.set_block(x, y, z, blocks::AIR);
        }

        // block picking
        if frame.input.is_button_pressed_once(&glfw::MouseButton::Button3) && block_result.did_hit() {
            let x = block_result.pos.x.floor() as i32;
            let y = block_result.pos.y.floor() as i32;
            let z = block_result.pos.z.floor() as i32;
            self.selected_block = world.world.get_block(x, y, z);
        }

        // adding blocks
        if frame.input.is_button_pressed_once(&glfw::MouseButton::Button2) && block_result.did_hit() {
            let block_normal = block_result.normal;
            let block_pos = block_result.pos.add(block_normal);
            let x = block_pos.x.floor() as i32 as f32;
            let y = block_pos.y.floor() as i32 as f32;
            let z = block_pos.z.floor() as i32 as f32;

            let aabb = &player.aabb_def;
            let player_min_x = player.position.x + aabb.offset.x;
            let player_min_y = player.position.y + aabb.offset.y - 0.1; // add offset to prevent physics glitches
            let player_min_z = player.position.z + aabb.offset.z;
            let player_max_x = player.position.x + aabb.offset.x + aabb.extents.x;
            let player_max_y = player.position.y + aabb.offset.y + aabb.extents.y;
            let player_max_z = player.position.z + aabb.offset.z + aabb.extents.z;

            if (player_max_x < x || player_min_x > x + 1.0) ||
                (player_max_y < y || player_min_y > y + 1.0) ||
                (player_max_z < z || player_min_z > z + 1.0) ||
                player.caps.flying {
                let did_set = world.world.set_block(x as i32, y as i32, z as i32, self.selected_block);
                if !did_set {
                    // if the block could not be placed because of no chunk being present, manually add the chunk
                    let pos = BlockPos::new(x as i32, y as i32, z as i32);
                    let storage = world.chunk_storage_allocator.allocate();
                    let mut chunk = Chunk::new(pos.chunk, 5, storage);
                    chunk.set_block(pos.rel_x as u32, pos.rel_y as u32, pos.rel_z as u32, self.selected_block);
                    world.add_chunk(chunk);
                }
            }
        }
    }

    pub fn render_ui(&self, screen_size: (i32, i32)) {
        self.render_crosshair(screen_size);
    }

    fn render_crosshair(&self, screen_size: (i32, i32)) {
        self.crosshair_shader.bind();
        self.crosshair_shader.set_f32mat4("u_view", &self.ui_view);
        self.crosshair_shader.set_f32vec2("u_dimensions", &Vector2::new(screen_size.0 as f32, screen_size.1 as f32));

        unsafe {
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

            self.screen_quad.render();

            gl::Disable(gl::BLEND);
        }

        self.crosshair_shader.unbind();
    }
}
