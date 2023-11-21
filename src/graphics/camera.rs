#![allow(dead_code)]

use cgmath::{InnerSpace, Matrix4, Point3, SquareMatrix, Vector3};

pub struct Camera {
    pub position: Point3<f32>,
    pub forward: Vector3<f32>,
    pub up: Vector3<f32>,

    fov_y_deg: f32,
    projection: Matrix4<f32>,
}

impl Camera {
    pub fn new(fov_y_deg: f32, aspect: f32, near: f32, far: f32) -> Camera {
        let mut cam = Camera {
            position: Point3::new(0.0, 0.0, 0.0),
            forward: Vector3::new(0.0, 0.0, -1.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            fov_y_deg,
            projection: Matrix4::identity(),
        };
        cam.update_projection(fov_y_deg, aspect, near, far);
        cam
    }

    pub fn update_projection(&mut self, fov_y_deg: f32, aspect: f32, near: f32, far: f32) {
        self.fov_y_deg = fov_y_deg;
        self.projection = cgmath::perspective(cgmath::Deg(fov_y_deg), aspect, near, far);
    }

    pub fn right(&self) -> Vector3<f32> {
        self.forward.cross(self.up).normalize()
    }

    pub fn get_fov_y_deg(&self) -> f32 {
        self.fov_y_deg
    }

    pub fn get_projection_matrix(&self) -> &Matrix4<f32> {
        &self.projection
    }

    pub fn get_world_to_camera_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_to_rh(self.position, self.forward, self.up)
    }

    pub fn get_camera_to_world_matrix(&self) -> Matrix4<f32> {
        self.get_world_to_camera_matrix().invert().unwrap()
    }

    pub fn set_forward_from_euler(&mut self, euler: Vector3<f32>) {
        self.forward = Vector3::new(
            euler.y.cos() * euler.x.cos(),
            euler.x.sin(),
            euler.y.sin() * euler.x.cos(),
        ).normalize();
    }
}
