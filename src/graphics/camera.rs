use cgmath::{Matrix4, Vector3, Point3, InnerSpace, SquareMatrix};

pub struct Camera {
    pub position: Point3<f32>,
    pub forward: Vector3<f32>,
    pub up: Vector3<f32>,

    projection: Matrix4<f32>,
}

impl Camera {
    pub fn new(fovy: f32, aspect: f32, near: f32, far: f32) -> Camera {
        let mut cam = Camera {
            position: Point3::new(0.0, 0.0, 0.0),
            forward: Vector3::new(0.0, 0.0, -1.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            projection: Matrix4::identity(),
        };
        cam.update_projection(fovy, aspect, near, far);
        cam
    }

    pub fn update_projection(&mut self, fovy: f32, aspect: f32, near: f32, far: f32) {
        self.projection = cgmath::perspective(cgmath::Deg(fovy), aspect, near, far);
    }

    pub fn right(&self) -> Vector3<f32> {
        self.forward.cross(self.up).normalize()
    }

    pub fn get_projection_matrix(&self) -> &cgmath::Matrix4<f32> {
        &self.projection
    }

    pub fn get_world_to_camera_matrix(&self) -> cgmath::Matrix4<f32> {
        cgmath::Matrix4::look_to_rh(self.position, self.forward, self.up)
    }

    pub fn get_camera_to_world_matrix(&self) -> cgmath::Matrix4<f32> {
        self.get_world_to_camera_matrix().invert().unwrap()
    }

    pub fn set_forward_from_euler(&mut self, euler: Vector3<f32>) {
        self.forward = Vector3::new(
            (euler.y.cos() * euler.x.cos()) as f32,
            (euler.x.sin()) as f32,
            (euler.y.sin() * euler.x.cos()) as f32,
        ).normalize();
    }
}
