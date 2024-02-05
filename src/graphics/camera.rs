#![allow(dead_code)]

use std::ops::Sub;

use cgmath::{InnerSpace, Matrix4, Point3, SquareMatrix, Vector3};

pub struct Camera {
    pub position: Point3<f32>,
    pub forward: Vector3<f32>,
    pub up: Vector3<f32>,

    fov_y_deg: f32,
    aspect_ratio: f32,
    near: f32,
    far: f32,
    projection: Matrix4<f32>,
}

impl Camera {
    pub fn new(fov_y_deg: f32, aspect_ratio: f32, near: f32, far: f32) -> Camera {
        let mut cam = Camera {
            position: Point3::new(0.0, 0.0, 0.0),
            forward: Vector3::new(0.0, 0.0, -1.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            fov_y_deg,
            aspect_ratio,
            near,
            far,
            projection: Matrix4::identity(),
        };
        cam.update_projection(fov_y_deg, aspect_ratio, near, far);
        cam
    }

    pub fn update_projection(&mut self, fov_y_deg: f32, aspect_ratio: f32, near: f32, far: f32) {
        self.fov_y_deg = fov_y_deg;
        self.aspect_ratio = aspect_ratio;
        self.near = near;
        self.far = far;
        self.projection = cgmath::perspective(cgmath::Deg(fov_y_deg), aspect_ratio, near, far);
    }

    pub fn right(&self) -> Vector3<f32> {
        self.forward.cross(self.up).normalize()
    }

    pub fn get_fov_y_deg(&self) -> f32 {
        self.fov_y_deg
    }
    pub fn set_fov_y_deg(&mut self, fov: f32) {
        self.update_projection(fov, self.aspect_ratio, self.near, self.far);
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

    pub fn get_world_to_clip_space_matrix(&self) -> Matrix4<f32> {
        self.projection * self.get_world_to_camera_matrix()
    }

    /// is_in_frustum performs "radar frustum culling" to check if the given sphere is inside the
    /// camera's frustum.
    /// It transforms the point into camera view space and uses the distance to the near plane
    /// and the FOV to figure out the frustum width and height at the point's depth. Using that
    /// it culls the sphere against all frustum planes.
    pub fn is_in_frustum(&self, point: Point3<f32>, r: f32) -> bool {
        let cp = point.sub(self.position);

        let cz = cp.dot(self.forward);
        if cz + r < self.near || cz - r > self.far {
            return false;
        }
        let cz = cz - self.near;

        let right = self.right();
        let up = self.forward.cross(right);
        let cy = cp.dot(up);
        let hh = cz * f32::tan(self.get_fov_y_deg().to_radians() / 2.0);
        if cy + r < -hh || cy - r > hh {
            return false;
        }

        let cx = cp.dot(right);
        let wh = hh * self.aspect_ratio;
        if cx + r < -wh || cx - r > wh {
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use cgmath::{Point3, Vector3};

    use crate::graphics::camera::Camera;

    /// Tests if culling works along all axes of a camera's frustum.
    #[test]
    fn is_in_frustum() {
        let mut camera = Camera::new(72.0, 1.0, 0.01, 30.0);
        camera.position = Point3::new(0.0, 0.0, 0.0);
        camera.forward = Vector3::new(0.0, 0.0, 1.0);

        assert!(!camera.is_in_frustum(Point3::new(0.0, 0.0, 0.0), 0.0));
        assert!(camera.is_in_frustum(Point3::new(0.0, 0.0, 10.0), 0.0));
        assert!(camera.is_in_frustum(Point3::new(0.0, 0.0, 29.0), 0.0));
        assert!(!camera.is_in_frustum(Point3::new(0.0, 0.0, 31.0), 0.0));
        assert!(camera.is_in_frustum(Point3::new(0.0, 0.0, 0.0), 1.0));
        assert!(camera.is_in_frustum(Point3::new(0.0, 0.0, 31.0), 1.0));

        assert!(camera.is_in_frustum(Point3::new(0.0, 0.0, 3.0), 0.0));
        assert!(camera.is_in_frustum(Point3::new(0.0, 2.0, 3.0), 0.0));
        assert!(!camera.is_in_frustum(Point3::new(0.0, 3.0, 3.0), 0.0));
        assert!(camera.is_in_frustum(Point3::new(0.0, -2.0, 3.0), 0.0));
        assert!(!camera.is_in_frustum(Point3::new(0.0, -3.0, 3.0), 0.0));
        assert!(camera.is_in_frustum(Point3::new(0.0, 3.0, 3.0), 1.0));
        assert!(camera.is_in_frustum(Point3::new(0.0, -3.0, 3.0), 1.0));

        assert!(camera.is_in_frustum(Point3::new(0.0, 0.0, 3.0), 0.0));
        assert!(camera.is_in_frustum(Point3::new(2.0, 0.0, 3.0), 0.0));
        assert!(!camera.is_in_frustum(Point3::new(3.0, 0.0, 3.0), 0.0));
        assert!(camera.is_in_frustum(Point3::new(-2.0, 0.0, 3.0), 0.0));
        assert!(!camera.is_in_frustum(Point3::new(-3.0, 0.0, 3.0), 0.0));
        assert!(camera.is_in_frustum(Point3::new(3.0, 0.0, 3.0), 1.0));
        assert!(camera.is_in_frustum(Point3::new(-3.0, 0.0, 3.0), 1.0));
    }
}
