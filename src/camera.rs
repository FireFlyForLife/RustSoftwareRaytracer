use crate::math_extensions::*;
use crate::ray::Ray;

use cgmath::prelude::*;

#[derive(Debug)]
pub struct Camera {
    pub origin: Vec3,

    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    /// fovy is in degrees
    pub fn new(origin: Vec3, look_at: Vec3, up_vector: Vec3, width: u32, height: u32, fovy: f32) -> Camera {
        let aspect_ratio = width as f32 / height as f32;
        let theta = fovy * std::f32::consts::PI / 180.0;
        let half_height = (theta / 2.0).tan();
        let half_width = aspect_ratio * half_height;
        let w = (origin - look_at).normalize();
        let u = up_vector.cross(w).normalize();
        let v = w.cross(u);

        Camera{ origin: origin, 
            lower_left_corner: origin - half_width*u - half_height*v - w,
            horizontal: 2.0 * half_width * u,
            vertical: 2.0 * half_height * v }
    }

    pub fn get_ray(&self, u: f32, v: f32) -> Ray {
        Ray::new(self.origin, self.lower_left_corner + u*self.horizontal + v*self.vertical - self.origin)
    }
}
