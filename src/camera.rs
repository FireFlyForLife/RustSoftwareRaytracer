use crate::math_extensions::*;
use crate::ray::Ray;

#[derive(Debug)]
pub struct Camera {
    pub origin: Vec3,

    pub lower_left_corner: Vec3,
    pub horizontal: Vec3,
    pub vertical: Vec3,
}

impl Camera {
    pub fn new(origin: Vec3, width: u32, height: u32) -> Camera {
        let aspect_ratio = width as f32 / height as f32;

        Camera{ origin: origin, 
            lower_left_corner: Vec3::new(-1.0 * aspect_ratio, -1.0, -1.0),
            horizontal: Vec3::new(2.0, 0.0, 0.0) * aspect_ratio,
            vertical: Vec3::new(0.0, 2.0, 0.0) }
    }

    pub fn get_ray(&self, u: f32, v: f32) -> Ray {
        Ray::new(self.origin, self.lower_left_corner + u*self.horizontal + v*self.vertical)
    }
}
