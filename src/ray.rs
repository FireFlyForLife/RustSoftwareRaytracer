use crate::math_extensions::*;

#[derive(Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Ray {
        Ray{ origin, direction}
    }

    pub fn point_at_parameter(&self, t: f32) -> Vec3 {
        self.origin + t * self.direction
    }
}

impl Default for Ray {
    fn default() -> Self {
        Ray{origin: Vec3::new(0.0, 0.0, 0.0), direction: Vec3::new(0.0, 0.0, 0.0)}
    }
}