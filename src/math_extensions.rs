use cgmath::{Vector2, Vector3, Vector4, Quaternion};
use sdl2::pixels::Color;

pub type Vec2 = Vector2<f32>;
pub type Vec3 = Vector3<f32>;
pub type Vec4 = Vector4<f32>;

pub type IVec2 = Vector2<i32>;
pub type IVec3 = Vector3<i32>;
pub type IVec4 = Vector4<i32>;

pub type Quat = Quaternion<f32>;

// Create a RGBA color value from a 3 component vector, and fill make the alpha channel 255
pub fn vec3_to_color(vec: Vec3) -> Color {
    Color::RGBA((vec.x * 255.) as u8, (vec.y * 255.) as u8, (vec.z * 255.) as u8, 255)
}

// Create a Vec3 from normalizing the rgb components of the color
pub fn color_to_vec3(color: Color) -> Vec3 {
    Vec3::new(color.r as f32 / 255., color.g as f32 / 255., color.b as f32 / 255.)
}