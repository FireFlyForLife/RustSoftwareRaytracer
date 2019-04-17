use crate::math_extensions::*;

pub struct DiffuseMaterial {
    pub albedo: Vec3,
    //Size of 3*4 = 12 bytes
}

pub struct MetalicMaterial {
    pub albedo: Vec3,
    pub fuzz: f32,
    //Size of 3*4+1*3 = 16 bytes
}

pub enum Material {
    Diffuse(DiffuseMaterial),
    Metalic(MetalicMaterial),
}

pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    // Size: 3*4+1*4 = 16 bytes
}

#[allow(dead_code)]
struct AABB {
    pub top_left: Vec3,
    pub extend: Vec3,
    //Size: 6*4 = 24 bytes
}

pub struct Polygon {
    pub vertex_positions: [Vec3; 3],
    pub vertex_normals: [Vec3; 3],
    pub vertex_uvs: [Vec2; 3],
}

pub enum Shape {
    Sphere(Sphere),
    Mesh(Vec<Polygon>),
}

pub struct Object {
    pub material: Material,
    pub shape: Shape,
}
