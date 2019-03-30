use cgmath::Vector3;

pub struct Material {

}

pub struct Sphere {
    pub center: Vector3<f32>,
    pub radius: f32,
    // Size: 3*4+1*4 = 16 bytes
}

struct AABB {
    pub top_left: Vector3<f32>,
    pub extend: Vector3<f32>,
}

pub struct Polygon {
    pub vertex_positions: [Vector3<f32>; 3],
    pub vertex_normals: [Vector3<f32>; 3],
    pub vertex_uvs: [Vector3<f32>; 2],
}

pub enum Shape {
    Sphere(Sphere),
    Mesh(Vec<Polygon>),
}

pub struct Object {
    pub material: Material,
    pub shape: Shape,
}
