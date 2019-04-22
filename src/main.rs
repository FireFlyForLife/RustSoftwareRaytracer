#![feature(duration_float)]

mod geometry;
mod math_extensions;
mod ray;
mod camera;

use math_extensions::*;
use ray::Ray;
use geometry::*;
use camera::Camera;

use cgmath::prelude::*;
use cgmath::{Rad, Euler};

use sdl2::pixels::Color;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use std::time::Instant;
use std::vec::Vec;
use std::ptr;
use std::mem;

const SAMPLES_PER_PIXEL: u32 = 1;
const MAX_RAY_RECURSION_DEPTH: u32 = 50;
const WINDOW_WIDTH: u32 = 640; //1280;
const WINDOW_HEIGHT: u32 = 480; //720;

const USE_GAMMA_CORRECTION: bool = false;


fn random_in_unit_sphere() -> Vec3 {
    let mut p;
    /*DO*/ while {
        p = 2.0*Vec3::new(rand::random(), rand::random(), rand::random()) - Vec3::new(1.0, 1.0, 1.0);
    /*WHILE*/ p.magnitude2() >= 1.0
    }{}
    return p;
}

fn reflect(direction: Vec3, surface_normal: Vec3) -> Vec3 {
    return surface_normal - 2.0*direction.dot(surface_normal)*surface_normal;
}

fn refract(direction: Vec3, normal: Vec3, ni_over_nt: f32) -> Option<Vec3> {
    let uv = direction.normalize();
    let dt = uv.dot(normal);
    let discriminant = 1.0 - ni_over_nt*ni_over_nt*(1.0-dt*dt);
    if discriminant > 0.0 {
        let refracted = ni_over_nt*(uv-normal*dt)-normal*discriminant.sqrt();
        Some(refracted)
    } else {
        None
    }
}

fn schlick(cosine: f32, refraction_index: f32) -> f32 {
    let mut r0 = (1.0-refraction_index) / (1.0+refraction_index);
    r0 = r0 * r0;
    r0 + (1.0-r0)*(1.0-cosine).powf(5.0)
}

struct HitRecord<'a> {
    t: f32,
    point: Vec3,
    normal: Vec3,
    material: Option<&'a Material>,
}

impl<'a> Default for HitRecord<'a> {
    fn default() -> Self {
        HitRecord{t: std::f32::MAX, point: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::new(0.0, 0.0, 0.0), material: None}
    }
}

struct Raytracer {
    world: Vec<Object>,

    width: u32,
    height: u32,
    aspect_ratio: f32,
}

struct TotalPixel {
    color: Vec3,
    amount: f32,
}
impl Copy for TotalPixel { }
impl Clone for TotalPixel {
    fn clone(&self) -> TotalPixel {
        TotalPixel{color: self.color, amount: self.amount}
    }
}
impl Default for TotalPixel {
    fn default() -> TotalPixel {        
        TotalPixel{color: Vec3::new(0.0, 0.0, 0.0), amount: 0.0}
    }
}


impl Raytracer {
    pub fn new(width: u32, height: u32) -> Raytracer {
        let aspect_ratio = width as f32 / height as f32;

        Raytracer{ 
            world: Vec::new(), 
            width: width, 
            height: height, 
            aspect_ratio: aspect_ratio
        }
    }

    pub fn draw_frame(&self, camera: &Camera, pixels: &mut [TotalPixel]) {
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel: &mut TotalPixel = &mut pixels[(x + y*self.width) as usize];
                for _sample in 0..SAMPLES_PER_PIXEL {
                    let u = ((x as f32) + rand::random::<f32>()) / (self.width as f32);
                    let v = 1.0 - ((y as f32) + rand::random::<f32>()) / (self.height as f32);
                    let ray = camera.get_ray(u, v);

                    let color = self.trace_ray(&ray, 0);

                    pixel.color += color;
                    pixel.amount += 1.0;
                }
            }
        }
    }

    pub fn trace_ray(&self, ray: &Ray, recursion_depth: u32) -> Vec3 {
        let mut hit_record: HitRecord = Default::default();

        if self.intersect_world(ray, 0.001, std::f32::MAX, &mut hit_record) {
            let scattered: Ray;
            let attenuation: Vec3;

            let option = scatter_from_material(ray, hit_record.material.unwrap(), &hit_record);
            if recursion_depth < MAX_RAY_RECURSION_DEPTH && option.is_some() {
                let (attenuation, scattered) = option.unwrap();
                self.trace_ray(&scattered, recursion_depth+1).mul_element_wise(attenuation)
            } else {
                Vec3{x: 0.0, y: 0.0, z: 0.0}
            }
        } else {
            background_color(ray)
        }
    }

    pub fn intersect_world<'a, 'b>(&'a self, ray: &Ray, min_dist: f32, max_dist: f32, hit_record: &'b mut HitRecord<'a>) -> bool {
        let mut intersected_anything = false;
        let mut closest_hit = max_dist;

        for object in &self.world {
            match &object.shape {
                Shape::Sphere(sphere) => {
                    if ray_to_sphere(ray, &sphere, min_dist, closest_hit, hit_record) {
                        closest_hit = hit_record.t;
                        hit_record.material = Some(&object.material);

                        intersected_anything = true;
                    }
                },
                Shape::Mesh(_mesh) => {},
            }
        }

        return intersected_anything;
    }
}

fn present_to_window(back_buffer: &[TotalPixel], front_buffer: &mut [u8]) {
    assert!(back_buffer.len() == front_buffer.len()/4);
    let buffer_len = back_buffer.len();

    let front_buffer_ptr: *mut Color = unsafe{ mem::transmute( &front_buffer[0] ) };

    for pixelIndex in 0..buffer_len {
        let total_pixel = back_buffer[pixelIndex];
        let mut average_pixel = total_pixel.color / total_pixel.amount;

        if USE_GAMMA_CORRECTION {
            average_pixel.x = average_pixel.x.sqrt();
            average_pixel.y = average_pixel.y.sqrt();
            average_pixel.z = average_pixel.z.sqrt();
        }

        //The backbuffer format is BGRA so we swap the x and z channels here
        let color = Color{r: (average_pixel.z * 255.0) as u8, g: (average_pixel.y * 255.0) as u8, b: (average_pixel.x * 255.0) as u8, a: 255u8};
        unsafe{ ptr::write(front_buffer_ptr.offset(pixelIndex as isize), color); }
    }
}

// @returns Option<(attenuation: Vec3, scattered_ray: Ray)>
fn scatter_from_material(ray_in: &Ray, material: &Material, hit_record: &HitRecord) -> Option<(Vec3, Ray)> {
    match material {
        Material::Diffuse(diffuse_material) => {
            let target = hit_record.point + hit_record.normal + random_in_unit_sphere();
            let scattered = Ray::new(hit_record.point, target - hit_record.point);
            let attenuation = diffuse_material.albedo;

            Some( (attenuation, scattered) )
        },
        Material::Metalic(metalic_material) => {
            let reflected = reflect(ray_in.direction.normalize(), hit_record.normal);
            let scattered = Ray::new(hit_record.point, reflected + metalic_material.fuzz*random_in_unit_sphere());
            let attenuation = metalic_material.albedo;

            if scattered.direction.dot(hit_record.normal) > 0.0 {
                Some( (attenuation, scattered) )
            } else {
                None
            }
        },
        Material::Dielectric(dielectric_material) => {
            let outward_normal;
            let ni_over_nt;
            let reflected = reflect(ray_in.direction, hit_record.normal);
            let attenuation = Vec3::new(1.0, 1.0, 1.0);
            let reflect_probability;
            let cosine;
            if ray_in.direction.dot(hit_record.normal) > 0.0 {
                outward_normal = -hit_record.normal;
                ni_over_nt = dielectric_material.refraction_index;
                cosine = dielectric_material.refraction_index * ray_in.direction.dot(hit_record.normal) / ray_in.direction.magnitude();
            } else {
                outward_normal = hit_record.normal;
                ni_over_nt = 1.0 / dielectric_material.refraction_index;
                cosine = -ray_in.direction.dot(hit_record.normal) / ray_in.direction.magnitude();
            }

            let mut refracted_direction = Vec3::new(0.0, 0.0, 0.0);
            let mut scattered;
            match refract(ray_in.direction, outward_normal, ni_over_nt) {
                Some(refracted) => { 
                    refracted_direction = refracted;
                    reflect_probability = schlick(cosine, dielectric_material.refraction_index); 
                    },
                None => {
                    reflect_probability = 1.0;
                },
            }

            if rand::random::<f32>() < reflect_probability {
                scattered = Ray::new(hit_record.point, reflected);
            } else {
                scattered = Ray::new(hit_record.point, refracted_direction);
            }

            Some( (attenuation, scattered) )
        }
    }
}

fn background_color(ray: &Ray) -> Vec3 {
    let normalized_dir = ray.direction.normalize();
    let t = 0.5*(normalized_dir.y + 1.0);
    let color = (1.0-t)*Vec3::new(1.0, 1.0, 1.0) + t*Vec3::new(0.5, 0.7, 1.0);
    
    color
}

fn ray_to_sphere(ray: &Ray, sphere: &Sphere, t_min: f32, t_max: f32, hit_record: &mut HitRecord) -> bool {
    let sphere_center = sphere.center;
    let sphere_radius = sphere.radius;

    let oc = ray.origin - sphere_center;
    let a = ray.direction.dot(ray.direction);
    let b = oc.dot(ray.direction);
    let c = oc.dot(oc) - sphere_radius*sphere_radius;
    let discriminant = b*b - a*c;
    
    if discriminant > 0.0 {
        let mut temp = (-b - discriminant.sqrt()) / a;
        if temp < t_max && temp > t_min {
            hit_record.t = temp;
            hit_record.point = ray.point_at_parameter(temp);
            hit_record.normal = (hit_record.point - sphere_center) / sphere_radius;

            return true;
        }
        temp = (-b + discriminant.sqrt()) / a;
        if temp < t_max && temp > t_min {
            hit_record.t = temp;
            hit_record.point = ray.point_at_parameter(temp);
            hit_record.normal = (hit_record.point - sphere_center) / sphere_radius;

            return true;
        }
    }

    return false;
}


fn make_camera_rotation(yaw: f32, pitch: f32) -> Quat {
    Quat::from(Euler::new(Rad(pitch), Rad(yaw), Rad(0.0)))
}

//TODO: Put this functionality in the Camera struct
fn make_camera(yaw: f32, pitch: f32, origin: Vec3) -> Camera {
    let rotator = make_camera_rotation(yaw, pitch);
    let dir = rotator * Vec3::unit_z();

    Camera::new(origin, origin + dir, Vec3::unit_y(), WINDOW_WIDTH, WINDOW_HEIGHT, 90.0)
}

fn main() -> Result<(), String>{
    println!("Hello, world!");

    let mut raytracer = Raytracer::new(WINDOW_WIDTH, WINDOW_HEIGHT);
    raytracer.world.push(Object{material: Material::Diffuse(DiffuseMaterial{albedo: Vec3::new(0.5, 0.5, 0.0)}), shape: Shape::Sphere(Sphere{center: Vec3::new(0.0, 0.0, -1.0), radius: 0.5})});
    raytracer.world.push(Object{material: Material::Diffuse(DiffuseMaterial{albedo: Vec3::new(0.0, 0.8, 0.5)}), shape: Shape::Sphere(Sphere{center: Vec3::new(0.0, -100.5, -1.0), radius: 100.0})});
    raytracer.world.push(Object{material: Material::Metalic(MetalicMaterial{albedo: Vec3::new(0.8, 0.6, 0.2), fuzz: 0.2}), shape: Shape::Sphere(Sphere{center: Vec3::new(1.0, 0.0, -1.0), radius: 0.5})});
    raytracer.world.push(Object{material: Material::Dielectric(DielectricMaterial{refraction_index: 1.5}), shape: Shape::Sphere(Sphere{center: Vec3::new(-1.0, 0.0, -1.0), radius: 0.5})});
    raytracer.world.push(Object{material: Material::Dielectric(DielectricMaterial{refraction_index: 1.5}), shape: Shape::Sphere(Sphere{center: Vec3::new(-1.0, 0.0, -1.0), radius: -0.45})});

    //Vec3::new(-2.0, 2.0, 1.0), Vec3::new(0.0, 0.0, -1.0)
    let mut camera_pos = Vec3::new(-2.0, 2.0, 1.0);
    let mut camera_yaw: f32 = 2.5;
    let mut camera_pitch: f32 = -1.0;
    let mut camera_moved_this_frame = false;
    let mut camera = make_camera(camera_yaw, camera_pitch, camera_pos);

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let mut event_pump = sdl_context.event_pump().unwrap();

    let window = video_subsystem
        .window("Rust Raytracer",
                raytracer.width,
                raytracer.height)
        .position_centered()
        .build()
        .map_err(|e| e.to_string()).unwrap();

    let back_buffer_size: usize = (WINDOW_WIDTH * WINDOW_HEIGHT) as usize;
    let mut back_buffer: Vec<TotalPixel> = Vec::with_capacity(back_buffer_size);
    back_buffer.resize(back_buffer_size, Default::default());

    let mut frame : u32 = 0;
    let mut frame_start_time;
    'running: loop {
        frame_start_time = Instant::now();

        // get the inputs here
        {
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit {..} | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                        break 'running
                    },
                    Event::KeyDown { keycode: Some(key), .. } => {
                        camera_moved_this_frame = true;
                        match key {
                            Keycode::A => {camera_pos += make_camera_rotation(camera_yaw, camera_pitch) * Vec3::new(-1.0, 0.0, 0.0) * 0.1},
                            Keycode::D => {camera_pos += make_camera_rotation(camera_yaw, camera_pitch) * Vec3::new(1.0, 0.0, 0.0) * 0.1},
                            Keycode::W => {camera_pos += make_camera_rotation(camera_yaw, camera_pitch) * Vec3::new(0.0, 0.0, 1.0) * 0.1},
                            Keycode::S => {camera_pos += make_camera_rotation(camera_yaw, camera_pitch) * Vec3::new(1.0, 0.0, -1.0) * 0.1},

                            Keycode::Left => {camera_yaw += -0.1},
                            Keycode::Right => {camera_yaw += 0.1},
                            Keycode::Up => {camera_pitch += 0.1},
                            Keycode::Down => {camera_pitch += -0.1},
                            _ => {},
                        }
                        camera = make_camera(camera_yaw, camera_pitch, camera_pos);
                    },
                    // Event::MouseButtonDown { x, y, mouse_btn: MouseButton::Left, .. } => {
                    //     let x = (x as u32) / SQUARE_SIZE;
                    //     let y = (y as u32) / SQUARE_SIZE;
                    //     match game.get_mut(x as i32, y as i32) {
                    //         Some(square) => {*square = !(*square);},
                    //         None => unreachable!(),
                    //     };
                    // },
                    _ => {}
                }
            }
        }
        if camera_moved_this_frame {
            for pixel in &mut back_buffer {
                pixel.color = Vec3::new(0.0, 0.0, 0.0);
                pixel.amount = 0.0;
            }
            camera_moved_this_frame = false;
        }

        println!("frame {} event loop took: {} sec", frame, frame_start_time.elapsed().as_secs_f64());
        
        let start_render_time = Instant::now();

        let mut surface = window.surface(&event_pump).unwrap();
        // TODO: look into surface.enable_RLE()

        if !surface.must_lock() {
            let pixels = surface.without_lock_mut().unwrap();
            let drawing_start_time = Instant::now();
            raytracer.draw_frame(&camera, &mut back_buffer);
            println!("frame {} raycast drawing time: {} sec", frame, drawing_start_time.elapsed().as_secs_f64());

            let copy_backbuffer_start_time = Instant::now();
            present_to_window(&back_buffer, pixels);
            println!("frame {}, copy backbuffer to frontbuffer took: {} sec", frame, copy_backbuffer_start_time.elapsed().as_secs_f64());
        }

        let present_time_start = Instant::now();
        surface.finish().expect("Updating the screen surface failed!");
        println!("frame {} present time: {} sec", frame, present_time_start.elapsed().as_secs_f64());

        println!("frame {} render+present time: {} sec", frame, start_render_time.elapsed().as_secs_f64());

        let frame_time = frame_start_time.elapsed().as_secs_f64();
        println!("frame {} total time: {} sec. FPS={}", frame, frame_time, 1.0f64/frame_time);

        frame += 1;

        println!("=========================================================");
    }
    
    Ok(())
}
