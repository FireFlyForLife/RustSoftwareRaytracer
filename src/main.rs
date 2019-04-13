#![feature(duration_float)]
#![feature(uniform_paths)]

mod geometry;
mod math_extensions;
mod ray;
mod camera;

use sdl2::EventPump;
use math_extensions::*;
use ray::Ray;
use geometry::*;
use camera::Camera;

use cgmath::prelude::*;

use sdl2::pixels::Color;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use rand::prelude::*;

// use sdl2::rect::{Point, Rect};
// use sdl2::surface::Surface;
// use sdl2::mouse::MouseButton;
// use sdl2::video::{Window, WindowContext};
// use sdl2::render::{Canvas, Texture, TextureCreator};
// use std::time::Duration;
use std::time::Instant;
use std::vec::Vec;
use std::ptr;
use std::mem;

const SAMPLES_PER_PIXEL: u32 = 32;
const WINDOW_WIDTH: u32 = 1280;
const WINDOW_HEIGHT: u32 = 720;

#[derive(Default)]
struct World {
    objects: Vec<Object>
}

struct Raytracer {
    world: World,

    width: u32,
    height: u32,
    aspect_ratio: f32,

    // sdl_context: Sdl,
    // video_subsystem: VideoSubsystem,
    // event_pump: EventPump,
    // window: Window,
}

impl Raytracer {
    pub fn new(width: u32, height: u32) -> Raytracer {
        let aspect_ratio = width as f32 / height as f32;

        Raytracer{ 
            world: Default::default(), 
            width: width, 
            height: height, 
            aspect_ratio: aspect_ratio
        }
    }

    pub fn draw_frame(&self, pixels: &mut [u8]) {
        let aspect_ratio = self.width as f32 / self.height as f32;
        let lower_left_corner = Vec3::new(-1.0 * aspect_ratio, -1.0, -1.0);
        let horizontal = Vec3::new(2.0, 0.0, 0.0) * aspect_ratio;
        let vertical = Vec3::new(0.0, 2.0, 0.0);

        let pixels_color_ptr: *mut Color = unsafe{ mem::transmute( &pixels[0] ) };

        let origin = Vec3::new(0.0, 0.0, 0.0);

        for y in 0..self.height {
            for x in 0..self.width {
                let mut total_color = Vec3::new(0.0, 0.0, 0.0);
                for _sample in 0..SAMPLES_PER_PIXEL {
                    let u = ((x as f32) + rand::random::<f32>()) / (self.width as f32);
                    let v = 1.0 - ((y as f32) + rand::random::<f32>()) / (self.height as f32);
                    let ray = Ray::new(origin, lower_left_corner + u*horizontal + v*vertical);

                    let color = self.trace_ray(&ray);

                    total_color += color_to_vec3(color);
                }
                
                let color = vec3_to_color(total_color / (SAMPLES_PER_PIXEL as f32));
                let fixed_color = Color{r: color.b, g: color.g, b: color.r, a: color.a};
                // let pixelOffset = ((y*self.width + x)*4) as usize;
                unsafe{
                    ptr::write::<Color>(pixels_color_ptr.offset((y * self.width + x) as isize), fixed_color);
                }
                // pixels[pixelOffset  ] = color.b;
                // pixels[pixelOffset+1] = color.g;
                // pixels[pixelOffset+2] = color.r;
                // pixels[pixelOffset+3] = color.a;
            }
        }
    }

    pub fn trace_ray(&self, ray: &Ray) -> Color {
        let mut closest_hit: Option<&Shape> = None;
        let mut t: f32 = 100000.0;
        for object in &self.world.objects {
            match &object.shape {
                Shape::Sphere(sphere) => {
                    let this_t = ray_to_sphere(ray, &sphere.center, sphere.radius);
                    if this_t > 0.0 && this_t < t {
                        t = this_t;
                        closest_hit = Some(&object.shape);
                    }
                },
                Shape::Mesh(mesh) => {},
                _ => {},
            }
        }
        
        if t > 0.0 && closest_hit.is_some() {
            match closest_hit.unwrap() {
                Shape::Sphere(sphere) => {
                    let n = (ray.point_at_parameter(t) - sphere.center).normalize();
                    let vec_color = 0.5*Vec3::new(n.x+1., n.y+1., n.z+1.);
                    return vec3_to_color(vec_color);
                },
                Shape::Mesh(mesh) => {

                }
            }

        }

        background_color(ray)
    }
}


fn background_color(ray: &Ray) -> Color {
    let normalized_dir = ray.direction.normalize();
    let t = 0.5*(normalized_dir.y + 1.0);
    let color = (1.0-t)*Vec3::new(1.0, 1.0, 1.0) + t*Vec3::new(0.5, 0.7, 1.0);
    
    vec3_to_color(color)
}

fn ray_to_sphere(ray: &Ray, sphere_center: &Vec3, sphere_radius: f32) -> f32 {
    let oc = ray.origin - sphere_center;
    let a = ray.direction.dot(ray.direction);
    let b = 2.0 * oc.dot(ray.direction);
    let c = oc.dot(oc) - sphere_radius*sphere_radius;
    let discriminant = b*b - 4.0*a*c;
    
    if discriminant < 0.0 {
        return -1.0;
    } else {
        return (-b - discriminant.sqrt()) / (2.0 * a);
    }
}


fn main() -> Result<(), String>{
    println!("Hello, world!");

    let mut raytracer = Raytracer::new(WINDOW_WIDTH, WINDOW_HEIGHT);
    raytracer.world.objects.push(Object{material: Material{}, shape: Shape::Sphere(Sphere{center: Vec3::new(0.0, 0.0, -1.0), radius: 0.5})});
    raytracer.world.objects.push(Object{material: Material{}, shape: Shape::Sphere(Sphere{center: Vec3::new(0.0, -100.5, -1.0), radius: 100.0})});

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
                    // Event::KeyDown { keycode: Some(Keycode::Space), repeat: false, .. } => {
                    //     game.toggle_state();
                    // },
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
        println!("frame {} event loop took: {} sec", frame, frame_start_time.elapsed().as_float_secs());
        
        let start_render_time = Instant::now();

        let mut surface = window.surface(&event_pump).unwrap();
        // TODO: look into surface.enable_RLE()

        if !surface.must_lock() {
            let pixels = surface.without_lock_mut().unwrap();

            let drawing_start_time = Instant::now();
            raytracer.draw_frame(pixels);
            println!("frame {} raycast drawing time: {} sec", frame, drawing_start_time.elapsed().as_float_secs());
        }

        //surface.update_window().expect("Update didn't properly work");
        let present_time_start = Instant::now();
        surface.finish().expect("Updating the screen surface failed!");
        println!("frame {} present time: {} sec", frame, present_time_start.elapsed().as_float_secs());
        println!("frame {} render+present time: {} sec", frame, start_render_time.elapsed().as_float_secs());

        let frame_time = frame_start_time.elapsed().as_float_secs();
        println!("frame {} total time: {} sec. FPS={}", frame, frame_time, 1.0f64/frame_time);

        frame += 1;

        println!("=========================================================");
    }
    
    Ok(())
}
