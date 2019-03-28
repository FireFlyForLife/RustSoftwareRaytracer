#![feature(duration_float)]


//use cgmath::prelude::*;

use sdl2::rect::{Point, Rect};
use sdl2::pixels::Color;
use sdl2::event::Event;
use sdl2::surface::Surface;
use sdl2::mouse::MouseButton;
use sdl2::keyboard::Keycode;
use sdl2::video::{Window, WindowContext};
use sdl2::render::{Canvas, Texture, TextureCreator};
use std::time::Duration;
use std::time::Instant;


fn draw(pixels: &mut [u8], buffer: &mut [u8], width: u32, height: u32)
{
    for y in 0..height {
        for x in 0..width {
            buffer[((y*width + x)*4) as usize] = ((x) % 255) as u8;
            buffer[((y*width + x)*4+1) as usize] = ((y) % 255) as u8;
            buffer[((y*width + x)*4+2) as usize] = ((x+y) % 255) as u8;
            buffer[((y*width + x)*4+3) as usize] = 255;
        }
    }

    pixels.copy_from_slice(buffer);
}

fn main() -> Result<(), String>{
    println!("Hello, world!");

    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;

    let window_width: u32 = 1280;
    let window_height: u32 = 720;

    // the window is the representation of a window in your operating system,
    // however you can only manipulate properties of that window, like its size, whether it's
    // fullscreen, ... but you cannot change its content without using a Canvas or using the
    // `surface()` method.
    let window = video_subsystem
        .window("Rust Raytracer",
                window_width,
                window_height)
        .position_centered()
        .build()
        .map_err(|e| e.to_string())?;

    // the canvas allows us to both manipulate the property of the window and to change its content
    // via hardware or software rendering. See CanvasBuilder for more info.
    // let mut canvas = window.into_canvas()
    //     .target_texture()
    //     // .present_vsync()
    //     .software()
    //     .build()
    //     .map_err(|e| e.to_string())?;

    // println!("Using SDL_Renderer \"{}\"", canvas.info().name);
    // canvas.set_draw_color(Color::RGB(0, 135, 0));
    // // clears the canvas with the color we set in `set_draw_color`.
    // canvas.clear();
    // // However the canvas has not been updated to the window yet, everything has been processed to
    // // an internal buffer, but if we want our buffer to be displayed on the window, we need to call
    // // `present`. We need to call this everytime we want to render a new frame on the window.
    // canvas.present();

    let mut event_pump = sdl_context.event_pump()?;

    let width = window.surface(&event_pump).unwrap().width();
    let height = window.surface(&event_pump).unwrap().height();

    let mut buffer: Vec<u8> = Vec::new();
    buffer.resize((width*height*4) as usize, 0u8);

    println!("surface pixel format: {:?}", window.surface(&event_pump).unwrap().pixel_format_enum());

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
        

        // canvas.set_draw_color(Color::RGB(0, 135, 0));
        // canvas.clear();
        
        // let pixelBuffer: Vec<Color> = Vec::with_capacity(window_width * window_height);
        // pixelBuffer.resize_with(window_width * window_height, Color::RGBA(0, 0, 135, 255));
        
        // canvas.copy(square_texture,
        //             None,
        //             Rect::new(100 as i32,
        //                         100 as i32,
        //                         250,
        //                         250))?;
        
        // canvas.present();
        let start_render_time = Instant::now();

        let mut surface = window.surface(&event_pump).unwrap();
        // TODO: look into surface.enable_RLE()

        if !surface.must_lock() {


            let pixels = surface.without_lock_mut().unwrap();

            let drawing_start_time = Instant::now();
            draw(pixels, buffer.as_mut_slice(), width, height);
            println!("frame {} raycast drawing time: {} sec", frame, drawing_start_time.elapsed().as_float_secs());
        }

        //surface.update_window().expect("Update didn't properly work");
        surface.finish().expect("Updating the screen surface failed!");
        println!("frame {} render+present time: {} sec", frame, start_render_time.elapsed().as_float_secs());

        let frame_time = frame_start_time.elapsed().as_float_secs();
        println!("frame {} total time: {} sec. FPS={}", frame, frame_time, 1.0f64/frame_time);

        frame += 1;

        println!("=========================================================");
    }
    
    Ok(())
}