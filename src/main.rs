//! --------------------------------------------------------------------------------------
//! 30 - Lambert Lighting
//!
//! Created: July 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Passes through vertex normals and modified the vertex and fragment shaders to implement
//! Lambert lighting on the cube.
//! 
//! --------------------------------------------------------------------------------------

use winit::application::ApplicationHandler;
use winit::event::{WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes};
use std::error::Error;
use ash::vk;

mod vulkan;
mod graphics;
mod assets;
use crate::vulkan::base::VulkanBase;
use crate::assets::{ProjectAssets, LoadedShaders, LoadedMeshes};
use cgmath::prelude::*;
use cgmath::{Deg, Matrix4, Point3, Vector3, perspective};

/// Holds the window and Vulkan backend, orchestrating rendering and events.
#[derive(Default)]
struct App {
    window: Option<Window>,
    vulkan_base: Option<VulkanBase>,
    meshes: Option<LoadedMeshes>,
    shaders: Option<LoadedShaders>,
    angle: f32,
}

impl App {
    /// Starts the application event loop and initializes resources.
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or an error if initialization fails.
    pub fn run(mut self) -> Result<(), Box<dyn Error>> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        Ok(event_loop.run_app(&mut self)?)
    }
}

impl ApplicationHandler for App {
    /// Called when the application resumes; creates window and sets up Vulkan.
    /// # Arguments
    /// * `event_loop` - The active event loop to manage window and Vulkan events.
    /// # Notes
    /// * Initializes the window with a title and default attributes.
    /// * Creates a new `VulkanBase` instance, handling errors gracefully.
    /// * If Vulkan initialization fails, the event loop exits to prevent further errors.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = WindowAttributes::default().with_title("Rust Vulkan 30 - Lambert Lighting");
        let window = event_loop
            .create_window(window_attributes)
            .expect("Failed to create window");
        println!("🪟 Window created");
        self.window = Some(window);

        match VulkanBase::new(self.window.as_ref().unwrap(), event_loop) {
            Ok(vulkan_base) => self.vulkan_base = Some(vulkan_base),
            Err(e) => {
                eprintln!("Failed to create VulkanBase: {}", e);
                event_loop.exit();
            }
        }

        let vulkan_base_ref = self.vulkan_base.as_mut().unwrap();
        let assets = ProjectAssets::new();
        let vertices = match LoadedMeshes::load(
            &vulkan_base_ref.instance,
            &vulkan_base_ref.device,
            vulkan_base_ref.physical_device,
            &assets) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to load meshes: {}", e);
                event_loop.exit();
                return;
            }
        };
        self.meshes = Some(vertices);
        let shaders = match LoadedShaders::load(&vulkan_base_ref.device, &assets) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to load shaders: {}", e);
                event_loop.exit();
                return;
            }
        };
        self.shaders = Some(shaders);
        match vulkan_base_ref.recreate_pipeline_and_record(
            &[ &self.shaders.as_ref().unwrap().vertex,
               &self.shaders.as_ref().unwrap().fragment ])
        {
            Ok(_) => {},
            Err(e) => {
                eprintln!("Failed to recreate pipeline: {}", e);
                event_loop.exit();
                return;
            }
        };
    }

    /// Handles window events like closing, resizing, input, and redraw.
    /// # Arguments
    /// * `event_loop` - The active event loop managing the application.
    /// * `window_id` - The ID of the window that received the event.
    /// * `event` - The specific window event that occurred.
    /// # Notes
    /// * Closes the application on close requests.
    /// * Resizes the Vulkan swapchain on window resize events.
    /// * Toggles fullscreen mode when the 'F' key is pressed.
    /// * Requests a redraw on `RedrawRequested` events.
    /// * Handles errors during Vulkan operations gracefully.
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let Some(win) = &self.window {
            if window_id == win.id() {
                match event {
                    WindowEvent::CloseRequested => {
                        println!("Close requested!");
                        event_loop.exit();
                    }
                    WindowEvent::Resized(_)
                    | WindowEvent::ScaleFactorChanged { .. } => {
                        println!("📐 Window resized");
                        if let Some(vulkan_base) = &mut self.vulkan_base {
                            let stages = [
                                &self.shaders.as_ref().unwrap().vertex,
                                &self.shaders.as_ref().unwrap().fragment
                            ];
                            if let Err(e) = vulkan_base.recreate_swapchain(self.window.as_ref().unwrap(), &stages) {
                                eprintln!("Failed to recreate swapchain: {}", e);
                            }
                        }
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        match event.physical_key {
                            winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) => {
                                println!("🛑 Escape pressed, exiting");
                                event_loop.exit();
                            }
                            winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyF) => {
                                if event.state.is_pressed() {
                                    let fullscreen = if win.fullscreen().is_some() {
                                        None
                                    } else {
                                        Some(winit::window::Fullscreen::Borderless(None))
                                    };
                                    win.set_fullscreen(fullscreen);
                                    println!("🖥️ Toggled fullscreen");
                                }
                            }
                            winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyW) => {
                                if event.state.is_pressed() {
                                    if let Some(vulkan_base) = &mut self.vulkan_base {
                                        if let Some(shaders) = &self.shaders {
                                            vulkan_base.toggle_wireframe();
                                            println!("🔲 Wireframe mode toggled");
                                            if let Err(e) = vulkan_base.recreate_pipeline_and_record(&[ 
                                                &shaders.vertex, &shaders.fragment ]) {
                                                eprintln!("Failed to recreate pipeline in wireframe mode: {}", e);
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        // Update rotation angle
                        // I only reset the angle every 10 full rotations because some of the matrix ops below are angle * some value
                        // less than 1 which means that they'll reset and jump if that value is too small
                        // I then also add a light angle to the byte array that will be pushed to the vertex and fragment shaders
                        self.angle = (self.angle + 1.0) % 3600.0;

                        if let (Some(vb), Some(meshes)) = (self.vulkan_base.as_mut(), self.meshes.as_ref()) {
                            // Calculate these outside the closure to avoid borrowing
                            let layout = vb.pipeline.layout;
                            let push_bytes = compute_push_constant(self.angle, vb.swapchain.extent);

                            // Draw the frame with the updated MVP matrix
                            if let Err(e) = vb.draw_frame(move |cmd_buf, device| {
                                unsafe {
                                    device.cmd_push_constants(
                                        cmd_buf,
                                        layout,
                                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                        0,
                                        &push_bytes,
                                    );
                                }
                                meshes.record(device, cmd_buf);
                            }) {
                                eprintln!("Failed to draw frame: {}", e);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    
    /// Called before waiting for new events; requests a redraw each loop iteration.
    /// # Arguments
    /// * `event_loop` - The active event loop managing the application.
    /// # Notes
    /// * Requests a redraw of the window to ensure the latest frame is displayed.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // 1) Pull the VulkanBase out so we can take ownership
        if let Some(vb) = self.vulkan_base.take() {
            unsafe { let _ = vb.device.device_wait_idle(); }

            if let Some(meshes) = self.meshes.take() {
                println!("🗑️ Cleaning up meshes");
                meshes.cleanup(&vb.device);
            }

            if let Some(shaders) = self.shaders.take() {
                println!("🗑️ Cleaning up shaders");
                shaders.cleanup();
            }
        }
    }
}

/// Program entry point: initializes and runs the App, logging errors.
fn main() {
    if let Err(e) = App::default().run() {
        eprintln!("Application error: {}", e);
    }
}

/// Builds the push‐constant block (MVP, MV, lightDir),
/// flattened as column‐major bytes:
/// 16 floats for MVP, then 16 floats for MV, then 3 floats for lightDir.
fn compute_push_constant(angle_deg: f32, extent: vk::Extent2D) -> Vec<u8> {
    // 1) Projection: 45° FOV, aspect, near=0.1, far=100
    let aspect = extent.width as f32 / extent.height as f32;
    let mut proj: Matrix4<f32> = perspective(Deg(45.0), aspect, 0.1, 100.0);
    // Vulkan’s NDC has Y flipped, so invert Y
    proj.y.y *= -1.0;

    // 2) View: camera at (0,0,5) looking at origin
    let view = Matrix4::look_at_rh(
        Point3::new(0.0, 0.0, 5.0),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::unit_y(),
    );

    // 3) Model: your rolling‐cube rotations
    let rot_z = Matrix4::from_angle_z(Deg(angle_deg));
    let rot_x = Matrix4::from_angle_x(Deg(angle_deg * 0.5));
    let rot_y = Matrix4::from_angle_y(Deg(angle_deg * -0.25));
    let model = rot_y * (rot_z * (rot_x * rot_y));

    // 4) Compute both MV and full MVP
    let mv  = view * model;    // Model‐View matrix for normals
    let mvp = proj * mv;       // Projection × View × Model

    // 5) Flatten MVP (4×4) and MV (4×4) into column‐major bytes,
    //    then append lightDir (3 floats).
    let mut bytes = Vec::with_capacity((16 + 16 + 3) * 4);
    // Flatten a 4×4 in column-major by transposing then iterating row/col
    let flatten_mat4 = |m: Matrix4<f32>, buf: &mut Vec<u8>| {
        let cols = m.transpose();
        for row in 0..4 {
            for col in 0..4 {
                buf.extend_from_slice(&cols[col][row].to_ne_bytes());
            }
        }
    };

    // MVP first
    flatten_mat4(mvp, &mut bytes);
    // then MV
    flatten_mat4(mv,  &mut bytes);

    // 6) Append your light direction (e.g. from above‐right)
    let light_dir = [1.0f32, 1.0, 1.0];
    for &v in &light_dir {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }

    bytes
}