//! --------------------------------------------------------------------------------------
//! 32 - Start setting up for multiple objects
//!
//! Created: July 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! At the moment things are pretty hard coded for a single cube.  We're going to add
//! Scene and SceneObject structs to hold multiple objects, and then we'll update the
//! rendering code to draw all objects in the scene.  For now, the SceneObject will only
//! hold the transform data.  The code here creates two cubes, one at the origin
//! and one at (3,0,0), and renders them both.  We do sadly lose the spinning...
//! 
//! --------------------------------------------------------------------------------------

use winit::application::ApplicationHandler;
use winit::event::{WindowEvent, DeviceEvent};
use winit::event::ElementState::{Pressed, Released};
use winit::keyboard::PhysicalKey::Code;
use winit::keyboard::KeyCode;
use winit::keyboard::ModifiersState;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes};
use std::error::Error;
use ash::vk;

mod vulkan;
mod graphics;
mod assets; 
mod camera;
mod scene;
use crate::vulkan::base::VulkanBase;
use crate::assets::{ProjectAssets, LoadedShaders, LoadedMeshes};
use crate::camera::Camera;
use crate::scene::{Scene, SceneObject, Transform};
use cgmath::prelude::*;
use cgmath::Matrix4;

/// Holds the window and Vulkan backend, orchestrating rendering and events.
#[derive(Default)]
struct App {
    window: Option<Window>,
    vulkan_base: Option<VulkanBase>,
    scene: Scene,
    meshes: Option<LoadedMeshes>,
    shaders: Option<LoadedShaders>,
    camera: Camera,
    step: f32,
    modifiers: ModifiersState,
    moving_forward:  bool,  // W held
    moving_backward: bool,  // S held
    moving_left:     bool,  // A held
    moving_right:    bool,  // D held
    w_toggle_locked: bool,  // prevent rapid Ctrl+W repeats
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
        let window_attributes = WindowAttributes::default().with_title("Rust Vulkan 32 - Scene");
        let window = event_loop
            .create_window(window_attributes)
            .expect("Failed to create window");
        println!("🪟 Window created");
        self.window = Some(window);

        let win = self.window.as_ref().unwrap();
        win.set_cursor_grab(winit::window::CursorGrabMode::None).ok();
        win.set_cursor_visible(false);

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
        let cube = SceneObject {
            transform: Transform::identity(), 
        };
        let cube2 = SceneObject {
            transform: Transform {
                translation: cgmath::Vector3::new(3.0, 0.0, 0.0),
                rotation: cgmath::Quaternion::new(0.0, 0.0, 0.0, 1.0),
                scale: 1.0,
            },
        };

        self.scene = Scene::new();
        self.scene.add(cube);
        self.scene.add(cube2);
        self.camera = Camera::new();
        self.step = 0.1;
    }

    /// Called for *all* raw device events (mouse, keyboard, etc).
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            let (dx, dy) = (delta.0 as f32, delta.1 as f32);
            let sensitivity = 0.1;
            // convert to yaw/pitch deltas and feed into camera:
            self.camera.rotate(
                dx * sensitivity,
                -dy * sensitivity,
            );
        }
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
                        // Update camera’s projection
                        if let Some(window) = &self.window {
                            let size   = window.inner_size();
                            let aspect = size.width as f32 / size.height as f32;
                            self.camera.set_perspective_projection(45.0, aspect, 0.1, 100.0);
                        }
                    }
                    WindowEvent::ModifiersChanged(mods) => {
                        self.modifiers = mods.state();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        let key = event.physical_key;
                        match (key, event.state) {
                            // — Exit —
                            (Code(KeyCode::Escape), Pressed) => {
                                println!("🛑 Escape pressed, exiting");
                                event_loop.exit();
                            }

                            // — Fullscreen toggle on F —
                            (Code(KeyCode::KeyF), Pressed) => {
                                let fullscreen = if win.fullscreen().is_some() {
                                    None
                                } else {
                                    Some(winit::window::Fullscreen::Borderless(None))
                                };
                                win.set_fullscreen(fullscreen);
                                println!("🖥️ Toggled fullscreen");
                            }

                            // — Movement keys —
                            (Code(KeyCode::KeyW), Pressed) if !self.modifiers.control_key() => {
                                self.moving_forward = true;
                            }
                            (Code(KeyCode::KeyW), Released) => {
                                self.moving_forward = false;
                                // also unlock Ctrl+W toggle when released
                                self.w_toggle_locked = false;
                            }
                            (Code(KeyCode::KeyS), Pressed) => {
                                self.moving_backward = true;
                            }
                            (Code(KeyCode::KeyS), Released) => {
                                self.moving_backward = false;
                            }
                            (Code(KeyCode::KeyA), Pressed) => {
                                self.moving_left = true;
                            }
                            (Code(KeyCode::KeyA), Released) => {
                                self.moving_left = false;
                            }
                            (Code(KeyCode::KeyD), Pressed) => {
                                self.moving_right = true;
                            }
                            (Code(KeyCode::KeyD), Released) => {
                                self.moving_right = false;
                            }
                            // — Wireframe toggle on Ctrl+W, one shot —
                            (Code(KeyCode::KeyW), Pressed) if self.modifiers.control_key() && !self.w_toggle_locked => {
                                self.w_toggle_locked = true;
                                if let Some(vb) = &mut self.vulkan_base {
                                    if let Some(shaders) = &self.shaders {
                                        vb.toggle_wireframe();
                                        println!("🔲 Wireframe mode toggled");
                                        let stages = [&shaders.vertex, &shaders.fragment];
                                        if let Err(e) = vb.recreate_pipeline_and_record(&stages) {
                                            eprintln!("Failed to recreate pipeline: {}", e);
                                        }
                                    }
                                }
                            }

                            _ => {}
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        if let (Some(vb), Some(meshes), Some(scene)) = (self.vulkan_base.as_mut(), self.meshes.as_ref(), Some(&self.scene)) {
                            // Calculate these outside the closure to avoid borrowing
                            let layout = vb.pipeline.layout;

                            let s = self.step;
                            if self.moving_forward  { self.camera.translate( s,  0.0) }
                            if self.moving_backward { self.camera.translate(-s,  0.0) }
                            if self.moving_left     { self.camera.translate( 0.0, -s) }
                            if self.moving_right    { self.camera.translate( 0.0,  s) }

                            let camera = &self.camera;

                            // Draw the frame with the updated MVP matrix
                            if let Err(e) = vb.draw_frame(move |cmd_buf, device| {
                                for obj in &scene.objects {
                                    // Update the transform for each object
                                    let model_matrix = obj.transform.model_matrix();
                                    // Push the model matrix as a push constant
                                    let push_bytes = compute_push_constant_per_obj(&camera, &model_matrix);
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
                                }
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

/// Computes the push constant data for each object in the scene.
/// # Arguments
/// * `camera` - The camera used for the scene, providing view and projection matrices.
/// * `model_matrix` - The model matrix of the object being drawn.
/// # Returns
/// * `Vec<u8>` - The serialized push constant data containing the MVP matrix and light direction.
fn compute_push_constant_per_obj(camera: &Camera, model_matrix: &Matrix4<f32>) -> Vec<u8> {

    let proj: Matrix4<f32> = *camera.get_projection();
    let view: Matrix4<f32> = *camera.get_view();

    // 4) Compute both MV and full MVP
    let mv  = view * model_matrix;    // Model‐View matrix for normals
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
    for v in light_dir {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }

    bytes
}