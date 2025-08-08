//! --------------------------------------------------------------------------------------
//! 47 - Specular Highlights Blinn Phong
//!
//! Created: July 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Add Blinn-Phong spec to the shaders
//! 
//! --------------------------------------------------------------------------------------

use winit::application::ApplicationHandler;
use winit::event::{WindowEvent, DeviceEvent};
use winit::event::Event;
use winit::event::ElementState::{Pressed, Released};
use winit::keyboard::PhysicalKey::Code;
use winit::keyboard::KeyCode;
use winit::keyboard::ModifiersState;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes};
use std::error::Error;
use std::time::Instant;
use ash::vk;
use imgui::{Context as ImGuiContext, Condition, WindowFlags};
use imgui_winit_support::{WinitPlatform, HiDpiMode};

mod vulkan;
mod graphics;
mod camera;
mod scene;
use crate::vulkan::base::VulkanBase;
use crate::vulkan::imgui_renderer::ImGuiRenderer;
use crate::camera::Camera;
use crate::scene::{Scene, SceneObject, Transform as SceneTransform};
use crate::graphics::gltf_loader::import_gltf;
use crate::graphics::materialmanager::{MaterialManager, MaterialProperties};
use crate::graphics::meshmanager::MeshManager;
use cgmath::{prelude::*};
use cgmath::{Vector1,Vector3,Matrix4};

/// Holds the window and Vulkan backend, orchestrating rendering and events.
struct App {
    window: Option<Window>,
    vulkan_base: Option<VulkanBase>,
    scene: Scene,
    camera: Camera,
    step: f32,
    modifiers: ModifiersState,
    moving_forward:  bool,  // W held
    moving_backward: bool,  // S held
    moving_left:     bool,  // A held
    moving_right:    bool,  // D held
    toggle_locked:   bool,  // prevent rapid toggle repeats
    mouselook_enabled: bool,
    imgui: Option<ImGuiContext>,
    platform: Option<WinitPlatform>,
    imgui_renderer: Option<ImGuiRenderer>,
    material_manager: MaterialManager,
    mesh_manager: MeshManager,
    start_of_frame_time: Instant,
    current_ms_per_frame: f32,
    light_pos: Vector3<f32>,
    light_intensity: Vector1<f32>,
}

impl App {
    /// Simple constructor to initialize the App struct.
    pub fn new() -> Self {
        App {
            window: None,
            vulkan_base: None,
            scene: Scene::new(),
            camera: Camera::new(),
            step: 0.1,
            modifiers: ModifiersState::default(),
            moving_forward: false,
            moving_backward: false,
            moving_left: false,
            moving_right: false,
            toggle_locked: false,
            mouselook_enabled: false,
            imgui: None,
            platform: None,
            imgui_renderer: None,
            material_manager: MaterialManager::new(),
            mesh_manager: MeshManager::new(),
            start_of_frame_time: Instant::now(),
            current_ms_per_frame: 0.0,
            light_pos: Vector3::new(0.5, 0.8, 1.0),
            light_intensity: Vector1::new(1.0),
        }
    }

    /// Starts the application event loop and initializes resources.
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or an error if initialization fails.
    pub fn run(mut self) -> Result<(), Box<dyn Error>> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        Ok(event_loop.run_app(&mut self)?)
    }

    /// Prepares a new ImGui frame, builds UI, and returns a reference to DrawData.
    /// # Arguments
    /// * `ms_per_frame` - Milliseconds per frame for display.
    /// * `platform` - The WinitPlatform for ImGui.
    /// * `imgui` - The ImGui context.
    /// * `window` - The winit Window.
    /// * `show_ms_per_frame` - Whether to display milliseconds per frame.
    /// * `light_pos` - The light position vector.
    /// * `light_intensity` - The light intensity vector.
    /// # Returns
    /// * `&'a imgui::DrawData` - The ImGui draw data.
    fn prepare_imgui_draw_data<'a>(
        ms_per_frame: f32,
        platform: &'a mut WinitPlatform,
        imgui: &'a mut ImGuiContext,
        window: &Window,
        show_ms_per_frame: bool,
        light_pos: &mut Vector3<f32>,
        light_intensity: &mut Vector1<f32>
    ) -> &'a imgui::DrawData {
        // Let winit-platform prepare ImGui for a new frame
        platform
            .prepare_frame(imgui.io_mut(), window)
            .expect("Failed to prepare imgui frame");

        // Build ImGui UI and collect draw data
        let ui = imgui.frame();
        if show_ms_per_frame {
        ui.window("##ms_per_redraw")
            .position([10.0, 10.0], Condition::Always)
            .size([200.0, 30.0], Condition::Always)
            .flags(
                WindowFlags::NO_TITLE_BAR
            | WindowFlags::NO_RESIZE
            | WindowFlags::NO_MOVE
            | WindowFlags::NO_SCROLLBAR
            | WindowFlags::NO_BACKGROUND
            )
            .build(|| {
                ui.text(format!("Redraw ms: {:.2}", ms_per_frame));
            });
        }
        ui.window("Controls")
            .size([300.0, 180.0], Condition::FirstUseEver)
            .build(|| {
                ui.text("Light Position");
                ui.slider("X", -100.0, 100.0, &mut light_pos[0]);
                ui.slider("Y", -100.0, 100.0, &mut light_pos[1]);
                ui.slider("Z", -100.0, 100.0, &mut light_pos[2]);
                ui.text("Light Intensity");
                ui.slider("LI", 0.0, 20.0, &mut light_intensity[0]);
            });
        // Prepare the Vulkan render pass for ImGui
        platform.prepare_render(ui, window);
        // Return the collected draw lists
        imgui.render()
    }

    /// Computes the push constant data for each object in the scene.
    /// # Arguments
    /// * `camera` - The camera used for the scene, providing view and projection matrices.
    /// * `model_matrix` - The model matrix of the object being drawn.
    /// * `light_pos` - The position of the light.
    /// * `light_intensity` - The intensity of the light.
    /// # Returns
    /// * `Vec<u8>` - The serialized push constant data containing the MVP matrix and light direction.
    fn compute_push_constant_per_obj(camera: &Camera, model_matrix: &Matrix4<f32>, light_pos: Vector3<f32>, light_intensity: Vector1<f32>) -> Vec<u8> {
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

        // 6) Transform world-space light position into view-space
        // This saves doing that multiplication in every run of the vertex shader
        let light_pos = view.transform_vector(light_pos);
        // Append the three components of light_view
        bytes.extend_from_slice(&light_pos.x.to_ne_bytes());
        bytes.extend_from_slice(&light_pos.y.to_ne_bytes());
        bytes.extend_from_slice(&light_pos.z.to_ne_bytes());

        // Finally add the light intensity float
        bytes.extend_from_slice(&light_intensity[0].to_ne_bytes());

        bytes
    }

    /// Creates a new window with the specified title and default attributes.
    /// # Arguments
    /// * `event_loop` - The active event loop to manage window and Vulkan events.
    /// # Returns
    /// * A new window instance.
    fn create_window(&mut self, event_loop: &ActiveEventLoop) -> Window {
        let window_attributes = WindowAttributes::default().with_title("Rust Vulkan 47 - Specular Highlights Blinn Phong");
        let window = event_loop
            .create_window(window_attributes)
            .expect("Failed to create window");
        println!("🪟 Window created");

        window.set_cursor_grab(winit::window::CursorGrabMode::None).ok();
        window.set_cursor_visible(false);
        window
    }

    /// Sets up the initial assets and scene for the application
    /// # Arguments
    /// * `event_loop` - The active event loop to manage window and Vulkan events.
    fn set_up_scene(&mut self) {
        let vulkan_base = self.vulkan_base.as_mut().unwrap();

        let infinite_plane = SceneObject {
            transform: SceneTransform::identity(),
            material_id: self.material_manager.request_material(
                vulkan_base,
                MaterialProperties {
                    name: "InfinitePlaneMaterial".into(),
                    vs_path: "assets/shaders/spv/infinite_plane.vert.spv".into(),
                    fs_path: "assets/shaders/spv/infinite_plane.frag.spv".into(),
                    diffuse_texture_path: None,
                    normalmap_texture_path: None,
                    depth_write: false }
            ),
            mesh_id: self.mesh_manager.request_unit_plane(
                vulkan_base,
            ).expect("Failed to load unit plane mesh"),
        };

        let cube = SceneObject {
            transform: SceneTransform::from_euler(
                cgmath::Vector3::new(0.0, 1.0, 0.0),
                cgmath::Vector3::new(0.0, 0.0, 0.0),
                1.0,
            ),
            material_id: self.material_manager.request_material(
                vulkan_base,
                MaterialProperties {
                    name: "Cube1Material".into(),
                    vs_path: "assets/shaders/spv/point_light.vert.spv".into(),
                    fs_path: "assets/shaders/spv/point_light.frag.spv".into(),
                    diffuse_texture_path: Some("assets/textures/cube1/diffuse.png".into()),
                    normalmap_texture_path: Some("assets/textures/cube1/normal.png".into()),
                    depth_write: true,
                }
            ),
            mesh_id: self.mesh_manager.request_cube(
                vulkan_base,
            ).expect("Failed to load cube mesh"),
        };

        // Import duck mesh + material via glTF loader
        let prims = import_gltf(
            "assets/meshes/duck.gltf",
            vulkan_base,
            &mut self.mesh_manager,
            &mut self.material_manager,
        ).expect("Failed to import glTF mesh");
        let duck_prim = &prims[0];

        let duck = SceneObject {
            transform: SceneTransform::from_euler(
                cgmath::Vector3::new(3.0, 0.0, 0.0),
                cgmath::Vector3::new(0.0, -90.0, 0.0),
                0.015,
            ),
            material_id: duck_prim.mat_id,
            mesh_id:     duck_prim.mesh_id,
        };

        // Import duck mesh + material via glTF loader
        let prims2 = import_gltf(
            "assets/meshes/sphere.gltf",
            vulkan_base,
            &mut self.mesh_manager,
            &mut self.material_manager,
        ).expect("Failed to import glTF mesh");
        let sphere_prim = &prims2[0];

        let sphere = SceneObject {
            transform: SceneTransform::from_euler(
                cgmath::Vector3::new(-3.0, 1.0, 0.0),
                cgmath::Vector3::new(0.0, 0.0, 0.0),
                1.0,
            ),
            material_id: sphere_prim.mat_id,
            mesh_id:     sphere_prim.mesh_id,
        };

        self.scene = Scene::new();
        self.scene.add(cube);
        self.scene.add(duck);
        self.scene.add(sphere);
        self.scene.add(infinite_plane);

        self.camera = Camera::new();
        self.step = 0.1;
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
        self.window = Some(self.create_window(event_loop));

        // — ImGui Platformn setup —
        let mut imgui = ImGuiContext::create();
        let mut platform = WinitPlatform::new(&mut imgui);
        platform.attach_window(imgui.io_mut(), self.window.as_ref().unwrap(), HiDpiMode::Rounded);
        self.imgui = Some(imgui);
        self.platform = Some(platform);

        // Initialize VulkanBase and then upload the atlas
        match VulkanBase::new(self.window.as_ref().unwrap(), event_loop) {
            Ok(vulkan_base) => {
                self.vulkan_base = Some(vulkan_base);
            }
            Err(e) => {
                eprintln!("Failed to create VulkanBase: {}", e);
                event_loop.exit();
            }
        }

        let material_manager = MaterialManager::new();
        self.material_manager = material_manager;

        // Create the imgui renderer
        let renderer = ImGuiRenderer::new(self.vulkan_base.as_mut().unwrap(), self.imgui.as_mut().unwrap());
        self.imgui_renderer = Some(renderer);

        // Now set up the scene
        self.set_up_scene();
    }

    /// Called for *all* raw device events (mouse, keyboard, etc).
    /// # Arguments
    /// * `_event_loop` - The active event loop.
    /// * `_device_id` - The ID of the device that sent the event.
    /// * `event` - The device event.
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.mouselook_enabled {
                let (dx, dy) = (delta.0 as f32, delta.1 as f32);
                let sensitivity = 0.1;
                // convert to yaw/pitch deltas and feed into camera:
                self.camera.rotate(
                    dx * sensitivity,
                    -dy * sensitivity,
                );
            }
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
            // Forward winit event to ImGui for mouse/keyboard handling
            if let (Some(platform), Some(imgui)) = (self.platform.as_mut(), self.imgui.as_mut()) {
                // Wrap the WindowEvent into a full Event for ImGui, with explicit type parameter
                let full_event: Event<()> = Event::WindowEvent {
                    window_id: win.id(),
                    event: event.clone(),
                };
                platform.handle_event(imgui.io_mut(), win, &full_event);
            }
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
                            if let Err(e) = vulkan_base.recreate_swapchain(self.window.as_ref().unwrap()) {
                                eprintln!("Failed to recreate swapchain: {}", e);
                            }
                            if let Err(e) = self.material_manager.recreate_pipelines(vulkan_base) {
                                eprintln!("Failed to recreate material pipelines: {}", e);
                            };
                            if let Err(e) = vulkan_base.record_command_buffers() {
                                eprintln!("Failed to record command buffers: {}", e);
                            }
                            if let Err(e) = self.imgui_renderer.as_mut().unwrap().rebuild_pipeline(vulkan_base) {
                                eprintln!("Failed to rebuild imgui pipeline: {}", e);
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
                            (Code(KeyCode::KeyF), Pressed) if !self.modifiers.control_key() => {
                                let fullscreen = if win.fullscreen().is_some() {
                                    None
                                } else {
                                    Some(winit::window::Fullscreen::Borderless(None))
                                };
                                win.set_fullscreen(fullscreen);
                                println!("🖥️ Toggled fullscreen");
                            }
                            (Code(KeyCode::KeyF), Released) => {
                                self.toggle_locked = false;
                            }
                            // — Movement keys —
                            (Code(KeyCode::KeyW), Pressed) if !self.modifiers.control_key() => {
                                self.moving_forward = true;
                            }
                            (Code(KeyCode::KeyW), Released) => {
                                self.moving_forward = false;
                                // also unlock Ctrl+W toggle when released
                                self.toggle_locked = false;
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
                            (Code(KeyCode::KeyM), Pressed) if !self.toggle_locked => {
                                self.toggle_locked = true;
                                self.mouselook_enabled = !self.mouselook_enabled;
                            }
                            (Code(KeyCode::KeyM), Released) => {
                                self.toggle_locked = false;
                            }
                            // — Wireframe toggle on Ctrl+W, one shot —
                            (Code(KeyCode::KeyW), Pressed) if self.modifiers.control_key() && !self.toggle_locked => {
                                self.toggle_locked = true;
                                if let Some(vb) = &mut self.vulkan_base {
                                    vb.toggle_wireframe();
                                    println!("🔲 Wireframe mode toggled");
                                    if let Err(e) = self.material_manager.recreate_pipelines(vb) {
                                        eprintln!("Failed to recreate material pipelines: {}", e);
                                    };
                                    if let Err(e) = vb.record_command_buffers() {
                                        eprintln!("Failed to record command buffers: {}", e);
                                    }
                                }
                            }
                            // — ms per frame toggle on Ctrl+F, one shot —
                            (Code(KeyCode::KeyF), Pressed) if self.modifiers.control_key() && !self.toggle_locked => {
                                self.toggle_locked = true;
                                if let Some(vb) = &mut self.vulkan_base {
                                    vb.toggle_ms_per_frame();
                                }
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        self.start_of_frame_time = Instant::now();
                        // First check everything we need is built
                        if let Some(vb) = self.vulkan_base.as_mut() {
                            // Borrow VulkanBase, ImGui context, renderer, and platform for this frame
                            let imgui_renderer = self.imgui_renderer.as_mut().unwrap();
                            // Get a mutable reference to the scene for local use
                            let scene = &mut self.scene;

                            // Now do movement
                            let s = self.step;
                            if self.moving_forward  { self.camera.translate( s,  0.0) }
                            if self.moving_backward { self.camera.translate(-s,  0.0) }
                            if self.moving_left     { self.camera.translate( 0.0, -s) }
                            if self.moving_right    { self.camera.translate( 0.0,  s) }

                            // Prepare ImGui UI and get draw data
                            let window = self.window.as_ref().unwrap();
                            let draw_data = App::prepare_imgui_draw_data(
                                self.current_ms_per_frame,
                                self.platform.as_mut().unwrap(),
                                self.imgui.as_mut().unwrap(),
                                window,
                                vb.debug_settings.show_ms_per_frame,
                                &mut self.light_pos,
                                &mut self.light_intensity
                            );

                            if let Err(e) = vb.draw_frame({
                                |base, cmd_buf| {
                                    let device = &base.device;
                                    let mut current_pipeline_id = usize::MAX;
                                    for obj in &scene.objects {
                                        // Update the transform for each object
                                        let model_matrix = obj.transform.model_matrix();
                                        // Push the model matrix as a push constant
                                        let push_bytes = Self::compute_push_constant_per_obj(&self.camera, &model_matrix, self.light_pos, self.light_intensity);
                                        unsafe {
                                            if current_pipeline_id != obj.material_id {
                                                device.cmd_bind_pipeline(
                                                    cmd_buf,
                                                    vk::PipelineBindPoint::GRAPHICS,
                                                    self.material_manager.materials[obj.material_id].pipeline.vk_pipeline
                                                );
                                                if self.material_manager.materials[obj.material_id].textures.is_some() {
                                                    device.cmd_bind_descriptor_sets(
                                                        cmd_buf,
                                                        vk::PipelineBindPoint::GRAPHICS,
                                                        self.material_manager.materials[obj.material_id].pipeline.vk_layout,
                                                        0, // set index
                                                        &[self.material_manager.materials[obj.material_id].texture_descriptor_set],
                                                        &[],
                                                    );
                                                }
                                                current_pipeline_id = obj.material_id;
                                            };
                                            device.cmd_push_constants(
                                                cmd_buf,
                                                self.material_manager.materials[obj.material_id].pipeline.vk_layout,
                                                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                                0,
                                                &push_bytes,
                                            );
                                        }
                                        self.mesh_manager.meshes[obj.mesh_id].record(device, cmd_buf);
                                    }
                                    // Bind and draw ImGui on top
                                    unsafe {
                                        device.cmd_bind_pipeline(
                                            cmd_buf,
                                            vk::PipelineBindPoint::GRAPHICS,
                                            imgui_renderer.vk_pipeline);
                                    }
                                    imgui_renderer.render(device, cmd_buf, draw_data);
                                }
                            }) {
                                eprintln!("Failed to draw frame: {}", e);
                            }
                            let now = Instant::now();
                            let elapsed = now.duration_since(self.start_of_frame_time);
                            self.current_ms_per_frame = elapsed.as_millis() as f32;
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

            self.material_manager.cleanup(&vb.device);
            self.mesh_manager.cleanup(&vb.device);

            if let Some(renderer) = self.imgui_renderer.take() {
                println!("🗑️ Cleaning up ImGui renderer");
                renderer.cleanup(vb);
            }
        }
    }
}

/// Program entry point: initializes and runs the App, logging errors.
fn main() {
    if let Err(e) = App::new().run() {
        eprintln!("Application error: {}", e);
    }
}