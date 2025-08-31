//! --------------------------------------------------------------------------------------
//! Application Core (app.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Main application logic that wires together windowing, rendering, and state management.
//!
//! --------------------------------------------------------------------------------------

use std::error::Error;
use std::time::Instant;
use std::sync::{Arc, atomic::AtomicBool};

use imgui::Context as ImGuiContext;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::ModifiersState;
use winit::window::{Window, WindowAttributes};
use winit::dpi::LogicalSize;

use crate::graphics::camera::Camera;
use crate::graphics::gltf_loader::import_gltf;
use crate::graphics::materialmanager::{MaterialManager, MaterialProperties};
use crate::graphics::meshmanager::MeshManager;
use crate::app::scene::{Scene, SceneObject, ScenePart, Transform as SceneTransform};
use crate::vulkan::base::VulkanBase;
use crate::vulkan::imgui_renderer::ImGuiRenderer;
use crate::vulkan::render_graph::{RenderGraph, RenderPassNode};

use super::input;
use super::input::InputState;

/// Top-level application state that wires windowing, rendering, and scene.
///
/// Owns the window, Vulkan backend, scene graph, camera, UI, and managers
/// for meshes and materials. Implements `ApplicationHandler` to drive the
/// event loop.
pub struct App {
    pub window: Option<Window>,
    pub vulkan_base: Option<VulkanBase>,
    pub scene: Scene,
    pub camera: Camera,
    pub step: f32,
    pub modifiers: ModifiersState,
    pub input: InputState,
    pub imgui: Option<ImGuiContext>,
    pub platform: Option<WinitPlatform>,
    pub imgui_renderer: Option<ImGuiRenderer>,
    pub material_manager: MaterialManager,
    pub mesh_manager: MeshManager,
    pub start_of_frame_time: Instant,
    pub current_ms_per_frame: f32,
    pub world_controls: WorldControls,
    pub render_graph: RenderGraph,
    pub exit_flag: Arc<AtomicBool>,
    // Ground switching
    pub ground_obj_index: Option<usize>,
    pub infinite_plane_material_id: usize,
    pub sand_plane_material_id: usize,
}

impl App {
    /// Constructs a new `App` with default state. Call `run()` to start.
    pub fn new() -> Self {
        App {
            window: None,
            vulkan_base: None,
            scene: Scene::new(),
            camera: Camera::new(),
            step: 0.1,
            modifiers: ModifiersState::default(),
            input: InputState::default(),
            imgui: None,
            platform: None,
            imgui_renderer: None,
            material_manager: MaterialManager::new(),
            mesh_manager: MeshManager::new(),
            start_of_frame_time: Instant::now(),
            current_ms_per_frame: 0.0,
            world_controls: WorldControls::default(),
            render_graph: RenderGraph::new(),
            exit_flag: Arc::new(AtomicBool::new(false)),
            ground_obj_index: None,
            infinite_plane_material_id: 0,
            sand_plane_material_id: 0,
        }
    }

    /// Runs the winit event loop until exit.
    pub fn run(mut self) -> Result<(), Box<dyn Error>> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        Ok(event_loop.run_app(&mut self)?)
    }

    fn create_window(&mut self, event_loop: &ActiveEventLoop) -> Window {
        let window_attributes = WindowAttributes::default()
    .with_title("Rusty Vulkan")
    .with_inner_size(LogicalSize::new(1280, 720));
        let window = event_loop
            .create_window(window_attributes)
            .expect("Failed to create window");
        println!("🪟 Window created");

        window
            .set_cursor_grab(winit::window::CursorGrabMode::None)
            .ok();
        window.set_cursor_visible(false);
        window
    }

    fn set_up_scene(&mut self) {
        let vulkan_base = self.vulkan_base.as_mut().unwrap();
        // Create both materials needed for ground
        let infinite_plane_mat_id = self.material_manager.request_material(
            vulkan_base,
            MaterialProperties {
                name: "InfinitePlaneMaterial".into(),
                vs_path: "assets/shaders/spv/infinite_plane.vert.spv".into(),
                fs_path: "assets/shaders/spv/infinite_plane.frag.spv".into(),
                diffuse_texture_path: None,
                normalmap_texture_path: None,
                depth_write: false,
                uv_tiling: None,
            },
        );
        let sand_plane_mat_id = self.material_manager.request_material(
            vulkan_base,
            MaterialProperties {
                name: "SandPlaneMaterial".into(),
                vs_path: "assets/shaders/spv/main.vert.spv".into(),
                fs_path: "assets/shaders/spv/main.frag.spv".into(),
                diffuse_texture_path: Some("assets/textures/sand/color.jpg".into()),
                normalmap_texture_path: Some("assets/textures/sand/normal.png".into()),
                depth_write: true,
                uv_tiling: Some([100.0, 100.0]),
            },
        );

        // Ground mesh: reuse unit plane
        let ground_mesh_id = self
            .mesh_manager
            .request_unit_plane(vulkan_base)
            .expect("Failed to load unit plane mesh");

        let infinite_plane = SceneObject {
            transform: SceneTransform::from_euler(
                cgmath::Vector3::new(0.0, 0.0, 0.0),
                cgmath::Vector3::new(0.0, 0.0, 0.0),
                500.0,
            ),
            parts: vec![ScenePart { transform: SceneTransform::identity(), material_id: infinite_plane_mat_id, mesh_id: ground_mesh_id }],
        };

        let cube_mat_id = self.material_manager.request_material(
                vulkan_base,
                MaterialProperties {
                    name: "Cube1Material".into(),
                    vs_path: "assets/shaders/spv/main.vert.spv".into(),
                    fs_path: "assets/shaders/spv/main.frag.spv".into(),
                    diffuse_texture_path: Some("assets/textures/cube1/diffuse.png".into()),
                    normalmap_texture_path: Some("assets/textures/cube1/normal.png".into()),
                    depth_write: true,
                    uv_tiling: None,
                },
            );
        let cube_mesh_id = self
            .mesh_manager
            .request_cube(vulkan_base)
            .expect("Failed to load cube mesh");
        let cube = SceneObject {
            transform: SceneTransform::from_euler(
                cgmath::Vector3::new(0.0, 1.0, 0.0),
                cgmath::Vector3::new(0.0, 0.0, 0.0),
                1.0,
            ),
            parts: vec![ScenePart { transform: SceneTransform::identity(), material_id: cube_mat_id, mesh_id: cube_mesh_id }],
        };

        let cube2_mat_id = self.material_manager.request_material(
                vulkan_base,
                MaterialProperties {
                    name: "Cube2Material".into(),
                    vs_path: "assets/shaders/spv/main.vert.spv".into(),
                    fs_path: "assets/shaders/spv/main.frag.spv".into(),
                    diffuse_texture_path: Some("assets/textures/cube2/diffuse.png".into()),
                    normalmap_texture_path: Some("assets/textures/cube2/normal.png".into()),
                    depth_write: true,
                    uv_tiling: None,
                },
            );
        let cube2_mesh_id = self
            .mesh_manager
            .request_cube(vulkan_base)
            .expect("Failed to load cube mesh");
        let cube2 = SceneObject {
            transform: SceneTransform::from_euler(
                cgmath::Vector3::new(2.0, 1.2, -4.0),
                cgmath::Vector3::new(0.0, 15.0, 0.0),
                1.2,
            ),
            parts: vec![ScenePart { transform: SceneTransform::identity(), material_id: cube2_mat_id, mesh_id: cube2_mesh_id }],
        };

        let prims = import_gltf(
            "assets/meshes/duck.gltf",
            vulkan_base,
            &mut self.mesh_manager,
            &mut self.material_manager,
        )
        .expect("Failed to import glTF mesh");
        let duck_prim = &prims[0];

        let duck = SceneObject {
            transform: SceneTransform::from_euler(
                cgmath::Vector3::new(3.0, 0.0, 0.0),
                cgmath::Vector3::new(0.0, -90.0, 0.0),
                0.015,
            ),
            parts: vec![ScenePart { transform: SceneTransform::identity(), material_id: duck_prim.mat_id, mesh_id: duck_prim.mesh_id }],
        };

        let prims2 = import_gltf(
            "assets/meshes/sphere.gltf",
            vulkan_base,
            &mut self.mesh_manager,
            &mut self.material_manager,
        )
        .expect("Failed to import glTF mesh");
        let sphere_prim = &prims2[0];

        let sphere = SceneObject {
            transform: SceneTransform::from_euler(
                cgmath::Vector3::new(-3.0, 1.0, 0.0),
                cgmath::Vector3::new(0.0, 0.0, 0.0),
                1.0,
            ),
            parts: vec![ScenePart { transform: SceneTransform::identity(), material_id: sphere_prim.mat_id, mesh_id: sphere_prim.mesh_id }],
        };

        self.scene = Scene::new();
        self.scene.add(cube);
        self.scene.add(cube2);
        self.scene.add(duck);
        self.scene.add(sphere);
        // Record index of ground object after pushing
        let ground_index = self.scene.objects.len();
        self.scene.add(infinite_plane);
        self.ground_obj_index = Some(ground_index);
        self.infinite_plane_material_id = infinite_plane_mat_id;
        self.sand_plane_material_id = sand_plane_mat_id;

        self.camera = Camera::new();
        self.step = 0.1;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.window = Some(self.create_window(event_loop));

        let mut imgui = ImGuiContext::create();
        let mut platform = WinitPlatform::new(&mut imgui);
        platform.attach_window(
            imgui.io_mut(),
            self.window.as_ref().unwrap(),
            HiDpiMode::Rounded,
        );
        self.imgui = Some(imgui);
        self.platform = Some(platform);

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

        let renderer = ImGuiRenderer::new(
            self.vulkan_base.as_mut().unwrap(),
            self.imgui.as_mut().unwrap(),
        );
        self.imgui_renderer = Some(renderer);

        self.set_up_scene();

        self.render_graph.add(RenderPassNode::Shadow);
        self.render_graph.add(RenderPassNode::Main);
        let show = self.vulkan_base.as_ref().unwrap().engine_settings.show_ui;
        self.render_graph.set_pass_enabled(RenderPassNode::UI, show);
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        input::handle_device_event(self, event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        input::handle_window_event(self, event_loop, window_id, event);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        if let Some(vb) = self.vulkan_base.take() {
            unsafe {
                let _ = vb.device.device_wait_idle();
            }

            // Ensure render passes free their GPU objects before the device is destroyed
            self.render_graph.cleanup(&vb.device);

            self.material_manager.cleanup(&vb.device, vb.allocator.as_ref().unwrap());
            self.mesh_manager.cleanup(vb.allocator.as_ref().unwrap());

            if let Some(mut renderer) = self.imgui_renderer.take() {
                println!("🗑️ Cleaning up ImGui renderer");
                renderer.cleanup(vb.allocator.as_ref().unwrap());
            }
        }
    }
}

pub const MAX_LIGHTS: usize = 8;

/// Per-point-light controls exposed in the UI and mirrored to GPU.
#[derive(Clone, Copy)]
pub struct LightCtrl {
    pub position: [f32; 3],   // x, y, z in world space
    pub intensity: f32,
    pub color: [f32; 3],
}

/// Global world controls (sun, lights, ground material toggle).
#[derive(Clone, Copy)]
pub struct WorldControls {
    pub lights: [LightCtrl; MAX_LIGHTS],
    pub light_count: usize,
    pub sun_direction: [f32; 3],
    pub sun_intensity: f32,
    pub sun_color: [f32; 3],
    pub use_sand_ground: bool,
}

impl Default for WorldControls {
    fn default() -> Self {
        // Simple LCG for deterministic "random" without external crates
        fn lcg_next(state: &mut u32) -> f32 {
            // Numerical Recipes LCG parameters
            *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            // Map to 0.0..1.0
            (*state as f32) / (u32::MAX as f32)
        }

        let mut seed = 0xA17E_6C83; // arbitrary seed
        // Disable point lights for now (shadow mapping prep)
        let light_count = 0;

        let lights = std::array::from_fn(|_i| {
            // X and Z in [-5, 5], Y fixed at 2.0
            let x = (lcg_next(&mut seed) * 10.0) - 5.0;
            let z = (lcg_next(&mut seed) * 10.0) - 5.0;
            let y = 2.0;

            // Bright color: each channel in [0.5, 1.0]
            let r = 0.5 + 0.5 * lcg_next(&mut seed);
            let g = 0.5 + 0.5 * lcg_next(&mut seed);
            let b = 0.5 + 0.5 * lcg_next(&mut seed);

            LightCtrl {
                position: [x, y, z],
                intensity: 0.5,
                color: [r, g, b],
            }
        });

        // Reasonable default sun: slightly from above-left, white-ish
        let mut sdir = [-0.5f32, -1.0, -0.3];
        let len = (sdir[0]*sdir[0] + sdir[1]*sdir[1] + sdir[2]*sdir[2]).sqrt().max(1e-6);
        sdir[0] /= len; sdir[1] /= len; sdir[2] /= len;

        Self {
            lights,
            light_count,
            sun_direction: sdir,
            sun_intensity: 1.5,
            sun_color: [1.0, 1.0, 0.98],
            use_sand_ground: true,
        }
    }
}
