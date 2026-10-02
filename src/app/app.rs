//! --------------------------------------------------------------------------------------
//! Application Core (app.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Main application logic that wires together windowing, rendering, and state management.
//!
//! --------------------------------------------------------------------------------------

use std::collections::VecDeque;
use std::error::Error;
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Instant;

use imgui::Context as ImGuiContext;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::ModifiersState;
use winit::window::{Window, WindowAttributes};

use crate::app::scene::{Scene, SceneObject, ScenePart, Transform as SceneTransform};
use crate::graphics::camera::Camera;
use crate::graphics::grass::GrassRenderer;
use crate::graphics::import::{
    import_model_as_object, import_model_as_object_with_shared_textured_material,
};
use crate::graphics::materialmanager::{MaterialManager, MaterialProperties};
use crate::graphics::meshmanager::MeshManager;
use crate::graphics::rocks::{ROCK_ASSET_PATHS, ROCK_MATERIAL_NAMES, cluster_members};
use crate::graphics::sky::SkyRenderer;
use crate::graphics::terrain::{ROCK_CLUSTERS, TerrainSettings, build_heightfield};
use crate::vulkan::base::{GpuPassTimings, VulkanBase};
use crate::vulkan::imgui_renderer::ImGuiRenderer;
use crate::vulkan::render_graph::{RenderGraph, RenderPassNode};

use super::input;
use super::input::InputState;

/// Enables the cursor mode used while playing in first-person view.
pub fn capture_first_person_cursor(window: &Window) {
    // Locked works on most platforms; confined is a useful fallback for window systems
    // without relative-pointer support.
    if window
        .set_cursor_grab(winit::window::CursorGrabMode::Locked)
        .is_err()
    {
        window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
            .ok();
    }
    window.set_cursor_visible(false);
}

/// Releases the pointer for ImGui without changing whether the UI is rendered.
pub fn release_first_person_cursor(window: &Window) {
    window
        .set_cursor_grab(winit::window::CursorGrabMode::None)
        .ok();
    window.set_cursor_visible(true);
}

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
    /// Walking speed in metres per second; input multiplies it by elapsed frame time.
    pub walk_speed_mps: f32,
    /// Timestamp used to make input frame-rate independent.
    pub last_movement_update: Instant,
    /// Shared terrain settings: scene placement and the first-person camera query it.
    pub terrain_settings: TerrainSettings,
    pub modifiers: ModifiersState,
    pub input: InputState,
    pub imgui: Option<ImGuiContext>,
    pub platform: Option<WinitPlatform>,
    pub imgui_renderer: Option<ImGuiRenderer>,
    pub material_manager: MaterialManager,
    pub mesh_manager: MeshManager,
    /// Separate instanced renderer for the dense outdoor grass field.
    pub grass_renderer: Option<GrassRenderer>,
    /// Fullscreen panorama renderer kept outside the ordinary object/material list.
    pub sky_renderer: Option<SkyRenderer>,
    /// Monotonic scene time for procedural animation (wind, water, and similar effects).
    pub scene_start_time: Instant,
    pub current_ms_per_frame: f32,
    pub current_gpu_ms_per_frame: Option<f32>,
    /// Previous completed GPU frame split into shadow, scene, and vegetation work.
    pub current_gpu_pass_timings: GpuPassTimings,
    pub world_controls: WorldControls,
    pub render_graph: RenderGraph,
    pub exit_flag: Arc<AtomicBool>,
    // Rolling ms history for ImGui graphs
    pub cpu_ms_history: VecDeque<f32>,
    pub gpu_ms_history: VecDeque<f32>,
    // Ground switching (two objects toggled via visibility)
    pub infinite_plane_obj_index: Option<usize>,
    pub terrain_obj_index: Option<usize>,
    pub infinite_plane_material_id: usize,
    pub terrain_material_id: usize,
}

/// CPU-side recipe for instancing one imported rock without duplicating GPU resources.
struct RockTemplate {
    parts: Vec<ScenePart>,
    local_base_y: f32,
    normalisation_scale: f32,
}

impl App {
    /// Constructs a new `App` with default state. Call `run()` to start.
    pub fn new() -> Self {
        App {
            window: None,
            vulkan_base: None,
            scene: Scene::new(),
            camera: Camera::new(),
            walk_speed_mps: 4.0,
            last_movement_update: Instant::now(),
            terrain_settings: TerrainSettings::default(),
            modifiers: ModifiersState::default(),
            input: InputState::default(),
            imgui: None,
            platform: None,
            imgui_renderer: None,
            material_manager: MaterialManager::new(),
            mesh_manager: MeshManager::new(),
            grass_renderer: None,
            sky_renderer: None,
            scene_start_time: Instant::now(),
            current_ms_per_frame: 0.0,
            current_gpu_ms_per_frame: None,
            current_gpu_pass_timings: GpuPassTimings::default(),
            world_controls: WorldControls::default(),
            render_graph: RenderGraph::new(),
            exit_flag: Arc::new(AtomicBool::new(false)),
            cpu_ms_history: VecDeque::new(),
            gpu_ms_history: VecDeque::new(),
            infinite_plane_obj_index: None,
            terrain_obj_index: None,
            infinite_plane_material_id: 0,
            terrain_material_id: 0,
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

        capture_first_person_cursor(&window);
        window
    }

    fn set_up_scene(&mut self) -> Result<(), Box<dyn Error>> {
        // Scene construction reads VulkanBase but mutates the two managers. Rust
        // permits those simultaneous borrows because they are disjoint `App` fields.
        let vulkan_base = self
            .vulkan_base
            .as_ref()
            .expect("VulkanBase must exist before scene setup");
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
        )?;
        let terrain_mat_id = self.material_manager.request_material(
            vulkan_base,
            MaterialProperties {
                // Vertex colours carry the biome; the dedicated fragment shader adds
                // stable fine detail without requiring a large ground texture asset.
                name: "TerrainBiomeMaterial".into(),
                vs_path: "assets/shaders/spv/terrain.vert.spv".into(),
                fs_path: "assets/shaders/spv/terrain.frag.spv".into(),
                diffuse_texture_path: None,
                normalmap_texture_path: None,
                depth_write: true,
                uv_tiling: None,
            },
        )?;

        // The grid is generated and uploaded once at startup, just like an imported mesh.
        let terrain_settings = self.terrain_settings;
        let terrain_mesh_id = self.mesh_manager.request_mesh_from_cpu(
            "LowRollingTerrain".into(),
            vulkan_base,
            build_heightfield(terrain_settings),
        )?;
        // The infinite grid remains a separate, flat debug ground option.
        let infinite_plane_mesh_id = self.mesh_manager.request_unit_plane(vulkan_base)?;

        // Import each rock shape once. Files from the same asset pack share one atlas and
        // material name, avoiding five near-identical pipelines and texture uploads.
        let mut rock_templates = Vec::with_capacity(ROCK_ASSET_PATHS.len());
        for (path, material_name) in ROCK_ASSET_PATHS.into_iter().zip(ROCK_MATERIAL_NAMES) {
            let rock = import_model_as_object_with_shared_textured_material(
                path,
                material_name,
                [
                    "assets/shaders/spv/rock.vert.spv",
                    "assets/shaders/spv/rock.frag.spv",
                ],
                vulkan_base,
                &mut self.mesh_manager,
                &mut self.material_manager,
            )?;
            let bounds = self
                .mesh_manager
                .object_local_bounds(&rock)
                .ok_or("imported rock has no mesh bounds")?;
            let longest_dimension = (0..3)
                .map(|axis| bounds.max[axis] - bounds.min[axis])
                .fold(0.0_f32, f32::max);
            rock_templates.push(RockTemplate {
                parts: rock.parts,
                local_base_y: bounds.min[1],
                normalisation_scale: longest_dimension.recip(),
            });
        }

        let infinite_plane = SceneObject {
            // Keep transform identity so the infinite grid shader sees stable derivatives
            transform: SceneTransform::from_euler(
                cgmath::Vector3::new(0.0, 0.0, 0.0),
                cgmath::Vector3::new(0.0, 0.0, 0.0),
                1.0,
            ),
            parts: vec![ScenePart {
                transform: SceneTransform::identity(),
                material_id: infinite_plane_mat_id,
                mesh_id: infinite_plane_mesh_id,
            }],
            visible: !self.world_controls.use_terrain_ground,
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
        )?;
        let cube_mesh_id = self.mesh_manager.request_cube(vulkan_base)?;
        let mut cube_transform = SceneTransform::from_euler(
            cgmath::Vector3::new(7.0, 0.0, 0.0),
            cgmath::Vector3::new(0.0, 0.0, 0.0),
            // The unit cube spans -1..+1, so scale 0.5 makes it one metre tall.
            0.5,
        );
        cube_transform.place_on_ground(
            terrain_settings.height_at(cube_transform.translation.x, cube_transform.translation.z),
            -1.0,
        );
        let cube = SceneObject {
            transform: cube_transform,
            parts: vec![ScenePart {
                transform: SceneTransform::identity(),
                material_id: cube_mat_id,
                mesh_id: cube_mesh_id,
            }],
            visible: true,
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
        )?;
        let cube2_mesh_id = self.mesh_manager.request_cube(vulkan_base)?;
        let mut cube2_transform = SceneTransform::from_euler(
            cgmath::Vector3::new(6.0, 0.0, -6.0),
            cgmath::Vector3::new(0.0, 15.0, 0.0),
            // The same two-metre source cube becomes 0.8 m at scale 0.4.
            0.4,
        );
        cube2_transform.place_on_ground(
            terrain_settings
                .height_at(cube2_transform.translation.x, cube2_transform.translation.z),
            -1.0,
        );
        let cube2 = SceneObject {
            transform: cube2_transform,
            parts: vec![ScenePart {
                transform: SceneTransform::identity(),
                material_id: cube2_mat_id,
                mesh_id: cube2_mesh_id,
            }],
            visible: true,
        };

        // Scale 10 makes the imported hut agree with the metre-sized reference objects
        // after the importer's glTF node transform has been applied.
        let mut hut = import_model_as_object(
            "assets/meshes/hut/hut.glb",
            vulkan_base,
            &mut self.mesh_manager,
            &mut self.material_manager,
        )?;
        let hut_local_base_y = self
            .mesh_manager
            .object_local_min_y(&hut)
            .ok_or("imported hut has no mesh bounds")?;
        hut.transform = SceneTransform::from_euler(
            // The importer has already applied the source node transform, so
            // the model's local base is already on the shared ground plane.
            // The starting camera looks along -Z, so -2 places the hut two
            // metres farther away without altering its height.
            cgmath::Vector3::new(0.0, 0.0, -2.0),
            cgmath::Vector3::new(0.0, 0.0, 0.0),
            10.0,
        );
        let hut_ground_y =
            terrain_settings.height_at(hut.transform.translation.x, hut.transform.translation.z);
        hut.transform
            .place_on_ground(hut_ground_y, hut_local_base_y);
        println!(
            "🏠 Grounded hut: local base {hut_local_base_y:.3}, world base {hut_ground_y:.3} m"
        );
        hut.visible = true;

        let mut sphere = import_model_as_object(
            "assets/meshes/sphere.gltf",
            vulkan_base,
            &mut self.mesh_manager,
            &mut self.material_manager,
        )?;
        sphere.transform = SceneTransform::from_euler(
            cgmath::Vector3::new(-6.0, 0.0, 0.0),
            cgmath::Vector3::new(0.0, 0.0, 0.0),
            // The sphere also spans -1..+1, giving it a one-metre diameter.
            0.5,
        );
        sphere.transform.place_on_ground(
            terrain_settings.height_at(
                sphere.transform.translation.x,
                sphere.transform.translation.z,
            ),
            -1.0,
        );
        sphere.visible = true;

        let mut rock_clusters = Vec::with_capacity(ROCK_CLUSTERS.len() * 3);
        for (cluster_index, placement) in ROCK_CLUSTERS.iter().copied().enumerate() {
            for member in cluster_members(cluster_index, placement) {
                let template = &rock_templates[member.variant];
                let x = placement.position[0] + member.offset[0];
                let z = placement.position[1] + member.offset[1];
                let mut transform = SceneTransform::from_euler(
                    cgmath::Vector3::new(x, 0.0, z),
                    cgmath::Vector3::new(0.0, member.rotation_degrees, 0.0),
                    template.normalisation_scale * member.size,
                );
                transform.place_on_ground(terrain_settings.height_at(x, z), template.local_base_y);
                // A shallow burial hides perfectly sharp model bottoms and makes each rock
                // feel embedded in the heightfield rather than balanced on top of it.
                transform.translation.y -= member.size * 0.07;
                rock_clusters.push(SceneObject {
                    transform,
                    // Cloning parts copies indices into manager-owned resources; it does not
                    // duplicate the Vulkan buffers or atlas image.
                    parts: template.parts.clone(),
                    visible: true,
                });
            }
        }

        // Terrain vertices already contain their world-space valley positions and heights,
        // so an identity object transform keeps height queries and rendered ground aligned.
        let terrain = SceneObject {
            transform: SceneTransform::identity(),
            parts: vec![ScenePart {
                transform: SceneTransform::identity(),
                material_id: terrain_mat_id,
                mesh_id: terrain_mesh_id,
            }],
            visible: self.world_controls.use_terrain_ground,
        };

        self.scene = Scene::new();
        self.scene.add(cube);
        self.scene.add(cube2);
        self.scene.add(hut);
        self.scene.add(sphere);
        for cluster in rock_clusters {
            self.scene.add(cluster);
        }
        // Push both ground variants and track indices
        let inf_idx = self.scene.objects.len();
        self.scene.add(infinite_plane);
        let terrain_idx = self.scene.objects.len();
        self.scene.add(terrain);
        self.infinite_plane_obj_index = Some(inf_idx);
        self.terrain_obj_index = Some(terrain_idx);
        self.infinite_plane_material_id = infinite_plane_mat_id;
        self.terrain_material_id = terrain_mat_id;

        // Grass owns its own instanced buffers and pipeline, rather than becoming tens of
        // thousands of ordinary SceneObjects.  It samples the same terrain settings used
        // above, so every clump starts at the surface height.
        self.grass_renderer = Some(GrassRenderer::new(vulkan_base, terrain_settings)?);

        // The sky has no mesh or world position.  Its panorama joins the same batched
        // texture upload as scene materials, while its tiny draw uses a dedicated pipeline.
        self.sky_renderer = Some(SkyRenderer::new(vulkan_base, &mut self.material_manager)?);

        // All scene materials have now recorded their texture uploads.  Submit
        // them together once, before the first frame can sample the images.
        self.material_manager.finish_loading(vulkan_base)?;

        self.camera = Camera::new();
        let camera_position = self.camera.position();
        self.camera.set_height_above_ground(
            terrain_settings.height_at(camera_position.x, camera_position.z),
            FIRST_PERSON_EYE_HEIGHT,
        );
        self.last_movement_update = Instant::now();
        Ok(())
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
                return;
            }
        }

        let material_manager = MaterialManager::new();
        self.material_manager = material_manager;

        let renderer = ImGuiRenderer::new(
            self.vulkan_base.as_mut().unwrap(),
            self.imgui.as_mut().unwrap(),
        );
        self.imgui_renderer = Some(renderer);

        if let Err(e) = self.set_up_scene() {
            eprintln!("Failed to set up scene: {}", e);
            event_loop.exit();
            return;
        }

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
        // `take` moves VulkanBase out of the Option.  Its own Drop runs only
        // after the managers below release resources that depend on its device.
        if let Some(vb) = self.vulkan_base.take() {
            unsafe {
                let _ = vb.device.device_wait_idle();
            }

            // Rust drops fields automatically, but Vulkan handles need this
            // explicit dependency order: passes -> managers -> renderer -> device.
            self.render_graph.cleanup(&vb.device);

            if let Some(mut sky) = self.sky_renderer.take() {
                sky.cleanup(&vb.device);
            }

            if let Some(mut grass) = self.grass_renderer.take() {
                grass.cleanup(&vb.device, vb.allocator.as_ref().unwrap());
            }

            self.material_manager.cleanup(
                &vb.device,
                vb.allocator.as_ref().unwrap(),
                vb.command_pool,
            );
            self.mesh_manager.cleanup(vb.allocator.as_ref().unwrap());

            if let Some(mut renderer) = self.imgui_renderer.take() {
                println!("🗑️ Cleaning up ImGui renderer");
                renderer.cleanup(vb.allocator.as_ref().unwrap());
            }
        }
    }
}

pub const MAX_LIGHTS: usize = 8;
/// First-person camera height above the procedural terrain, in metres.
pub const FIRST_PERSON_EYE_HEIGHT: f32 = 1.8;

/// Per-point-light controls exposed in the UI and mirrored to GPU.
#[derive(Clone, Copy)]
pub struct LightCtrl {
    pub position: [f32; 3], // x, y, z in world space
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
    pub use_terrain_ground: bool,
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
        let mut sdir = [0.117f32, -0.846, -0.520];
        let len = (sdir[0] * sdir[0] + sdir[1] * sdir[1] + sdir[2] * sdir[2])
            .sqrt()
            .max(1e-6);
        sdir[0] /= len;
        sdir[1] /= len;
        sdir[2] /= len;

        Self {
            lights,
            light_count,
            sun_direction: sdir,
            sun_intensity: 2.0,
            sun_color: [1.0, 1.0, 0.98],
            use_terrain_ground: true,
        }
    }
}
