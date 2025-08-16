use std::error::Error;
use std::time::Instant;

use imgui::Context as ImGuiContext;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::ModifiersState;
use winit::window::{Window, WindowAttributes};

use crate::camera::Camera;
use crate::graphics::gltf_loader::import_gltf;
use crate::graphics::materialmanager::{MaterialManager, MaterialProperties};
use crate::graphics::meshmanager::MeshManager;
use crate::scene::{Scene, SceneObject, Transform as SceneTransform};
use crate::vulkan::base::VulkanBase;
use crate::vulkan::imgui_renderer::ImGuiRenderer;

use super::input;
use super::input::InputState;

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
}

impl App {
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
        }
    }

    pub fn run(mut self) -> Result<(), Box<dyn Error>> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        Ok(event_loop.run_app(&mut self)?)
    }

    fn create_window(&mut self, event_loop: &ActiveEventLoop) -> Window {
        let window_attributes = WindowAttributes::default().with_title("Rusty Vulkan");
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
                    depth_write: false,
                },
            ),
            mesh_id: self
                .mesh_manager
                .request_unit_plane(vulkan_base)
                .expect("Failed to load unit plane mesh"),
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
                },
            ),
            mesh_id: self
                .mesh_manager
                .request_cube(vulkan_base)
                .expect("Failed to load cube mesh"),
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
            material_id: duck_prim.mat_id,
            mesh_id: duck_prim.mesh_id,
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
            material_id: sphere_prim.mat_id,
            mesh_id: sphere_prim.mesh_id,
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

            self.material_manager.cleanup(&vb.device, &vb.allocator);
            self.mesh_manager.cleanup(&vb.allocator);

            if let Some(mut renderer) = self.imgui_renderer.take() {
                println!("🗑️ Cleaning up ImGui renderer");
                renderer.cleanup();
            }
        }
    }
}

pub const MAX_LIGHTS: usize = 8;

#[derive(Clone, Copy)]
pub struct LightCtrl {
    pub radius: f32,
    pub height: f32,
    pub intensity: f32,
    pub color: [f32; 3],
    pub phase: f32,   // per-light offset around the circle (0..2π)
}

#[derive(Clone, Copy)]
pub struct WorldControls {
    pub lights: [LightCtrl; MAX_LIGHTS],
    pub light_count: usize,
    pub lights_rotation: f32,
}

impl Default for WorldControls {
    fn default() -> Self {
        let light_count = 8;
        use std::f32::consts::TAU;
        let base = LightCtrl {
            radius: 8.0, height: 4.0, intensity: 0.5,
            color: [1.0, 1.0, 1.0], phase: 0.0,
        };
        let lights = std::array::from_fn(|i| {
            let mut l = base;
            l.phase = (i as f32) * (TAU / light_count as f32);
            // (optional) tint each light a bit:
            let hue = i as f32 / light_count as f32;
            l.color = [hue, 1.0 - hue, hue * hue];
            l
        });
        Self { lights, light_count, lights_rotation: 0.0 }
    }
}