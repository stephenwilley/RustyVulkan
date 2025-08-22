//! --------------------------------------------------------------------------------------
//! Render Graph Code (render_graph.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Used for constructing the path through the render engine
//!
//! --------------------------------------------------------------------------------------

use std::error::Error;

use crate::app::render::draw_frame;
use crate::app::app::App;
use crate::graphics::camera::Camera;
use crate::app::scene::Scene;
use crate::graphics::materialmanager::MaterialManager;
use crate::graphics::meshmanager::MeshManager;
use crate::app::app::WorldControls;
use crate::vulkan::base::VulkanBase;
use crate::vulkan::imgui_renderer::ImGuiRenderer;

use imgui::{Context as ImGuiContext, Condition, WindowFlags};
use imgui_winit_support::WinitPlatform;
use winit::window::Window;


/// Render pass node enum
/// Will eventually contain things like
/// Shadow, UI
pub enum RenderPassNode {
    Main,
}

/// UiCtx Struct
/// Used to pass the appropriate bits'n'pieces to the UI
pub struct UiCtx<'a> {
    pub draw_data: &'a imgui::DrawData,
    pub renderer: &'a mut ImGuiRenderer,
}

/// RenderCtx Struct
/// Used to pass the appropriate bits'n'pieces to the render passes
pub struct RenderCtx<'a> {
    pub frame: Option<crate::vulkan::base::FrameCtx>,
    pub camera: &'a Camera,
    pub scene: &'a Scene,
    pub material_manager: &'a MaterialManager,
    pub mesh_manager: &'a MeshManager,
    pub world_controls: &'a mut WorldControls,
    pub vulkan_base: &'a mut VulkanBase,
    pub ui_ctx: Option<UiCtx<'a>>,
}

/// RenderPass Trait
/// The behaviours of a render pass
pub trait RenderPass {
    fn execute(&mut self, ctx: &mut RenderCtx) -> Result<(), Box<dyn std::error::Error>>;
}

/// RenderGraph
/// Contains a list of render pass nodes
pub struct RenderGraph {
    render_passes: Vec<RenderPassNode>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            render_passes: Vec::new(),
        }
    }

    pub fn prepare_imgui_draw_data<'a>(
        ms_per_frame: f32,
        platform: &'a mut WinitPlatform,
        imgui: &'a mut ImGuiContext,
        window: &Window,
        show_ms_per_frame: bool,
        _world_controls: &mut WorldControls,
    ) -> &'a imgui::DrawData {
        platform
            .prepare_frame(imgui.io_mut(), window)
            .expect("Failed to prepare imgui frame");

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
                        | WindowFlags::NO_BACKGROUND,
                )
                .build(|| {
                    ui.text(format!("Redraw ms: {:.2}", ms_per_frame));
                });
        }
        /*ui.window("Controls")
            .size([300.0, 180.0], Condition::FirstUseEver)
            .build(|| {
                ui.text("Light Position");
                ui.slider("Y", -100.0, 100.0, &mut world_controls.lights.height);
                ui.text("Light Intensity");
                ui.slider("LI", 0.0, 10.0, &mut world_controls.light_intensity);
                ui.text("Light Radius");
                ui.slider("LR", 0.0, 50.0, &mut world_controls.light_radius);
                ui.text(format!(
                    "Light Position: {:.1}, {:.1}, {:.1}",
                    world_controls.light_pos[0],
                    world_controls.light_pos[1],
                    world_controls.light_pos[2]
                ));
            });*/
        platform.prepare_render(ui, window);
        imgui.render()
    }

    pub fn execute(&self, app: &mut App) -> Result<(), Box<dyn Error>> {
        let draw_data = {
            let vb = app.vulkan_base.as_ref().unwrap();
            let window = app.window.as_ref().unwrap();
            Self::prepare_imgui_draw_data(
                app.current_ms_per_frame,
                app.platform.as_mut().unwrap(),
                app.imgui.as_mut().unwrap(),
                window,
                vb.engine_settings.show_ms_per_frame,
                &mut app.world_controls,
            )
        };
        let ui_ctx = UiCtx { draw_data, renderer: app.imgui_renderer.as_mut().unwrap() };
        let vb = app.vulkan_base.as_mut().unwrap();
        let mut ctx = RenderCtx {
            frame: None,
            camera: &app.camera,
            scene: &app.scene,
            material_manager: &app.material_manager,
            mesh_manager: &app.mesh_manager,
            world_controls: &mut app.world_controls,
            vulkan_base: vb,
            ui_ctx: Some(ui_ctx),
        };
        draw_frame(&mut ctx)?;
        Ok(())
    }

    pub fn add(&mut self, render_pass: RenderPassNode) {
        self.render_passes.push(render_pass);
    }
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}