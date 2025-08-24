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

use crate::app::app::App;
use crate::graphics::camera::Camera;
use crate::app::scene::Scene;
use crate::graphics::materialmanager::MaterialManager;
use crate::graphics::meshmanager::MeshManager;
use crate::app::app::WorldControls;
use crate::vulkan::base::VulkanBase;
use crate::vulkan::imgui_renderer::ImGuiRenderer;
use crate::vulkan::base::FrameCtx;
use crate::vulkan::main_pass::MainPass;
use crate::vulkan::ui_pass::UiPass;

use imgui::Context as ImGuiContext;
use imgui_winit_support::WinitPlatform;
use winit::window::Window;


/// Render pass node enum
/// Will eventually contain things like
/// Shadow, UI
pub enum RenderPassNode {
    Main,
    UI,
}

/// UiCtx Struct
/// Used to pass the appropriate bits'n'pieces to the UI
pub struct UiCtx<'a> {
    pub ms_per_frame: f32,
    pub imgui: &'a mut ImGuiContext,
    pub window: &'a Window,
    pub platform: &'a mut WinitPlatform,
    pub show_ms_per_frame: bool,
    pub renderer: &'a mut ImGuiRenderer,
}

/// RenderCtx Struct
/// Used to pass the appropriate bits'n'pieces to the render passes
pub struct RenderCtx<'a> {
    pub frame: Option<FrameCtx>,
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
    render_passes: Vec<Box<dyn RenderPass>>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            render_passes: Vec::new(),
        }
    }

    pub fn execute(&mut self, app: &mut App) -> Result<(), Box<dyn Error>> {
        let vb = app.vulkan_base.as_mut().unwrap();

        let ui_ctx = UiCtx {
            ms_per_frame: app.current_ms_per_frame,
            imgui: app.imgui.as_mut().unwrap(),
            window: app.window.as_ref().unwrap(),
            platform: app.platform.as_mut().unwrap(),
            show_ms_per_frame: vb.engine_settings.show_ms_per_frame,
            renderer: app.imgui_renderer.as_mut().unwrap()
        };

        let frame = vb.begin_frame()?;
        let mut ctx = RenderCtx {
            frame,
            camera: &app.camera,
            scene: &app.scene,
            material_manager: &app.material_manager,
            mesh_manager: &app.mesh_manager,
            world_controls: &mut app.world_controls,
            vulkan_base: vb,
            ui_ctx: Some(ui_ctx),
        };
        
        // Iterate over all render passes and execute them
        for pass in &mut self.render_passes {
            pass.execute(&mut ctx)?;
        }
        
        ctx.vulkan_base.end_frame(ctx.frame.unwrap())?;
        
        Ok(())
    }

    pub fn add(&mut self, render_pass: RenderPassNode) {
        let pass: Box<dyn RenderPass> = match render_pass {
            RenderPassNode::Main => Box::new(MainPass::new()),
            RenderPassNode::UI => Box::new(UiPass::new()),
        };
        self.render_passes.push(pass);
    }
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}