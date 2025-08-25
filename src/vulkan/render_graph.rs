//! --------------------------------------------------------------------------------------
//! Render Graph Code (render_graph.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Used for constructing the path through the render engine
//!
//! --------------------------------------------------------------------------------------

use std::collections::HashMap;
use std::error::Error;

use ash::vk;

use crate::app::app::App;
use crate::app::app::WorldControls;
use crate::app::scene::Scene;
use crate::graphics::camera::Camera;
use crate::graphics::materialmanager::MaterialManager;
use crate::graphics::meshmanager::MeshManager;
use crate::vulkan::attachments::{AttachmentHandle, AttachmentKind, AttachmentRequest};
use crate::vulkan::base::{FrameCtx, ImageTransition, PassAttachments, VulkanBase};
use crate::vulkan::imgui_renderer::ImGuiRenderer;
use crate::vulkan::main_pass::MainPass;
use crate::vulkan::shadow_pass::ShadowPass;
use crate::vulkan::ui_pass::UiPass;

use imgui::Context as ImGuiContext;
use imgui_winit_support::WinitPlatform;
use winit::window::Window;

/// Render pass node enum
/// Will eventually contain things like
/// Shadow, UI
pub enum RenderPassNode {
    Main,
    Shadow,
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
    pub attachments: HashMap<AttachmentKind, AttachmentHandle>,
    pub attachment_states: HashMap<AttachmentKind, AttachmentState>,
}

/// Tracks the current usage state of an attachment
#[derive(Clone, Copy)]
pub struct AttachmentState {
    pub handle: AttachmentHandle,
    pub access: vk::AccessFlags,
    pub stage: vk::PipelineStageFlags,
    pub aspect: vk::ImageAspectFlags,
}

/// RenderPass Trait
/// The behaviours of a render pass
pub trait RenderPass {
    fn execute(&mut self, ctx: &mut RenderCtx) -> Result<(), Box<dyn std::error::Error>>;
    fn attachments(&self) -> Vec<AttachmentRequest>;
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
            renderer: app.imgui_renderer.as_mut().unwrap(),
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
            attachments: HashMap::new(),
            attachment_states: HashMap::new(),
        };
        if ctx.frame.is_some() {
            // Gather command buffer and image index before borrowing ctx mutably
            let (cmd_buf, idx) = {
                let f = ctx.frame.as_ref().unwrap();
                (f.cmd_buf, f.image_index as usize)
            };
            // Build attachments for the main swapchain render pass
            let attachments = PassAttachments {
                color: AttachmentHandle {
                    image: ctx.vulkan_base.swapchain.color_msaa_image,
                    view: ctx.vulkan_base.swapchain.color_msaa_image_view,
                    layout: ctx.vulkan_base.swapchain.color_msaa_layout,
                },
                resolve: Some(AttachmentHandle {
                    image: ctx.vulkan_base.swapchain.images[idx],
                    view: ctx.vulkan_base.swapchain.swapchain_image_views[idx],
                    layout: ctx.vulkan_base.swapchain.image_layouts[idx],
                }),
                depth: Some(AttachmentHandle {
                    image: ctx.vulkan_base.swapchain.depth_msaa_image,
                    view: ctx.vulkan_base.swapchain.depth_msaa_image_view,
                    layout: ctx.vulkan_base.swapchain.depth_msaa_layout,
                }),
            };

            ctx.vulkan_base.begin_rendering(cmd_buf, &attachments);

            // Gather all attachment requests from passes
            let mut requests: HashMap<AttachmentKind, AttachmentRequest> = HashMap::new();
            for pass in &self.render_passes {
                for req in pass.attachments() {
                    requests.insert(req.kind, req);
                }
            }

            // Acquire all attachments from VulkanBase and initialise their usage state
            for (kind, req) in &requests {
                let handle = ctx.vulkan_base.get_attachment(*req);
                let aspect = match kind {
                    AttachmentKind::Color => vk::ImageAspectFlags::COLOR,
                    AttachmentKind::Depth | AttachmentKind::Shadow => vk::ImageAspectFlags::DEPTH,
                };
                ctx.attachment_states.insert(
                    *kind,
                    AttachmentState {
                        handle,
                        access: vk::AccessFlags::empty(),
                        stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                        aspect,
                    },
                );
            }

            // Iterate over all render passes, providing the requested attachments
            for pass in &mut self.render_passes {
                ctx.attachments.clear();
                let pass_reqs = pass.attachments();
                for req in &pass_reqs {
                    if let Some(state) = ctx.attachment_states.get(&req.kind) {
                        ctx.attachments.insert(req.kind, state.handle);
                    }
                }
                pass.execute(&mut ctx)?;

                if let Some(frame) = ctx.frame.as_ref() {
                    let mut transitions: Vec<ImageTransition> = Vec::new();
                    for req in pass_reqs {
                        if let Some(state) = ctx.attachment_states.get_mut(&req.kind) {
                            let (new_layout, new_access, new_stage) = match req.kind {
                                AttachmentKind::Color | AttachmentKind::Shadow => (
                                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                    vk::AccessFlags::SHADER_READ,
                                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                                ),
                                AttachmentKind::Depth => (
                                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                                ),
                            };
                            if state.handle.layout != new_layout
                                || state.access != new_access
                                || state.stage != new_stage
                            {
                                transitions.push(ImageTransition {
                                    image: state.handle.image,
                                    old_layout: state.handle.layout,
                                    new_layout,
                                    src_access_mask: state.access,
                                    dst_access_mask: new_access,
                                    src_stage_mask: state.stage,
                                    dst_stage_mask: new_stage,
                                    aspect_mask: state.aspect,
                                });
                                state.handle.layout = new_layout;
                                state.access = new_access;
                                state.stage = new_stage;
                            }
                        }
                    }
                    ctx.vulkan_base
                        .insert_attachment_barriers(frame.cmd_buf, &transitions);
                }
            }

            ctx.vulkan_base.end_rendering(cmd_buf, &attachments);
            let frame = ctx.frame.take().unwrap();
            ctx.vulkan_base.end_frame(frame)?;
        }

        Ok(())
    }

    pub fn add(&mut self, render_pass: RenderPassNode) {
        let pass: Box<dyn RenderPass> = match render_pass {
            RenderPassNode::Main => Box::new(MainPass::new()),
            RenderPassNode::Shadow => Box::new(ShadowPass::new()),
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
