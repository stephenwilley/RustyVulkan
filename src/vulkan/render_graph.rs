//! --------------------------------------------------------------------------------------
//! Render Graph Code (render_graph.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Used for constructing the path through the render engine
//!
//! --------------------------------------------------------------------------------------

use crate::app::app::{App, WorldControls, MAX_LIGHTS};
use crate::app::scene::Scene;
use crate::graphics::camera::Camera;
use crate::vulkan::base::{GlobalUbo, GpuLight, GpuDirLight}; 
use cgmath::{Matrix4, Vector4};
use crate::graphics::materialmanager::MaterialManager;
use crate::graphics::meshmanager::MeshManager;
use crate::vulkan::attachments::{AttachmentHandle, AttachmentKind, AttachmentRequest};
use crate::vulkan::base::{FrameCtx, ImageTransition, VulkanBase};
use crate::vulkan::imgui_renderer::ImGuiRenderer;
use crate::vulkan::shadow_pass::ShadowPass;
use crate::vulkan::main_pass::MainPass;
use crate::vulkan::ui_pass::UiPass;
use crate::graphics::shadow_math::compute_tight_light_mats;
use ash::vk;
use imgui::Context as ImGuiContext;
use imgui_winit_support::WinitPlatform;
use std::collections::HashMap;
use std::error::Error;
use winit::window::Window;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Identifiers for the concrete render passes that can be added to a
/// [`RenderGraph`].  The graph executes passes in the order they were
/// inserted.
#[allow(dead_code)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum RenderPassNode {
    Main,
    Shadow,
    UI,
}

fn pass_order(node: RenderPassNode) -> u8 {
    match node {
        RenderPassNode::Shadow => 50,
        RenderPassNode::Main => 100,
        RenderPassNode::UI => 200,
    }
}

/// Context passed to the UI pass.  Bundles the bits and pieces ImGui needs
/// to render a frame.
pub struct UiCtx<'a> {
    pub ms_per_frame: f32,
    pub imgui: &'a mut ImGuiContext,
    pub window: &'a Window,
    pub platform: &'a mut WinitPlatform,
    pub show_ui: bool,
    pub renderer: &'a mut ImGuiRenderer,
    pub exit_flag: Arc<AtomicBool>,
}

/// Per-pass context provided to [`RenderPass::execute`].  Gives each pass
/// access to common engine state as well as any attachments it requested.
pub struct RenderCtx<'a> {
    pub frame: &'a FrameCtx,
    pub camera: &'a Camera,
    pub scene: &'a Scene,
    pub material_manager: &'a MaterialManager,
    pub mesh_manager: &'a MeshManager,
    pub vulkan_base: &'a mut VulkanBase,
    pub ui_ctx: Option<UiCtx<'a>>,
    pub world_controls: &'a mut WorldControls,
    pub attachments: &'a HashMap<AttachmentKind, AttachmentHandle>,
}

/// Tracks the current usage state of an attachment so that the render graph
/// can insert the necessary image barriers between passes.
#[derive(Clone, Copy, Debug)]
pub struct AttachmentState {
    pub handle: AttachmentHandle,
    pub layout: vk::ImageLayout,
    pub access: vk::AccessFlags,
    pub stage: vk::PipelineStageFlags,
}

/// Trait implemented by all render passes.
pub trait RenderPass {
    /// Record commands for this pass using the supplied [`RenderCtx`].
    fn execute(&mut self, ctx: &mut RenderCtx) -> Result<(), Box<dyn std::error::Error>>;

    /// List of attachment requests needed by this pass.
    fn attachments(&self) -> &[AttachmentRequest];

    /// Provides the graph with the required layout and access mask for a given attachment.
    fn attachment_info(&self, kind: AttachmentKind) -> (vk::ImageLayout, vk::AccessFlags);

    /// Optional cleanup hook for passes that own GPU objects (pipelines, layouts, etc.).
    /// Default is no-op.
    fn cleanup(&mut self, _device: &ash::Device) {}
}

/// Simple render graph that executes a linear sequence of render passes and
/// manages the lifetime and state transitions of off-screen attachments.
pub struct RenderGraph {
    render_passes: Vec<(RenderPassNode, Box<dyn RenderPass>)>,
    swap_gen_seen: u64,
}

impl RenderGraph {
    /// Create an empty render graph.
    pub fn new() -> Self {
        Self {
            render_passes: Vec::new(),
            swap_gen_seen: 0,
        }
    }

    /// Prepares the global uniform buffer data for the current frame.
    fn prepare_global_ubo(&self, world: &WorldControls, camera: &Camera, shadow_res: u32, shadow_distance: f32) -> GlobalUbo {
        let mut ubo = GlobalUbo::default();
        ubo.light_count = world.light_count.min(MAX_LIGHTS) as u32;

        let view: Matrix4<f32> = *camera.get_view();

        // Fill directional light (transform direction to view space; w=0)
        let d = world.sun_direction;
        let dir_view4 = view * Vector4::new(d[0], d[1], d[2], 0.0);
        let dv = dir_view4.truncate();
        let len = (dv.x * dv.x + dv.y * dv.y + dv.z * dv.z).sqrt().max(1e-6);
        let dir_norm = [dv.x / len, dv.y / len, dv.z / len];
        ubo.dir_light = GpuDirLight {
            direction: dir_norm,
            intensity: world.sun_intensity,
            color: world.sun_color,
            _pad: 0.0,
        };

        for i in 0..MAX_LIGHTS {
            if i < world.light_count {
                let lc = world.lights[i];
                let p_view4 = view * Vector4::new(lc.position[0], lc.position[1], lc.position[2], 1.0);
                let p_view = p_view4.truncate();
                ubo.lights[i] = GpuLight {
                    position: [p_view.x, p_view.y, p_view.z],
                    intensity: lc.intensity,
                    color: lc.color,
                    _pad: 0.0,
                };
            } else {
                // Empty slot: keep position as 0 and intensity 0 so shader can safely loop MAX_LIGHTS
                ubo.lights[i] = GpuLight {
                    position: [0.0, 0.0, 0.0],
                    intensity: 0.0,
                    color: [0.0, 0.0, 0.0],
                    _pad: 0.0,
                };
            }
        }

        // --- Compute a tight directional-light VP that follows the camera frustum ---
        let mats = compute_tight_light_mats(camera, world.sun_direction, shadow_res, shadow_distance);
        let light_vp = mats.view_to_light_clip;

        // Store as column-major [[f32;4];4] matching GLSL/std140 default (Matrix4 fields are columns)
        let m = light_vp; // no transpose
        ubo.light_vp = [
            [m.x.x, m.x.y, m.x.z, m.x.w],
            [m.y.x, m.y.y, m.y.z, m.y.w],
            [m.z.x, m.z.y, m.z.z, m.z.w],
            [m.w.x, m.w.y, m.w.z, m.w.w],
        ];

        ubo
    }

    

    /// Execute all passes in insertion order for the current frame.
    pub fn execute(&mut self, app: &mut App) -> Result<(), Box<dyn Error>> {
        let vb = app.vulkan_base.as_mut().unwrap();

        // Apply queued changes (e.g., MSAA) at a safe point, before acquiring the image
        vb.apply_pending_surface_changes(app.window.as_ref().unwrap())?;
        // Rebuild pipelines only when engine-signaled generation changes (swapchain/MSAA/wireframe)
        let swap_gen = vb.pipeline_generation();
        if swap_gen != self.swap_gen_seen {
            app.material_manager.recreate_pipelines(vb)?;
            if let Some(renderer) = app.imgui_renderer.as_mut() {
                renderer.rebuild_pipeline(vb)?;
            }
            self.swap_gen_seen = swap_gen;
        }

        let frame = match vb.begin_frame()? {
            Some(frame) => frame,
            None => return Ok(()), // Swapchain out of date
        };

        let image_index = frame.image_index as usize;

        // --- 1. Handle per-frame app state & UBO ---
        // Toggle visibility of ground variants rather than swapping materials
        if let (Some(inf_idx), Some(sand_idx)) = (app.infinite_plane_obj_index, app.sand_plane_obj_index) {
            let use_sand = app.world_controls.use_sand_ground;
            app.scene.objects[inf_idx].visible = !use_sand;
            app.scene.objects[sand_idx].visible = use_sand;
        }

        // Prepare and update frame-global data
        let ubo = self.prepare_global_ubo(
            &app.world_controls,
            &app.camera,
            vb.engine_settings.shadow_map_resolution,
            vb.engine_settings.shadow_distance,
        );
        vb.update_global_ubo(image_index, &ubo);

        let mut attachment_handles: HashMap<AttachmentKind, AttachmentHandle> = HashMap::new();
        let mut attachment_states: HashMap<AttachmentKind, AttachmentState> = HashMap::new();

        // --- 2. Collect all attachment handles and initial states ---
        let mut all_requests: Vec<&AttachmentRequest> = Vec::new();
        for (_, pass) in &self.render_passes {
            all_requests.extend(pass.attachments());
        }

        for req in all_requests {
            if attachment_handles.contains_key(&req.kind) {
                continue;
            }
            let handle = match req.kind {
                AttachmentKind::SwapchainColor => AttachmentHandle {
                    image: vb.swapchain.images[image_index],
                    view: vb.swapchain.swapchain_image_views[image_index],
                },
                AttachmentKind::MsaaColor => AttachmentHandle {
                    image: vb.swapchain.color_msaa_image,
                    view: vb.swapchain.color_msaa_image_view,
                },
                AttachmentKind::MsaaDepth => AttachmentHandle {
                    image: vb.swapchain.depth_msaa_image,
                    view: vb.swapchain.depth_msaa_image_view,
                },
                AttachmentKind::Depth => {
                    // Build a concrete request for a single-sample depth attachment
                    let mut depth_req = *req;
                    depth_req.format = vb.swapchain.depth_format; // engine’s chosen depth format
                    depth_req.extent = vb.swapchain.extent;        // match swapchain size
                    depth_req.samples = vk::SampleCountFlags::TYPE_1; // single-sampled
                    vb.get_attachment(depth_req)
                }
                AttachmentKind::Shadow => {
                    // Apply current engine setting for shadow map resolution
                    let res = vb.engine_settings.shadow_map_resolution;
                    let mut shadow_req = *req;
                    shadow_req.format = vk::Format::D32_SFLOAT;
                    shadow_req.extent = vk::Extent2D { width: res, height: res };
                    shadow_req.samples = vk::SampleCountFlags::TYPE_1;
                    vb.get_attachment(shadow_req)
                }
                _ => vb.get_attachment(*req),
            };
            attachment_handles.insert(req.kind, handle);
            attachment_states.insert(
                req.kind,
                AttachmentState {
                    handle,
                    layout: vk::ImageLayout::UNDEFINED,
                    access: vk::AccessFlags::empty(),
                    stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                },
            );
        }

        // If a shadow attachment exists in this frame, update set=0 binding=1 to reference it
        if let Some(shadow_handle) = attachment_handles.get(&AttachmentKind::Shadow) {
            vb.update_shadow_descriptor(image_index, shadow_handle.view);
        }

        // --- 3. Execute passes with transitions ---
        let last_pass_index = self.render_passes.len().saturating_sub(1);
        for (i, (pass_node, pass)) in self.render_passes.iter_mut().enumerate() {
            let _is_last_pass = i == last_pass_index;
            let cmd = frame.cmd_buf;

            let mut transitions: Vec<ImageTransition> = Vec::new();
            for req in pass.attachments() {
                let (new_layout, new_access) = pass.attachment_info(req.kind);
                let new_stage = pipeline_stage_for_access(new_access);

                let state = attachment_states.get_mut(&req.kind).unwrap();

                if state.layout != new_layout || state.access != new_access {
                    transitions.push(ImageTransition {
                        image: state.handle.image,
                        old_layout: state.layout,
                        new_layout,
                        src_access_mask: state.access,
                        dst_access_mask: new_access,
                        src_stage_mask: state.stage,
                        dst_stage_mask: new_stage,
                        aspect_mask: aspect_for_kind(req.kind),
                    });
                    state.layout = new_layout;
                    state.access = new_access;
                    state.stage = new_stage;
                }
            }

            if !transitions.is_empty() {
                vb.insert_attachment_barriers(cmd, &transitions);
            }

            let ui_ctx = if *pass_node == RenderPassNode::UI {
                Some(UiCtx {
                    ms_per_frame: app.current_ms_per_frame,
                    imgui: app.imgui.as_mut().unwrap(),
                    window: app.window.as_ref().unwrap(),
                    platform: app.platform.as_mut().unwrap(),
                    show_ui: vb.engine_settings.show_ui,
                    renderer: app.imgui_renderer.as_mut().unwrap(),
                    exit_flag: app.exit_flag.clone(),
                })
            } else {
                None
            };

            let mut ctx = RenderCtx {
                frame: &frame,
                camera: &app.camera,
                scene: &app.scene,
                material_manager: &app.material_manager,
                mesh_manager: &app.mesh_manager,
                vulkan_base: vb,
                ui_ctx,
                world_controls: &mut app.world_controls,
                attachments: &attachment_handles,
            };

            pass.execute(&mut ctx)?;
        }

        // --- 4. Final transition for presentation ---
        let swapchain_state = attachment_states
            .get_mut(&AttachmentKind::SwapchainColor)
            .unwrap();
        if swapchain_state.layout != vk::ImageLayout::PRESENT_SRC_KHR {
            vb.insert_attachment_barriers(
                frame.cmd_buf,
                &[ImageTransition {
                    image: swapchain_state.handle.image,
                    old_layout: swapchain_state.layout,
                    new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    src_access_mask: swapchain_state.access,
                    dst_access_mask: vk::AccessFlags::empty(), // No access needed for present
                    src_stage_mask: swapchain_state.stage,
                    dst_stage_mask: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                }],
            );
        }

        vb.end_frame(frame)?;
        Ok(())
    }

    /// Add a new pass to the graph.
    pub fn add(&mut self, render_pass: RenderPassNode) {
        let pass: Box<dyn RenderPass> = match render_pass {
            RenderPassNode::Main => Box::new(MainPass::new()),
            RenderPassNode::Shadow => Box::new(ShadowPass::new()),
            RenderPassNode::UI => Box::new(UiPass::new()),
        };
        let new_key = pass_order(render_pass);
        let idx = self
            .render_passes
            .iter()
            .position(|(n, _)| pass_order(*n) > new_key)
            .unwrap_or(self.render_passes.len());
        self.render_passes.insert(idx, (render_pass, pass));
    }

    /// Enables or disables a pass by node. Inserts/removes it from the graph.
    pub fn set_pass_enabled(&mut self, node: RenderPassNode, enabled: bool) {
        let has_node = self.render_passes.iter().any(|(n, _)| *n == node);
        if enabled {
            if !has_node {
                self.add(node);
            }
        } else {
            self.render_passes.retain(|(n, _)| *n != node);
        }
    }

    /// Explicitly clean up GPU resources owned by passes. Call before destroying the device.
    pub fn cleanup(&mut self, device: &ash::Device) {
        for (_node, pass) in &mut self.render_passes {
            pass.cleanup(device);
        }
    }
}

/// Computes a light view-projection that tightly fits the camera frustum.
///
/// Steps (kept simple and documented to be easy to understand):
/// - Invert the camera's view-projection to reconstruct the 8 world-space frustum corners
///   from NDC cube corners (x,y in [-1,1], z in [0,1] for Vulkan).
/// - Choose a light view that looks from the sun direction toward the frustum center.
///   We place the light eye a couple of frustum radii away along -dir.
/// - Transform the 8 corners into light view space and compute an AABB (min/max in x,y,z).
/// - Optionally snap the AABB center to the shadow map texel grid to reduce shimmering.
/// - Build an orthographic projection from the AABB extents and correct for Vulkan depth (0..1).
// moved tight light math into graphics::shadow_math

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}

fn pipeline_stage_for_access(access: vk::AccessFlags) -> vk::PipelineStageFlags {
    if access.contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE) {
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
    } else if access.contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE) {
        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
    } else if access.contains(vk::AccessFlags::SHADER_READ) {
        vk::PipelineStageFlags::FRAGMENT_SHADER
    } else {
        vk::PipelineStageFlags::TOP_OF_PIPE
    }
}

fn aspect_for_kind(kind: AttachmentKind) -> vk::ImageAspectFlags {
    match kind {
        AttachmentKind::SwapchainColor
        | AttachmentKind::MsaaColor
        | AttachmentKind::Color => vk::ImageAspectFlags::COLOR,
        AttachmentKind::MsaaDepth | AttachmentKind::Depth | AttachmentKind::Shadow => {
            vk::ImageAspectFlags::DEPTH
        }
    }
}
