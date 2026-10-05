//! --------------------------------------------------------------------------------------
//! Main Pass Implementation (main_pass.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module implements the main rendering pass that draws all scene objects.
//!
//! --------------------------------------------------------------------------------------

use crate::graphics::gpu_data::ScenePushConstants;
use crate::vulkan::attachments::{AttachmentKind, AttachmentRequest};
use crate::vulkan::render_graph::{RenderCtx, RenderPass};
use ash::vk;

/// Main rendering pass that draws all scene objects
pub struct MainPass {
    attachments: [AttachmentRequest; 5],
}

impl MainPass {
    /// Create a new [`MainPass`].
    pub fn new() -> Self {
        let attachments = [
            AttachmentRequest::new(AttachmentKind::MsaaColor),
            AttachmentRequest::new(AttachmentKind::MsaaDepth),
            AttachmentRequest::new(AttachmentKind::SwapchainColor),
            AttachmentRequest::new(AttachmentKind::Depth),
            // Read-only shadow map so the graph can manage transitions
            AttachmentRequest::new(AttachmentKind::Shadow),
        ];
        Self { attachments }
    }
}

impl RenderPass for MainPass {
    /// Render all scene objects to the swapchain and depth attachments.
    fn execute(&mut self, ctx: &mut RenderCtx) -> Result<(), Box<dyn std::error::Error>> {
        let image_index = ctx.frame.image_index as usize;
        let device = &ctx.vulkan_base.device;
        let cmd = ctx.frame.cmd_buf;

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        let msaa_samples = ctx.vulkan_base.engine_settings.msaa_samples;

        // Build attachment infos in locals that will live until we call cmd_begin_rendering.
        let (color_attachment_info, depth_attachment_info) = if msaa_samples > 1 {
            // MSAA path: render to MSAA color/depth and resolve to swapchain
            let color_att = ctx.attachments[&AttachmentKind::MsaaColor];
            let depth_att = ctx.attachments[&AttachmentKind::MsaaDepth];
            let resolve_att = ctx.attachments[&AttachmentKind::SwapchainColor];

            let color_attachment_info = vk::RenderingAttachmentInfo::default()
                .image_view(color_att.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE) // resolve writes final color
                .clear_value(clear_color)
                .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                .resolve_image_view(resolve_att.view)
                .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

            let depth_attachment_info = vk::RenderingAttachmentInfo::default()
                .image_view(depth_att.view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(clear_depth);

            (color_attachment_info, depth_attachment_info)
        } else {
            // 1x path: render directly to swapchain color and single-sample depth
            let color_att = ctx.attachments[&AttachmentKind::SwapchainColor];
            let depth_att = ctx.attachments[&AttachmentKind::Depth];

            let color_attachment_info = vk::RenderingAttachmentInfo::default()
                .image_view(color_att.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE) // we will present this image
                .clear_value(clear_color);
            // NOTE: no resolve_* fields set in 1x path

            let depth_attachment_info = vk::RenderingAttachmentInfo::default()
                .image_view(depth_att.view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(clear_depth);

            (color_attachment_info, depth_attachment_info)
        };

        // The RenderingInfo borrows from small local arrays; build them and call immediately.
        let color_attachments = [color_attachment_info];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: ctx.vulkan_base.swapchain.extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment_info);

        unsafe {
            device.cmd_begin_rendering(cmd, &rendering_info);

            // The shared pipelines use dynamic viewport/scissor so they survive an
            // extent-only swapchain resize. These values apply to sky, scene and grass.
            let extent = ctx.vulkan_base.swapchain.extent;
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            };
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);

            // The sky does not touch depth.  Drawing it first fills only the background;
            // all subsequent scene geometry naturally paints over the fullscreen triangle.
            if let Some(sky) = ctx.sky_renderer.as_deref() {
                sky.draw(ctx.vulkan_base, cmd, ctx.camera);
            }

            let mut current_pipeline_id = usize::MAX;
            for obj in &ctx.scene.objects {
                if !obj.visible {
                    continue;
                }
                let obj_model = obj.transform.model_matrix();
                for part in &obj.parts {
                    let model_matrix = obj_model * part.transform.model_matrix();
                    let material = &ctx.material_manager.materials[part.material_id];
                    let push =
                        ScenePushConstants::new(ctx.camera, &model_matrix, material.uv_tiling);

                    if current_pipeline_id != part.material_id {
                        let material = &ctx.material_manager.materials[part.material_id];
                        device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            material.pipeline.vk_pipeline,
                        );
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            material.pipeline.vk_layout,
                            0,
                            &[ctx.vulkan_base.set0_descriptor_sets[image_index]],
                            &[],
                        );
                        material.push_textures(ctx.vulkan_base, cmd);
                        current_pipeline_id = part.material_id;
                    }

                    device.cmd_push_constants(
                        cmd,
                        material.pipeline.vk_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&push),
                    );

                    ctx.mesh_manager.meshes[part.mesh_id].record(device, cmd);
                }
            }

            // Dense grass stays outside `SceneObject`: static instance buffers are selected
            // by visible terrain chunk rather than recording one draw per individual blade.
            // The alternate infinite-plane debug view is perfectly flat, so hide grass
            // there rather than leaving terrain-following blades apparently floating.
            // These boundaries deliberately surround only vegetation: the profiler's
            // `scene_ms` is the opaque scene recorded immediately before this block.
            ctx.vulkan_base
                .mark_vegetation_timing_start(cmd, image_index as u32);
            if ctx.world_controls.use_terrain_ground
                && let Some(grass) = ctx.grass_renderer.as_deref_mut()
            {
                grass.draw(
                    device,
                    cmd,
                    ctx.vulkan_base,
                    image_index,
                    ctx.camera,
                    ctx.time_seconds,
                );
            }
            ctx.vulkan_base
                .mark_vegetation_timing_end(cmd, image_index as u32);

            device.cmd_end_rendering(cmd);
        }

        Ok(())
    }

    fn attachments(&self) -> &[AttachmentRequest] {
        &self.attachments
    }

    fn uses_attachment(&self, kind: AttachmentKind, msaa_samples: u32) -> bool {
        // MainPass either renders directly to the swapchain (1x) or uses the
        // two MSAA attachments and resolves into it (>1x).
        match kind {
            AttachmentKind::MsaaColor | AttachmentKind::MsaaDepth => msaa_samples > 1,
            AttachmentKind::Depth => msaa_samples == 1,
            AttachmentKind::SwapchainColor | AttachmentKind::Shadow | AttachmentKind::Color => true,
        }
    }

    fn attachment_info(&self, kind: AttachmentKind) -> (vk::ImageLayout, vk::AccessFlags2) {
        match kind {
            AttachmentKind::MsaaColor | AttachmentKind::SwapchainColor | AttachmentKind::Color => (
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            AttachmentKind::MsaaDepth | AttachmentKind::Depth => (
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
            AttachmentKind::Shadow => (
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags2::SHADER_READ,
            ),
            /*_ => (
                vk::ImageLayout::UNDEFINED,
                vk::AccessFlags2::empty(),
            ),*/
        }
    }
}
