//! --------------------------------------------------------------------------------------
//! Main Pass Implementation (main_pass.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module implements the main rendering pass that draws all scene objects.
//!
//! --------------------------------------------------------------------------------------

use crate::graphics::camera::Camera;
use crate::vulkan::attachments::{AttachmentKind, AttachmentRequest};
use crate::vulkan::render_graph::{RenderCtx, RenderPass};
use ash::vk;
use cgmath::{Matrix4, prelude::*};

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
            .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: ctx.vulkan_base.swapchain.extent })
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment_info);

        unsafe {
            device.cmd_begin_rendering(cmd, &rendering_info);

            let mut current_pipeline_id = usize::MAX;
            for obj in &ctx.scene.objects {
                if !obj.visible { continue; }
                let obj_model = obj.transform.model_matrix();
                for part in &obj.parts {
                    let model_matrix = obj_model * part.transform.model_matrix();
                    let material = &ctx.material_manager.materials[part.material_id];
                    let push_bytes = compute_push_constant_per_obj(ctx.camera, &model_matrix, material.uv_tiling);

                    if current_pipeline_id != part.material_id {
                        let material = &ctx.material_manager.materials[part.material_id];
                        device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            material.pipeline.vk_pipeline,
                        );
                        let set0 = ctx.vulkan_base.set0_descriptor_sets[image_index];
                        let sets_to_bind = if material.textures.is_some() {
                            vec![set0, material.texture_descriptor_set]
                        } else {
                            vec![set0]
                        };
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            material.pipeline.vk_layout,
                            0,
                            &sets_to_bind,
                            &[],
                        );
                        current_pipeline_id = part.material_id;
                    }

                    device.cmd_push_constants(
                        cmd,
                        material.pipeline.vk_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        &push_bytes,
                    );

                    ctx.mesh_manager.meshes[part.mesh_id].record(device, cmd);
                }
            }

            device.cmd_end_rendering(cmd);
        }

        Ok(())
    }

    fn attachments(&self) -> &[AttachmentRequest] {
        &self.attachments
    }

    fn attachment_info(&self, kind: AttachmentKind) -> (vk::ImageLayout, vk::AccessFlags) {
        match kind {
            AttachmentKind::MsaaColor | AttachmentKind::SwapchainColor | AttachmentKind::Color => (
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ),
            AttachmentKind::MsaaDepth | AttachmentKind::Depth => (
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
            AttachmentKind::Shadow => (
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::SHADER_READ,
            ),
            /*_ => (
                vk::ImageLayout::UNDEFINED,
                vk::AccessFlags::empty(),
            ),*/
        }
    }
}

/// Compute push constants for a single object
pub fn compute_push_constant_per_obj(camera: &Camera, model_matrix: &Matrix4<f32>, uv_tiling: [f32; 2]) -> Vec<u8> {
    let proj: Matrix4<f32> = *camera.get_projection();
    let view: Matrix4<f32> = *camera.get_view();

    let mv = view * model_matrix;
    let mvp = proj * mv;

    let mut bytes = Vec::with_capacity((16 + 16 + 4) * 4);
    let flatten_mat4 = |m: Matrix4<f32>, buf: &mut Vec<u8>| {
        let cols = m.transpose();
        for row in 0..4 {
            for col in 0..4 {
                buf.extend_from_slice(&cols[col][row].to_ne_bytes());
            }
        }
    };

    flatten_mat4(mvp, &mut bytes);
    flatten_mat4(mv, &mut bytes);

    // Append uv_tiling as a vec4 (xy used, zw padding) for alignment
    bytes.extend_from_slice(&uv_tiling[0].to_ne_bytes());
    bytes.extend_from_slice(&uv_tiling[1].to_ne_bytes());
    bytes.extend_from_slice(&0.0f32.to_ne_bytes());
    bytes.extend_from_slice(&0.0f32.to_ne_bytes());

    bytes
}
