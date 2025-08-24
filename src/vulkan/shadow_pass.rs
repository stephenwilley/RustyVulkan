//! --------------------------------------------------------------------------------------
//! Shadow Pass Implementation (shadow_pass.rs)
//!
//! Provides a placeholder render pass that will eventually generate shadow maps.
//! --------------------------------------------------------------------------------------

use crate::vulkan::attachments::{AttachmentKind, AttachmentRequest};
use crate::vulkan::render_graph::{RenderCtx, RenderPass};
use ash::vk;

/// Resolution of the shadow map. TODO: make this configurable.
const SHADOW_MAP_RESOLUTION: u32 = 2048;

/// Render pass responsible for building the scene's shadow map.
pub struct ShadowPass;

impl ShadowPass {
    pub fn new() -> Self {
        Self
    }
}

impl RenderPass for ShadowPass {
    fn execute(&mut self, ctx: &mut RenderCtx) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(frame) = ctx.frame.as_mut() {
            let device = &ctx.vulkan_base.device;
            let depth = *ctx
                .attachments
                .get(&AttachmentKind::Shadow)
                .expect("shadow attachment");

            unsafe {
                // Transition depth image to DEPTH_ATTACHMENT_OPTIMAL for rendering.
                let barrier = vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    old_layout: depth.layout,
                    new_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: depth.image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                };
                device.cmd_pipeline_barrier(
                    frame.cmd_buf,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );

                // Begin depth-only rendering.
                let depth_attachment = vk::RenderingAttachmentInfo {
                    image_view: depth.view,
                    image_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE,
                    clear_value: vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                    ..Default::default()
                };
                let rendering_info = vk::RenderingInfo {
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: SHADOW_MAP_RESOLUTION,
                            height: SHADOW_MAP_RESOLUTION,
                        },
                    },
                    layer_count: 1,
                    color_attachment_count: 0,
                    p_color_attachments: std::ptr::null(),
                    p_depth_attachment: &depth_attachment,
                    ..Default::default()
                };
                device.cmd_begin_rendering(frame.cmd_buf, &rendering_info);

                // Placeholder: actual shadow map rendering would occur here.
                println!("TODO: shadow map generation");

                device.cmd_end_rendering(frame.cmd_buf);

                // Transition depth image for shader read so later passes can sample it.
                let barrier2 = vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    old_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: depth.image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                };
                device.cmd_pipeline_barrier(
                    frame.cmd_buf,
                    vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier2],
                );
            }
        }
        Ok(())
    }

    fn attachments(&self) -> Vec<AttachmentRequest> {
        vec![AttachmentRequest {
            kind: AttachmentKind::Shadow,
            format: vk::Format::D32_SFLOAT,
            extent: vk::Extent2D {
                width: SHADOW_MAP_RESOLUTION,
                height: SHADOW_MAP_RESOLUTION,
            },
            samples: vk::SampleCountFlags::TYPE_1,
        }]
    }
}
