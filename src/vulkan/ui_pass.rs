//! --------------------------------------------------------------------------------------
//! UI Pass Implementation (ui_pass.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module implements the UI rendering pass.
//!
//! --------------------------------------------------------------------------------------

use crate::vulkan::attachments::{AttachmentKind, AttachmentRequest};
use crate::vulkan::render_graph::{RenderCtx, RenderPass};
use ash::vk;
use imgui::{Condition, WindowFlags};

/// Render pass responsible for drawing the ImGui user interface.
pub struct UiPass {
    attachments: [AttachmentRequest; 1],
}

impl UiPass {
    /// Create a new [`UiPass`].
    pub fn new() -> Self {
        Self {
            attachments: [AttachmentRequest::new(AttachmentKind::SwapchainColor)],
        }
    }

    
}

impl RenderPass for UiPass {
    /// Render the ImGui user interface over the final swapchain image.
    fn execute(&mut self, ctx: &mut RenderCtx) -> Result<(), Box<dyn std::error::Error>> {
        let device = &ctx.vulkan_base.device;
        let cmd = ctx.frame.cmd_buf;
        let ui_ctx = ctx.ui_ctx.as_mut().unwrap();

        ui_ctx
            .platform
            .prepare_frame(ui_ctx.imgui.io_mut(), ui_ctx.window)
            .expect("Failed to prepare imgui frame");

        let ui = ui_ctx.imgui.frame();
        if ui_ctx.show_ms_per_frame {
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
                    ui.text(format!("Redraw ms: {:.2}", ui_ctx.ms_per_frame));
                });
        }

        ui_ctx.platform.prepare_render(&ui, ui_ctx.window);
        let draw_data = ui_ctx.imgui.render();

        let color_att = ctx.attachments[&AttachmentKind::SwapchainColor];

        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(color_att.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD) // Load the results of the main pass
            .store_op(vk::AttachmentStoreOp::STORE); // Store the UI on top

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: ctx.vulkan_base.swapchain.extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment_info));

        unsafe {
            device.cmd_begin_rendering(cmd, &rendering_info);

            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                ui_ctx.renderer.vk_pipeline,
            );

            ui_ctx.renderer.render(
                device,
                ctx.vulkan_base.allocator.as_ref().unwrap(),
                cmd,
                draw_data,
            );

            device.cmd_end_rendering(cmd);
        }

        Ok(())
    }

    fn attachments(&self) -> &[AttachmentRequest] {
        &self.attachments
    }

    fn attachment_info(&self, kind: AttachmentKind) -> (vk::ImageLayout, vk::AccessFlags) {
        match kind {
            AttachmentKind::SwapchainColor => (
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ),
            _ => (
                vk::ImageLayout::UNDEFINED,
                vk::AccessFlags::empty(),
            ),
        }
    }
}