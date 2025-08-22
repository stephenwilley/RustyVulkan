//! --------------------------------------------------------------------------------------
//! UI Pass Implementation (ui_pass.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module implements the UI rendering pass.
//!
//! --------------------------------------------------------------------------------------

use ash::vk;
use crate::vulkan::render_graph::{RenderCtx, RenderPass};

/// Main rendering pass that draws all scene objects
pub struct UiPass {
    // Any state that the main pass needs can be stored here
}

impl UiPass {
    pub fn new() -> Self {
        Self {}
    }
}

impl RenderPass for UiPass {
    fn execute(&mut self, ctx: &mut RenderCtx) -> Result<(), Box<dyn std::error::Error>> {
        if ctx.frame.is_some() {
            let frame = ctx.frame.as_mut().unwrap();
            let device = &ctx.vulkan_base.device;

            // Render UI
            let ui_ctx = ctx.ui_ctx.as_mut().unwrap();
            unsafe {
                device.cmd_bind_pipeline(
                    frame.cmd_buf,
                    vk::PipelineBindPoint::GRAPHICS,
                    ui_ctx.renderer.vk_pipeline,
                );
            }
            ui_ctx.renderer.render(&ctx.vulkan_base.device, ctx.vulkan_base.allocator.as_ref().unwrap(), frame.cmd_buf, ui_ctx.draw_data);
        }
        Ok(())
    }
}