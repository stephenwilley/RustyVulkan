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
use crate::vulkan::attachments::{AttachmentRequest, AttachmentKind};
use crate::vulkan::base::PassAttachments;

use crate::app::app::WorldControls;
use imgui::{Context as ImGuiContext, Condition, WindowFlags};
use imgui_winit_support::WinitPlatform;
use winit::window::Window;

/// Main rendering pass that draws all scene objects
pub struct UiPass {
    // Any state that the main pass needs can be stored here
}

impl UiPass {
    pub fn new() -> Self {
        Self {}
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
}

impl RenderPass for UiPass {
    fn execute(&mut self, ctx: &mut RenderCtx) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(frame) = ctx.frame.as_mut() {
            let device = &ctx.vulkan_base.device;
            let ui_ctx = ctx.ui_ctx.as_mut().unwrap();

            let draw_data = Self::prepare_imgui_draw_data(
                ui_ctx.ms_per_frame,
                ui_ctx.platform,
                ui_ctx.imgui,
                ui_ctx.window,
                ui_ctx.show_ms_per_frame,
                ctx.world_controls,
            );

            let color = *ctx
                .attachments
                .get(&AttachmentKind::Color)
                .expect("color attachment");
            let pass_atts = PassAttachments {
                color,
                resolve: None,
                depth: None,
            };

            ctx.vulkan_base.begin_rendering(frame.cmd_buf, &pass_atts);

            // Render UI
            unsafe {
                device.cmd_bind_pipeline(
                    frame.cmd_buf,
                    vk::PipelineBindPoint::GRAPHICS,
                    ui_ctx.renderer.vk_pipeline,
                );
            }
            ui_ctx.renderer.render(
                &ctx.vulkan_base.device,
                ctx.vulkan_base.allocator.as_ref().unwrap(),
                frame.cmd_buf,
                draw_data,
            );

            ctx.vulkan_base.end_rendering(frame.cmd_buf, &pass_atts);
        }
        Ok(())
    }

    fn attachments(&self) -> Vec<AttachmentRequest> {
        vec![AttachmentRequest {
            kind: AttachmentKind::Color,
            format: vk::Format::B8G8R8A8_UNORM,
            extent: vk::Extent2D::default(),
            samples: vk::SampleCountFlags::TYPE_1,
        }]
    }
}