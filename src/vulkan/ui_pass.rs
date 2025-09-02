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
use std::sync::atomic::Ordering;
use imgui::PlotLines;

/// Render pass responsible for drawing the ImGui user interface.
pub struct UiPass {
    attachments: [AttachmentRequest; 1],
    show_point_lights_window: bool,
    show_sun_window: bool,
    show_ground_window: bool,
    show_shadow_map_window: bool,
    show_profiler_window: bool,
}

impl UiPass {
    /// Create a new [`UiPass`].
    pub fn new() -> Self {
        Self {
            attachments: [AttachmentRequest::new(AttachmentKind::SwapchainColor)],
            show_point_lights_window: false,
            show_sun_window: false,
            show_ground_window: false,
            show_shadow_map_window: false,
            show_profiler_window: false,
        }
    }

    
}

impl RenderPass for UiPass {
    /// Render the ImGui user interface over the final swapchain image.
    fn execute(&mut self, ctx: &mut RenderCtx) -> Result<(), Box<dyn std::error::Error>> {
        let cmd = ctx.frame.cmd_buf;
        let ui_ctx = ctx.ui_ctx.as_mut().unwrap();

        ui_ctx
            .platform
            .prepare_frame(ui_ctx.imgui.io_mut(), ui_ctx.window)
            .expect("Failed to prepare imgui frame");

        let ui = ui_ctx.imgui.frame();
        if ui_ctx.show_ui {
            if let Some(_menu_bar) = ui.begin_main_menu_bar() {
                if let Some(_main_menu) = ui.begin_menu("Main") {
                    if ui.menu_item("Exit") {
                        ui_ctx.exit_flag.store(true, Ordering::Relaxed);
                    }
                    _main_menu.end();
                }

                if let Some(_world_menu) = ui.begin_menu("World Controls") {
                    if ui.menu_item("Point Lights") {
                        self.show_point_lights_window = true;
                    }
                    if ui.menu_item("Sun Light") {
                        self.show_sun_window = true;
                    }
                    if ui.menu_item("Ground Surface") {
                        self.show_ground_window = true;
                    }
                    _world_menu.end();
                }

                if let Some(_engine_menu) = ui.begin_menu("Engine") {
                    if let Some(_msaa_menu) = ui.begin_menu("MSAA") {
                        let options = ctx.vulkan_base.supported_msaa_samples();
                        let mut current = ctx.vulkan_base.get_msaa_samples() as i32;
                        let prev = current;
                        for (i, samples) in options.iter().enumerate() {
                            if ui.radio_button(&format!("{}x", samples), &mut current, *samples as i32) {}
                            if i + 1 != options.len() { ui.same_line(); }
                        }
                        ui.new_line();
                        if current != prev {
                            ctx.vulkan_base.request_msaa_samples(current as u32)
                        }
                        _msaa_menu.end();
                    }
                    if let Some(_shadow_menu) = ui.begin_menu("Shadow Map Resolution") {
                        let mut cur_res = ctx.vulkan_base.engine_settings.shadow_map_resolution as i32;
                        let prev = cur_res;
                        let choices = [512, 1024, 2048, 4096];
                        for (i, &opt) in choices.iter().enumerate() {
                            if ui.radio_button(&format!("{}", opt), &mut cur_res, opt as i32) {}
                            if i + 1 != choices.len() { ui.same_line(); }
                        }
                        ui.new_line();
                        if cur_res != prev {
                            ctx.vulkan_base.engine_settings.shadow_map_resolution = cur_res as u32;
                        }
                        _shadow_menu.end();
                    }
                    // Checkable toggle: wireframe on/off
                    let wireframe = ctx.vulkan_base.engine_settings.wireframe;
                    if ui.menu_item_config("Wireframe").selected(wireframe).build() {
                        ctx.vulkan_base.toggle_wireframe();
                    }
                    if ui.menu_item("Show Shadow Map") {
                        self.show_shadow_map_window = true;
                    }
                    if ui.menu_item("Profiler") {
                        self.show_profiler_window = true;
                    }
                    _engine_menu.end();
                }

                // Right-align the text with CPU and optional GPU time
                let text = match ui_ctx.gpu_ms_per_frame {
                    Some(g) => format!("CPU: {:.2} ms | GPU: {:.2} ms", ui_ctx.cpu_ms_per_frame, g),
                    None => format!("CPU: {:.2} ms | GPU: --", ui_ctx.cpu_ms_per_frame),
                };
                let text_size = ui.calc_text_size(&text);
                let menu_bar_width = ui.content_region_avail()[0];
                ui.set_cursor_pos([menu_bar_width - text_size[0], 0.0]);
                ui.text(text);

                _menu_bar.end();
            }
        }

        if self.show_profiler_window {
            let mut open = true;
            ui.window("Profiler")
                .opened(&mut open)
                .always_auto_resize(true)
                .build(|| {
                    ui.text("Frame Timings");
                    ui.separator();
                    match ui_ctx.gpu_ms_per_frame {
                        Some(g) => ui.text(format!("Total GPU: {:.2} ms", g)),
                        None => ui.text("Total GPU: --"),
                    }
                    ui.text(format!("Total CPU record: {:.2} ms", ui_ctx.cpu_ms_per_frame));

                    ui.new_line();
                    ui.text("CPU ms (last ~5s)");
                    if !ui_ctx.cpu_ms_history.is_empty() {
                        let cpu_max = ui_ctx
                            .cpu_ms_history
                            .iter()
                            .cloned()
                            .fold(0.0_f32, f32::max);
                        let cpu_overlay = format!("max {:.1} ms", cpu_max);
                        PlotLines::new(&ui, "CPU", &ui_ctx.cpu_ms_history)
                            .graph_size([300.0, 80.0])
                            .scale_min(0.0)
                            .scale_max(33.0)
                            .overlay_text(&cpu_overlay)
                            .build();
                    } else {
                        ui.text("(no data yet)");
                    }

                    ui.new_line();
                    ui.text("GPU ms (last ~5s)");
                    if !ui_ctx.gpu_ms_history.is_empty() {
                        let gpu_max = ui_ctx
                            .gpu_ms_history
                            .iter()
                            .cloned()
                            .fold(0.0_f32, f32::max);
                        let gpu_overlay = format!("max {:.1} ms", gpu_max);
                        PlotLines::new(&ui, "GPU", &ui_ctx.gpu_ms_history)
                            .graph_size([300.0, 80.0])
                            .scale_min(0.0)
                            .scale_max(33.0)
                            .overlay_text(&gpu_overlay)
                            .build();
                    } else {
                        ui.text("(no data yet)");
                    }
                });
            if !open { self.show_profiler_window = false; }
        }

        // Shadow Map debug window
        if self.show_shadow_map_window {
            // Prepare/refresh the texture ID outside the closure to avoid borrowing issues
            let shadow_tex_id_opt: Option<imgui::TextureId> = ctx
                .attachments
                .get(&AttachmentKind::Shadow)
                .map(|handle| {
                    let view = handle.view;
                    if ui_ctx.renderer.last_shadow_view != Some(view) {
                        // If we already have a set, wait for GPU to be idle, then safely update it.
                        if let Some(id) = ui_ctx.renderer.shadow_tex_id {
                            unsafe { ctx.vulkan_base.device.device_wait_idle().ok(); }
                            let _ = ui_ctx.renderer.ensure_texture(
                                ctx.vulkan_base,
                                ctx.vulkan_base.shadow_sampler,
                                view,
                                Some(id),
                            );
                        } else {
                            // First-time creation.
                            let id = ui_ctx.renderer.ensure_texture(
                                ctx.vulkan_base,
                                ctx.vulkan_base.shadow_sampler,
                                view,
                                None,
                            );
                            ui_ctx.renderer.shadow_tex_id = Some(id);
                        }
                        ui_ctx.renderer.last_shadow_view = Some(view);
                    }
                    ui_ctx.renderer.shadow_tex_id.unwrap()
                });

            let mut open = true;
            ui.window("Shadow Map")
                .opened(&mut open)
                .always_auto_resize(true)
                .build(|| {
                    if let Some(tex_id) = shadow_tex_id_opt {
                        imgui::Image::new(tex_id, [256.0, 256.0])
                            .uv0([0.0, 1.0])
                            .uv1([1.0, 0.0])
                            .build(&ui);
                    } else {
                        ui.text("Shadow attachment not available");
                    }
                });
            if !open { self.show_shadow_map_window = false; }
        }

        if self.show_point_lights_window {
            let mut open = true;
            ui.window("Point Lights")
                .opened(&mut open)
                .always_auto_resize(true)
                .build(|| {
                    ui.text("Adjust point light positions");
                    ui.separator();

                    let count = ctx.world_controls.light_count.min(crate::app::app::MAX_LIGHTS);
                    for i in 0..count {
                        ui.text(format!("Light {}", i + 1));
                        let l = &mut ctx.world_controls.lights[i];

                        // X slider: [-5, 5]
                        ui.slider(&format!("X##{}", i), -5.0, 5.0, &mut l.position[0]);
                        // Y slider: [0, 5] (default 2.0)
                        ui.slider(&format!("Y##{}", i), 0.0, 5.0, &mut l.position[1]);
                        // Z slider: [-5, 5]
                        ui.slider(&format!("Z##{}", i), -5.0, 5.0, &mut l.position[2]);
                        // Intensity slider: [0, 5]
                        ui.slider(&format!("Intensity##{}", i), 0.0, 5.0, &mut l.intensity);
                        // Full-range color editor for RGB
                        ui.color_edit3(&format!("Color##{}", i), &mut l.color);

                        if i + 1 != count { ui.separator(); }
                    }
                });
            if !open {
                self.show_point_lights_window = false;
            }
        }

        if self.show_sun_window {
            let mut open = true;
            ui.window("Sun Light")
                .opened(&mut open)
                .always_auto_resize(true)
                .build(|| {
                    ui.text("Adjust directional sun light");
                    ui.separator();

                    // Direction sliders in [-1, 1], then normalize
                    let mut dir = ctx.world_controls.sun_direction;
                    ui.slider("Dir X", -1.0, 1.0, &mut dir[0]);
                    ui.slider("Dir Y", -1.0, 0.0, &mut dir[1]);
                    ui.slider("Dir Z", -1.0, 1.0, &mut dir[2]);

                    // Normalize to keep it a unit vector
                    let len = (dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]).sqrt();
                    if len > 1e-6 {
                        dir[0] /= len; dir[1] /= len; dir[2] /= len;
                    }
                    ctx.world_controls.sun_direction = dir;

                    // Intensity and color
                    ui.slider("Intensity", 0.0, 5.0, &mut ctx.world_controls.sun_intensity);
                    ui.color_edit3("Color", &mut ctx.world_controls.sun_color);
                });
            if !open {
                self.show_sun_window = false;
            }
        }

        if self.show_ground_window {
            let mut open = true;
            ui.window("Ground Surface")
                .opened(&mut open)
                .always_auto_resize(true)
                .build(|| {
                    ui.text("Choose ground rendering");
                    ui.separator();
                    let mut use_sand = ctx.world_controls.use_sand_ground;
                    if ui.radio_button("Infinite Plane", &mut use_sand, false) {}
                    ui.same_line();
                    if ui.radio_button("Sand", &mut use_sand, true) {}
                    ctx.world_controls.use_sand_ground = use_sand;
                });
            if !open {
                self.show_ground_window = false;
            }
        }

        // MSAA controlled via Engine Settings submenu

        // (Wireframe uses checkable menu item; no pop-up window)

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
            ctx.vulkan_base.device.cmd_begin_rendering(cmd, &rendering_info);

            ctx.vulkan_base.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                ui_ctx.renderer.vk_pipeline,
            );

            ui_ctx.renderer.render(
                &ctx.vulkan_base.device,
                ctx.vulkan_base.allocator.as_ref().unwrap(),
                cmd,
                draw_data,
            );

            ctx.vulkan_base.device.cmd_end_rendering(cmd);
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
