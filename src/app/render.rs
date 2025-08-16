use std::error::Error;

use ash::vk;
use cgmath::{prelude::*};
use cgmath::{Matrix4, Vector4};
use imgui::{Context as ImGuiContext, Condition, WindowFlags};
use imgui_winit_support::WinitPlatform;
use winit::window::Window;
use std::f32::consts::TAU;

use crate::camera::Camera;
use crate::vulkan::base::{GlobalUbo, VulkanBase, GpuLight};

use super::{App, WorldControls};

pub fn prepare_imgui_draw_data<'a>(
    ms_per_frame: f32,
    platform: &'a mut WinitPlatform,
    imgui: &'a mut ImGuiContext,
    window: &Window,
    show_ms_per_frame: bool,
    world_controls: &mut WorldControls,
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

pub fn compute_push_constant_per_obj(
    camera: &Camera,
    model_matrix: &Matrix4<f32>,
) -> Vec<u8> {
    let proj: Matrix4<f32> = *camera.get_projection();
    let view: Matrix4<f32> = *camera.get_view();

    let mv = view * model_matrix;
    let mvp = proj * mv;

    let mut bytes = Vec::with_capacity((16 + 16) * 4);
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

    bytes
}

pub fn update_ubo(
    base: &VulkanBase,
    image_index: usize,
    world: &mut WorldControls,
    camera: &Camera,
) {
    world.lights_rotation = (world.lights_rotation + 0.01) % TAU;

    // Build CPU-side UBO
    let mut ubo = GlobalUbo::default();
    ubo.light_count = world.light_count as u32;

    let view: Matrix4<f32> = *camera.get_view();

    // Evenly distribute *active* lights around the circle so they don't bunch up
    let active = world.light_count.min(crate::app::app::MAX_LIGHTS);
    let n = active.max(1) as f32;
    for i in 0..active {
        let lc = world.lights[i];
        // Spread the active lights evenly (ignore the stored phase so N lights are 2π/N apart)
        let base_phase = (i as f32) * (TAU / n);
        let angle = base_phase + world.lights_rotation;

        let x = angle.sin() * lc.radius;
        let z = angle.cos() * lc.radius;

        let p_view4 = view * Vector4::new(x, lc.height, z, 1.0);
        let p_view = p_view4.truncate();

        ubo.lights[i] = GpuLight {
            position: [p_view.x, p_view.y, p_view.z],
            intensity: lc.intensity,
            color: lc.color,
            _pad: 0.0,
        };
    }

    // Clear any remaining (inactive) light slots to avoid stale data
    for i in active..crate::app::app::MAX_LIGHTS {
        ubo.lights[i] = GpuLight {
            position: [0.0, 0.0, 0.0],
            intensity: 0.0,
            color: [0.0, 0.0, 0.0],
            _pad: 0.0,
        };
    }

    // Upload (same mapping pattern you already use)
    let mem = base.ubo_memory[image_index];
    let size = std::mem::size_of::<GlobalUbo>() as vk::DeviceSize;
    unsafe {
        let ptr = base
            .device
            .map_memory(mem, 0, size, vk::MemoryMapFlags::empty())
            .expect("Map UBO");
        std::ptr::copy_nonoverlapping(
            &ubo as *const GlobalUbo as *const u8,
            ptr as *mut u8,
            std::mem::size_of::<GlobalUbo>(),
        );
        base.device.unmap_memory(mem);
    }
}

pub fn draw_frame(app: &mut App) -> Result<(), Box<dyn Error>> {
    if let Some(vb) = app.vulkan_base.as_mut() {
        let imgui_renderer = app.imgui_renderer.as_mut().unwrap();

        let s = app.step;
        if app.input.moving_forward {
            app.camera.translate(s, 0.0)
        }
        if app.input.moving_backward {
            app.camera.translate(-s, 0.0)
        }
        if app.input.moving_left {
            app.camera.translate(0.0, -s)
        }
        if app.input.moving_right {
            app.camera.translate(0.0, s)
        }

        let window = app.window.as_ref().unwrap();
        let draw_data = prepare_imgui_draw_data(
            app.current_ms_per_frame,
            app.platform.as_mut().unwrap(),
            app.imgui.as_mut().unwrap(),
            window,
            vb.debug_settings.show_ms_per_frame,
            &mut app.world_controls,
        );

        if let Some(frame) = vb.begin_frame()? {
            let image_index = frame.image_index as usize;
            let device = &vb.device;

            update_ubo(&vb, image_index, &mut app.world_controls, &app.camera);

            let mut current_pipeline_id = usize::MAX;
            for obj in &app.scene.objects {
                let model_matrix = obj.transform.model_matrix();
                let push_bytes = compute_push_constant_per_obj(&app.camera, &model_matrix);

                unsafe {
                    if current_pipeline_id != obj.material_id {
                        device.cmd_bind_pipeline(
                            frame.cmd_buf,
                            vk::PipelineBindPoint::GRAPHICS,
                            app.material_manager.materials[obj.material_id].pipeline.vk_pipeline,
                        );
                        let set0 = vb.set0_descriptor_sets[image_index];
                        if let Some(_tex) = app.material_manager.materials[obj.material_id].textures.as_ref() {
                            device.cmd_bind_descriptor_sets(
                                frame.cmd_buf,
                                vk::PipelineBindPoint::GRAPHICS,
                                app.material_manager.materials[obj.material_id].pipeline.vk_layout,
                                0,
                                &[set0, app.material_manager.materials[obj.material_id].texture_descriptor_set],
                                &[],
                            );
                        } else {
                            device.cmd_bind_descriptor_sets(
                                frame.cmd_buf,
                                vk::PipelineBindPoint::GRAPHICS,
                                app.material_manager.materials[obj.material_id].pipeline.vk_layout,
                                0,
                                &[set0],
                                &[],
                            );
                        }
                        current_pipeline_id = obj.material_id;
                    }
                    device.cmd_push_constants(
                        frame.cmd_buf,
                        app.material_manager.materials[obj.material_id].pipeline.vk_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        &push_bytes,
                    );
                }
                app.mesh_manager.meshes[obj.mesh_id].record(device, frame.cmd_buf);
            }

            unsafe {
                device.cmd_bind_pipeline(
                    frame.cmd_buf,
                    vk::PipelineBindPoint::GRAPHICS,
                    imgui_renderer.vk_pipeline,
                );
            }
            imgui_renderer.render(device, frame.cmd_buf, draw_data);
            vb.end_frame(frame)?;
        }
    }
    Ok(())
}

