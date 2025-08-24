//! --------------------------------------------------------------------------------------
//! Main Pass Implementation (main_pass.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module implements the main rendering pass that draws all scene objects.
//!
//! --------------------------------------------------------------------------------------

use ash::vk;
use cgmath::{prelude::*, Matrix4, Vector4};
use crate::vulkan::base::{GlobalUbo, VulkanBase, GpuLight};
use crate::vulkan::render_graph::{RenderCtx, RenderPass};
use crate::vulkan::attachments::AttachmentRequest;
use crate::graphics::camera::Camera;
use crate::app::app::WorldControls;
use crate::app::app::MAX_LIGHTS;

/// Main rendering pass that draws all scene objects
pub struct MainPass {
    // Any state that the main pass needs can be stored here
}

impl MainPass {
    pub fn new() -> Self {
        Self {}
    }
}

impl RenderPass for MainPass {
    fn execute(&mut self, ctx: &mut RenderCtx) -> Result<(), Box<dyn std::error::Error>> {
        if ctx.frame.is_some() {
            let frame = ctx.frame.as_mut().unwrap();
            let image_index = frame.image_index as usize;

            update_ubo(ctx.vulkan_base, image_index, &mut ctx.world_controls, ctx.camera);
            let device = &ctx.vulkan_base.device;

            let mut current_pipeline_id = usize::MAX;
            for obj in &ctx.scene.objects {
                let model_matrix = obj.transform.model_matrix();
                let push_bytes = compute_push_constant_per_obj(ctx.camera, &model_matrix);

                unsafe {
                    if current_pipeline_id != obj.material_id {
                        device.cmd_bind_pipeline(
                            frame.cmd_buf,
                            vk::PipelineBindPoint::GRAPHICS,
                            ctx.material_manager.materials[obj.material_id].pipeline.vk_pipeline,
                        );
                        let set0 = ctx.vulkan_base.set0_descriptor_sets[image_index];
                        if let Some(_tex) = ctx.material_manager.materials[obj.material_id].textures.as_ref() {
                            device.cmd_bind_descriptor_sets(
                                frame.cmd_buf,
                                vk::PipelineBindPoint::GRAPHICS,
                                ctx.material_manager.materials[obj.material_id].pipeline.vk_layout,
                                0,
                                &[set0, ctx.material_manager.materials[obj.material_id].texture_descriptor_set],
                                &[],
                            );
                        } else {
                            device.cmd_bind_descriptor_sets(
                                frame.cmd_buf,
                                vk::PipelineBindPoint::GRAPHICS,
                                ctx.material_manager.materials[obj.material_id].pipeline.vk_layout,
                                0,
                                &[set0],
                                &[],
                            );
                        }
                        current_pipeline_id = obj.material_id;
                    }
                    device.cmd_push_constants(
                        frame.cmd_buf,
                        ctx.material_manager.materials[obj.material_id].pipeline.vk_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        &push_bytes,
                    );
                }
                ctx.mesh_manager.meshes[obj.mesh_id].record(&device, frame.cmd_buf);
            }
        }
        Ok(())
    }

    fn attachments(&self) -> Vec<AttachmentRequest> {
        Vec::new()
    }
}

/// Compute push constants for a single object
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

/// Update the global UBO with lighting data
pub fn update_ubo(
    base: &mut VulkanBase,
    image_index: usize,
    world: &mut WorldControls,
    camera: &Camera,
) {
    world.lights_rotation = (world.lights_rotation + 0.01) % std::f32::consts::TAU;

    // Build CPU-side UBO
    let mut ubo = GlobalUbo::default();
    ubo.light_count = world.light_count as u32;

    let view: Matrix4<f32> = *camera.get_view();

    // Evenly distribute *active* lights around the circle so they don't bunch up
    let active = world.light_count.min(MAX_LIGHTS);
    let n = active.max(1) as f32;
    for i in 0..active {
        let lc = world.lights[i];
        // Spread the active lights evenly (ignore the stored phase so N lights are 2π/N apart)
        let base_phase = (i as f32) * (std::f32::consts::TAU / n);
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
    for i in active..MAX_LIGHTS {
        ubo.lights[i] = GpuLight {
            position: [0.0, 0.0, 0.0],
            intensity: 0.0,
            color: [0.0, 0.0, 0.0],
            _pad: 0.0,
        };
    }

    // Upload (same mapping pattern you already use)
    let allocation = &mut base.ubo_allocations[image_index];
    unsafe {
        let ptr = base
            .allocator
            .as_ref().unwrap()
            .map_memory(allocation)
            .expect("Map UBO") as *mut u8;
        std::ptr::copy_nonoverlapping(
            &ubo as *const GlobalUbo as *const u8,
            ptr,
            std::mem::size_of::<GlobalUbo>(),
        );
        base.allocator.as_ref().unwrap().unmap_memory(allocation);
    }
}