//! --------------------------------------------------------------------------------------
//! Shadow Pass (shadow_pass.rs)
//!
//! Renders the scene into four camera-depth slices of a directional shadow map.
//! --------------------------------------------------------------------------------------

use crate::graphics::gpu_data::ShadowPushConstants;
use crate::graphics::mesh::Vertex;
use crate::graphics::shaders::ShaderStageInfo;
use crate::graphics::shadow_math::compute_shadow_cascades;
use crate::vulkan::attachments::{AttachmentKind, AttachmentRequest};
use crate::vulkan::render_graph::{RenderCtx, RenderPass};
use ash::vk;

/// Depth-only pass that renders from the sun’s point of view into a shadow map.
pub struct ShadowPass {
    attachments: [AttachmentRequest; 1],
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl ShadowPass {
    /// Creates a new shadow pass with a default depth request. The render graph
    /// overrides the extent/format based on engine settings.
    pub fn new() -> Self {
        // Default request; actual size is overridden by the render graph from engine settings.
        let req = AttachmentRequest {
            kind: AttachmentKind::Shadow,
            format: vk::Format::D32_SFLOAT,
            extent: vk::Extent2D {
                width: 1024,
                height: 1024,
            },
            samples: vk::SampleCountFlags::TYPE_1,
        };
        Self {
            attachments: [req],
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        }
    }

    fn create_pipeline(
        &mut self,
        device: &ash::Device,
        pipeline_cache: vk::PipelineCache,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Push constants: one mat4 (mvp)
        let push_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: std::mem::size_of::<[[f32; 4]; 4]>() as u32,
        };

        let layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 0,
            p_set_layouts: std::ptr::null(),
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_range,
            ..Default::default()
        };
        self.pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        // Load vertex shader (no fragment stage)
        let vs_stage =
            ShaderStageInfo::load(vk::ShaderStageFlags::VERTEX, "shadow_depth.vert.spv")?;
        let mut vs_module = vs_stage.module_create_info();
        let stages = [vs_stage.to_create_info(&mut vs_module)];

        // Vertex input: just position
        let binding_descs = [Vertex::binding_description()];
        let attribute_descs = [vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        }];
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: binding_descs.len() as u32,
            p_vertex_binding_descriptions: binding_descs.as_ptr(),
            vertex_attribute_description_count: attribute_descs.len() as u32,
            p_vertex_attribute_descriptions: attribute_descs.as_ptr(),
            ..Default::default()
        };
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        // Viewport/scissor are dynamic; provide dummy state here
        let dummy_viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let dummy_scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
        };
        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            p_viewports: &dummy_viewport,
            scissor_count: 1,
            p_scissors: &dummy_scissor,
            ..Default::default()
        };

        // The light projection has no Y flip (the camera's does), so winding is reversed
        // relative to the main pass: cull FRONT here drops the faces pointing away from the
        // light, leaving light-facing surfaces in the map.  Acne on those faces is handled by
        // the slope-scaled bias below plus the receiver bias in main.frag.
        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::FRONT,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            // Enable a small depth bias to reduce acne
            depth_bias_enable: vk::TRUE,
            depth_bias_constant_factor: 0.0,
            depth_bias_slope_factor: 1.0,
            depth_bias_clamp: 0.0,
            line_width: 1.0,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            ..Default::default()
        };

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::LESS,
            stencil_test_enable: vk::FALSE,
            ..Default::default()
        };

        // Dynamic rendering: depth-only (no color attachments)
        let rendering_info = vk::PipelineRenderingCreateInfo {
            color_attachment_count: 0,
            p_color_attachment_formats: std::ptr::null(),
            depth_attachment_format: vk::Format::D32_SFLOAT,
            ..Default::default()
        };

        // Enable dynamic viewport and scissor states
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_depth_stencil_state: &depth_stencil,
            p_dynamic_state: &dynamic_state,
            layout: self.pipeline_layout,
            render_pass: vk::RenderPass::null(),
            subpass: 0,
            ..Default::default()
        };
        pipeline_info.p_next = &rendering_info as *const _ as *const std::ffi::c_void;

        let pipelines = unsafe {
            device
                .create_graphics_pipelines(pipeline_cache, &[pipeline_info], None)
                .map_err(|(_, e)| e)?
        };
        self.pipeline = pipelines[0];
        Ok(())
    }
}

impl RenderPass for ShadowPass {
    fn execute(&mut self, ctx: &mut RenderCtx) -> Result<(), Box<dyn std::error::Error>> {
        // Acquire the per-frame shadow attachment and set up depth-only rendering.
        let cmd = ctx.frame.cmd_buf;

        // Lazy-create depth-only pipeline
        if self.pipeline == vk::Pipeline::null() {
            self.create_pipeline(&ctx.vulkan_base.device, ctx.vulkan_base.pipeline_cache)?;
        }

        let device = &ctx.vulkan_base.device;

        let shadow_att = ctx.attachments[&AttachmentKind::Shadow];
        let res = ctx.vulkan_base.engine_settings.shadow_map_resolution;

        // Clear to far (1.0) so empty map yields no shadowing later.
        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            // Dynamic viewport/scissor to match current resolution
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: res as f32,
                height: res as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: res,
                    height: res,
                },
            };
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);

            let cascades = compute_shadow_cascades(
                ctx.camera,
                ctx.world_controls.sun_direction,
                res,
                ctx.vulkan_base.engine_settings.shadow_distance,
            );

            for (cascade_index, cascade) in cascades.iter().enumerate() {
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(shadow_att.layer_views[cascade_index])
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(clear_depth);
                let rendering_info = vk::RenderingInfo::default()
                    .render_area(scissor)
                    .layer_count(1)
                    .depth_attachment(&depth_attachment);
                device.cmd_begin_rendering(cmd, &rendering_info);

                for object in &ctx.scene.objects {
                    if !object.visible {
                        continue;
                    }
                    let object_model = object.transform.model_matrix();
                    for part in &object.parts {
                        let model = object_model * part.transform.model_matrix();
                        let push = ShadowPushConstants {
                            mvp: (cascade.world_to_light_clip * model).into(),
                        };
                        device.cmd_push_constants(
                            cmd,
                            self.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            bytemuck::bytes_of(&push),
                        );
                        ctx.mesh_manager.meshes[part.mesh_id].record(device, cmd);
                    }
                }
                device.cmd_end_rendering(cmd);
            }
        }

        Ok(())
    }

    fn attachments(&self) -> &[AttachmentRequest] {
        &self.attachments
    }

    fn attachment_info(&self, kind: AttachmentKind) -> (vk::ImageLayout, vk::AccessFlags2) {
        match kind {
            AttachmentKind::Shadow => (
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
            _ => (vk::ImageLayout::UNDEFINED, vk::AccessFlags2::empty()),
        }
    }

    fn cleanup(&mut self, device: &ash::Device) {
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                device.destroy_pipeline(self.pipeline, None);
                self.pipeline = vk::Pipeline::null();
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.pipeline_layout, None);
                self.pipeline_layout = vk::PipelineLayout::null();
            }
        }
    }
}
