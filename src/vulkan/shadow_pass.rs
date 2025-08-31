//! --------------------------------------------------------------------------------------
//! Shadow Pass (shadow_pass.rs)
//!
//! Minimal skeleton pass for directional light shadow mapping.
//! For now it only:
//!   - Requests a depth-only shadow attachment (1024x1024, D32_SFLOAT, 1x)
//!   - Begins depth-only dynamic rendering and clears to 1.0, then ends
//!   - Does not render any geometry yet
//!
//! This lets the render graph create and manage the shadow map resource and
//! scheduling without changing on-screen visuals.
//! --------------------------------------------------------------------------------------

use crate::graphics::mesh::Vertex;
use crate::graphics::shaders::{ShaderModule, ShaderStageInfo};
use crate::vulkan::attachments::{AttachmentKind, AttachmentRequest};
use crate::vulkan::render_graph::{RenderCtx, RenderPass};
use ash::vk;
use cgmath::{Matrix4, SquareMatrix};
use cgmath::{InnerSpace, Matrix};

pub struct ShadowPass {
    attachments: [AttachmentRequest; 1],
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl ShadowPass {
    pub fn new() -> Self {
        // Default request; actual size is overridden by the render graph from engine settings.
        let req = AttachmentRequest {
            kind: AttachmentKind::Shadow,
            format: vk::Format::D32_SFLOAT,
            extent: vk::Extent2D { width: 1024, height: 1024 },
            samples: vk::SampleCountFlags::TYPE_1,
        };
        Self {
            attachments: [req],
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        }
    }

    fn create_pipeline(&mut self, device: &ash::Device) -> Result<(), Box<dyn std::error::Error>> {
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
        let vs = ShaderModule::from_spv_file(device, "assets/shaders/spv/shadow_depth.vert.spv")?;
        let entry = c"main";
        let vs_stage = ShaderStageInfo { stage: vk::ShaderStageFlags::VERTEX, shader_module: vs, entry_name: entry };
        let stages = [vs_stage.to_create_info()];

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
        let dummy_viewport = vk::Viewport { x: 0.0, y: 0.0, width: 1.0, height: 1.0, min_depth: 0.0, max_depth: 1.0 };
        let dummy_scissor = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: 1, height: 1 } };
        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            p_viewports: &dummy_viewport,
            scissor_count: 1,
            p_scissors: &dummy_scissor,
            ..Default::default()
        };

        // Cull front faces to reduce peter-panning; render back faces into the map
        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            // Cull FRONT faces to render back faces into the shadow map
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
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?
        };
        self.pipeline = pipelines[0];

        // We can drop the temporary shader module now
        vs_stage.shader_module.cleanup();
        Ok(())
    }

    fn compute_light_vp(&self, ctx: &RenderCtx) -> Matrix4<f32> {
        use cgmath::{Point3, Vector3};
        use cgmath::ortho;

        const LIGHT_DISTANCE: f32 = 10.0;
        const ORTHO_EXTENT: f32 = 10.0;
        const NEAR_PLANE: f32 = 0.1;
        const FAR_PLANE: f32 = 50.0;

        let dir = Vector3::new(
            ctx.world_controls.sun_direction[0],
            ctx.world_controls.sun_direction[1],
            ctx.world_controls.sun_direction[2],
        );
        let dir = dir / dir.magnitude().max(1e-6);
        let center = Point3::new(0.0, 0.0, 0.0);
        let eye = center - dir * LIGHT_DISTANCE;
        let up = Vector3::new(0.0, 1.0, 0.0);
        let view = Matrix4::look_at_rh(eye, center, up);
        let proj_gl = ortho(-ORTHO_EXTENT, ORTHO_EXTENT, -ORTHO_EXTENT, ORTHO_EXTENT, NEAR_PLANE, FAR_PLANE);
        // Vulkan depth correction to map z from [-1,1] to [0,1]
        let zcorr = Matrix4::from_cols(
            cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
            cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
            cgmath::Vector4::new(0.0, 0.0, 0.5, 0.0),
            cgmath::Vector4::new(0.0, 0.0, 0.5, 1.0),
        );
        let proj = zcorr * proj_gl;
        proj * view
    }

    fn flatten_mat4(m: Matrix4<f32>) -> Vec<u8> {
        // Flatten to a byte array in column-major order to match GLSL.
        let mut bytes = Vec::with_capacity(16 * 4);
        let m_ref: &[f32; 16] = m.as_ref();
        for &float in m_ref {
            bytes.extend_from_slice(&float.to_ne_bytes());
        }
        bytes
    }
}

impl RenderPass for ShadowPass {
    fn execute(&mut self, ctx: &mut RenderCtx) -> Result<(), Box<dyn std::error::Error>> {
        // Acquire the per-frame shadow attachment and set up depth-only rendering.
        let cmd = ctx.frame.cmd_buf;

        // Lazy-create depth-only pipeline
        if self.pipeline == vk::Pipeline::null() {
            self.create_pipeline(&ctx.vulkan_base.device)?;
        }

        let device = &ctx.vulkan_base.device;

        let shadow_att = ctx.attachments[&AttachmentKind::Shadow];
        let res = ctx.vulkan_base.engine_settings.shadow_map_resolution;

        // Clear to far (1.0) so empty map yields no shadowing later.
        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
        };

        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(shadow_att.view)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_depth);

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: res, height: res },
            })
            .layer_count(1)
            .depth_attachment(&depth_attachment_info);

        unsafe {
            device.cmd_begin_rendering(cmd, &rendering_info);
            // Bind pipeline
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            // Dynamic viewport/scissor to match current resolution
            let viewport = vk::Viewport { x: 0.0, y: 0.0, width: res as f32, height: res as f32, min_depth: 0.0, max_depth: 1.0 };
            let scissor = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: res, height: res } };
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);

            // Draw all scene meshes with light MVP push constants
            let light_vp = self.compute_light_vp(ctx);
            for obj in &ctx.scene.objects {
                // Build model matrix and push mvp
                let model = obj.transform.model_matrix();
                let mvp = light_vp * model;
                let push = Self::flatten_mat4(mvp);
                device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    &push,
                );

                ctx.mesh_manager.meshes[obj.mesh_id].record(device, cmd);
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
            AttachmentKind::Shadow => (
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
            _ => (vk::ImageLayout::UNDEFINED, vk::AccessFlags::empty()),
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
