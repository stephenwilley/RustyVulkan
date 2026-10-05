//! ImGui graphics-pipeline creation.

use super::{ImGuiRenderer, ShaderStageInfo, VulkanBase};
use ash::vk;
use imgui::DrawVert;
use std::{error::Error, mem::offset_of};

impl ImGuiRenderer {
    // create_pipeline
    // find_memory_type
    // create_buffer
    // update_buffers
    // ----------------------------------------------------------
    /// Creates the ImGui graphics pipeline with blending and the UI vertex layout.
    /// # Arguments
    /// * `base` - The VulkanBase instance.
    /// * `extent` - The extent of the swapchain.
    /// * `color_format` - The color attachment format.
    /// * `shader_stages` - The shader stage create infos.
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or an error on failure.
    fn create_pipeline(
        &mut self,
        base: &VulkanBase,
        extent: vk::Extent2D,
        color_format: vk::Format,
        shader_stages: &[vk::PipelineShaderStageCreateInfo],
    ) -> Result<(), Box<dyn Error>> {
        let device = &base.device;
        // Vertex input: ImGui's DrawVert (pos, uv, col)
        let binding_desc = vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<DrawVert>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        };
        let attribute_descs = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(DrawVert, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(DrawVert, uv) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                offset: offset_of!(DrawVert, col) as u32,
            },
        ];
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: &binding_desc,
            vertex_attribute_description_count: attribute_descs.len() as u32,
            p_vertex_attribute_descriptions: attribute_descs.as_ptr(),
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            p_viewports: &viewport,
            scissor_count: 1,
            p_scissors: &scissor,
            ..Default::default()
        };

        // Enable dynamic viewport and scissor
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        // Disable depth testing for UI
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::FALSE,
            depth_write_enable: vk::FALSE,
            depth_compare_op: vk::CompareOp::ALWAYS,
            stencil_test_enable: vk::FALSE,
            ..Default::default()
        };

        // Enable alpha blending
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        };
        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: vk::FALSE,
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            ..Default::default()
        };

        // Push constant for projection matrix
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: std::mem::size_of::<[[f32; 4]; 4]>() as u32,
        };

        // Pipeline layout with the texture set and push constant
        let layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: &self.descriptor_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            ..Default::default()
        };
        let new_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        // Finally create the graphics pipeline
        let color_formats = [color_format];
        let rendering_info = vk::PipelineRenderingCreateInfo {
            color_attachment_count: color_formats.len() as u32,
            p_color_attachment_formats: color_formats.as_ptr(),
            // UiPass begins rendering with no depth attachment.
            depth_attachment_format: vk::Format::UNDEFINED,
            ..Default::default()
        };
        let mut pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_depth_stencil_state: &depth_stencil,
            p_color_blend_state: &color_blending,
            p_dynamic_state: &dynamic_state,
            layout: new_layout,
            render_pass: vk::RenderPass::null(),
            subpass: 0,
            ..Default::default()
        };
        pipeline_info.p_next = &rendering_info as *const _ as *const std::ffi::c_void;
        let new_pipeline = match unsafe {
            device.create_graphics_pipelines(base.pipeline_cache, &[pipeline_info], None)
        } {
            Ok(pipelines) => pipelines[0],
            Err((partial, error)) => {
                unsafe {
                    for pipeline in partial {
                        device.destroy_pipeline(pipeline, None);
                    }
                    device.destroy_pipeline_layout(new_layout, None);
                }
                return Err(error.into());
            }
        };
        // The caller waits for the GPU before rebuilding. Commit ownership only after
        // both creations succeed, leaving the old pair intact on any error.
        unsafe {
            device.destroy_pipeline(std::mem::replace(&mut self.vk_pipeline, new_pipeline), None);
            device.destroy_pipeline_layout(
                std::mem::replace(&mut self.pipeline_layout, new_layout),
                None,
            );
        }
        Ok(())
    }

    /// Rebuilds the ImGui rendering pipeline.
    /// # Arguments
    /// * `base` - The VulkanBase instance.
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or an error on failure.
    pub fn rebuild_pipeline(&mut self, base: &VulkanBase) -> Result<(), Box<dyn Error>> {
        let vert_stage = ShaderStageInfo::load(vk::ShaderStageFlags::VERTEX, "imgui.vert.spv")?;
        let frag_stage = ShaderStageInfo::load(vk::ShaderStageFlags::FRAGMENT, "imgui.frag.spv")?;
        let mut vert_module = vert_stage.module_create_info();
        let mut frag_module = frag_stage.module_create_info();
        let shader_stages = [
            vert_stage.to_create_info(&mut vert_module),
            frag_stage.to_create_info(&mut frag_module),
        ];

        let extent = base.swapchain.extent;
        let color = base.swapchain.color_format;
        self.create_pipeline(base, extent, color, &shader_stages)?;
        Ok(())
    }
}
