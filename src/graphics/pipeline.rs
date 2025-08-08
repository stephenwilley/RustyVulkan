//! --------------------------------------------------------------------------------------
//! Pipeline Module (pipeline.rs)
//!
//! Created: July 2025  
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module defines the `Pipeline` type, which encapsulates creation and management of
//! the Vulkan graphics pipeline. It handles:
//!   • Loading SPIR-V shader stages (vertex & fragment)  
//!   • Configuring fixed-function state (vertex input, input assembly, viewport & scissor
//!     rasterizer, multisampling, color blending)  
//!   • Creating the `vk::PipelineLayout` and `vk::Pipeline` objects  
//!   • Providing a `cleanup` method to destroy the pipeline layout when no longer needed  
//!
//! Usage:
//!   1. `Pipeline::new`            - initializes the pipeline layout  
//!   2. `create_graphics_pipeline` - stitches together all shader and fixed-function state into a usable pipeline  
//!   3. `cleanup`                  - tears down the pipeline layout  
//!
//! This keeps all graphics-pipeline logic centralized and reusable by `VulkanBase`.
//! --------------------------------------------------------------------------------------

use ash::vk;
use std::error::Error;

use super::shaders::ShaderStageInfo;
use super::mesh::Vertex;

/// Represents the Vulkan graphics pipeline, including shader modules and layout.
/// It encapsulates the shader modules used for vertex and fragment stages,
/// and the pipeline layout used for rendering.
/// It is responsible for creating the graphics pipeline and managing its resources.
/// This struct is used by the `VulkanBase` to set up the rendering pipeline.
pub struct Pipeline {
    pub vk_layout: vk::PipelineLayout,
    pub vk_pipeline: vk::Pipeline,
    pub depth_write: bool,
}

impl Pipeline {
    /// The Vulkan pipeline layout, defining the interface between shaders and resources.    
    /// Creates a new `Pipeline` instance, initializing the pipeline layout and shader modules.
    /// # Arguments
    /// * `device` - The Vulkan logical device to use for creating the pipeline.
    /// * `set_layouts` - The descriptor set layouts to use for the pipeline.
    /// * `depth_write` - Whether depth writing is enabled for this pipeline.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - Returns the initialized `Pipeline` on success, or an error on failure.
    /// # Errors
    /// * Returns an error if the shader modules cannot be created or if the pipeline layout cannot be created.
    /// # Notes
    /// * The shader modules are loaded from SPIR-V files located in the `assets/shaders` directory.
    pub fn new(
        device: &ash::Device,
        set_layouts: &[vk::DescriptorSetLayout],
        depth_write: bool
    ) -> Result<Self, Box<dyn Error>> {
        // 1 - Define a PushConstantRange covering 2 4×4 MVP matrices (16 floats = 64 bytes) for MV and MVP, a
        // light position vector (3 floats = 12 bytes) and a light intensity float (1 float = 4 bytes)
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset:      0,
            size:        std::mem::size_of::<[[f32; 4]; 4]>() as u32
                       + std::mem::size_of::<[[f32; 4]; 4]>() as u32
                            + std::mem::size_of::<[f32; 3]>() as u32
                                 + std::mem::size_of::<f32>() as u32,
        };

        // 2 - Build your PipelineLayoutCreateInfo with that push-constant baked in and descriptor set layouts
        let layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count:        set_layouts.len() as u32,
            p_set_layouts:           set_layouts.as_ptr(),
            push_constant_range_count: 1,
            p_push_constant_ranges:    &push_constant_range as *const _,
            ..Default::default()
        };

        // 3 - Create the layout as before
        let vk_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };
        println!("🛠️ Pipeline layout created with push‐constant support");

        Ok(Self {
            vk_layout,
            vk_pipeline: vk::Pipeline::null(),
            depth_write,
        })
    }

    /// Loads shaders, ties them to the given render_pass/extent, and creates the pipeline.
    /// # Arguments
    /// * `device` - The Vulkan logical device.
    /// * `extent` - The extent of the swapchain.
    /// * `render_pass` - The render pass to use.
    /// * `shader_infos` - The shader stage information.
    /// * `wireframe` - Whether to enable wireframe mode.
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or an error on failure.
    pub fn create_graphics_pipeline(
        &mut self,
        device: &ash::Device,
        extent: vk::Extent2D,
        render_pass: vk::RenderPass,
        shader_infos: &[&ShaderStageInfo],
        wireframe: bool,
    ) -> Result<(), Box<dyn Error>> {
        let binding_descs   = [Vertex::binding_description()];
        let attribute_descs = Vertex::attribute_descriptions();

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count:   binding_descs.len() as u32,
            p_vertex_binding_descriptions:      binding_descs.as_ptr(),
            vertex_attribute_description_count: attribute_descs.len() as u32,
            p_vertex_attribute_descriptions:    attribute_descs.as_ptr(),
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

        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: if wireframe {
                vk::PolygonMode::LINE
            } else {
                vk::PolygonMode::FILL
            },
            line_width: 1.0,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            ..Default::default()
        };
        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            ..Default::default()
        };

        // Enable alpha blending for transparency
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
            blend_enable: vk::TRUE,
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
        };

        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: vk::FALSE,
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            ..Default::default()
        };

        let shader_stage_create_infos: Vec<_> = shader_infos
            .iter()
            .map(|s| s.to_create_info())
            .collect();

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable:     vk::TRUE,
            depth_write_enable:    if self.depth_write { vk::TRUE } else { vk::FALSE },
            depth_compare_op:      vk::CompareOp::LESS,
            // stencil is off for now—
            stencil_test_enable:   vk::FALSE,
            front:                 Default::default(),
            back:                  Default::default(),
            ..Default::default()
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: shader_stage_create_infos.len() as u32,
            p_stages: shader_stage_create_infos.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_color_blend_state: &color_blending,
            p_depth_stencil_state: &depth_stencil,
            layout: self.vk_layout,
            render_pass,
            subpass: 0,
            ..Default::default()
        };

        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?
        };

        println!("🛠️ Graphics pipeline created with {} stages", pipelines.len());
        self.vk_pipeline = pipelines[0];
        Ok(())
    }

    /// Recreate the pipeline
    /// # Arguments
    /// * `device` - The Vulkan logical device.
    /// * `extent` - The extent of the swapchain.
    /// * `render_pass` - The render pass to use.
    /// * `shader_infos` - The shader stage information.
    /// * `wireframe` - Whether to enable wireframe mode.
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or an error on failure.
    pub fn recreate(
        &mut self,
        device: &ash::Device,
        extent: vk::Extent2D,
        render_pass: vk::RenderPass,
        shader_infos: &[&ShaderStageInfo],
        wireframe: bool
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            device.device_wait_idle().expect("Failed to wait device idle");
            device.destroy_pipeline(self.vk_pipeline, None);
        }
        self.create_graphics_pipeline(
            device,
            extent,
            render_pass,
            shader_infos,
            wireframe
        )?;
        Ok(())
    }

    /// Cleans up the pipeline resources, destroying the shader modules and pipeline layout.
    /// # Arguments
    /// * `device` - The Vulkan logical device to use for destroying the resources.
    /// # Notes
    /// * This method should be called when the pipeline is no longer needed, such as during
    ///   application shutdown or when the pipeline is being recreated.
    pub fn cleanup(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline_layout(self.vk_layout, None);
            device.destroy_pipeline(self.vk_pipeline, None);
        }
    }
}