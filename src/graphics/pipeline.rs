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

use crate::vulkan::base::EngineSettings;

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
    /// Opaque pipelines avoid the blending read/modify/write path.
    pub alpha_blending: bool,
}

impl Pipeline {
    /// Creates a pipeline layout with alpha blending enabled, preserving the original material
    /// pipeline behaviour.  Shader modules are supplied later when the graphics pipeline is built.
    /// # Arguments
    /// * `device` - The Vulkan logical device to use for creating the pipeline.
    /// * `set_layouts` - The descriptor set layouts to use for the pipeline.
    /// * `depth_write` - Whether depth writing is enabled for this pipeline.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - Returns the initialized `Pipeline` on success, or an error on failure.
    /// # Errors
    /// * Returns an error if the pipeline layout cannot be created.
    pub fn new(
        device: &ash::Device,
        set_layouts: &[vk::DescriptorSetLayout],
        depth_write: bool
    ) -> Result<Self, Box<dyn Error>> {
        Self::new_with_blending(device, set_layouts, depth_write, true)
    }

    /// Creates a pipeline for geometry whose fragment shader always writes alpha one.
    /// Dense grass and reeds are opaque ribbons, so they do not need blend state.
    pub fn new_opaque(
        device: &ash::Device,
        set_layouts: &[vk::DescriptorSetLayout],
        depth_write: bool,
    ) -> Result<Self, Box<dyn Error>> {
        Self::new_with_blending(device, set_layouts, depth_write, false)
    }

    fn new_with_blending(
        device: &ash::Device,
        set_layouts: &[vk::DescriptorSetLayout],
        depth_write: bool,
        alpha_blending: bool,
    ) -> Result<Self, Box<dyn Error>> {
        // 1 - Define a PushConstantRange covering 2 4×4 MVP matrices (16 floats = 64 bytes) for MV and MVP, a
        // light position vector (3 floats = 12 bytes) and a light intensity float (1 float = 4 bytes)
        // Reserve space for two mat4 (mvp, mv) plus a vec4 for UV tiling (xy used, zw padding)
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset:      0,
            size:        (std::mem::size_of::<[[f32; 4]; 4]>()
                        + std::mem::size_of::<[[f32; 4]; 4]>()
                        + std::mem::size_of::<[f32; 4]>()) as u32,
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
            alpha_blending,
        })
    }

    /// Loads shaders, ties them to the given formats/extent, and creates the pipeline.
    /// # Arguments
    /// * `device` - The Vulkan logical device.
    /// * `pipeline_cache` - Shared cache that reuses pipeline compilation work.
    /// * `extent` - The extent of the swapchain.
    /// * `color_format` - The color attachment format.
    /// * `depth_format` - The depth attachment format.
    /// * `shader_infos` - The shader stage information.
    /// * `wireframe` - Whether to enable wireframe mode.
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or an error on failure.
    pub fn create_graphics_pipeline(
        &mut self,
        device: &ash::Device,
        pipeline_cache: vk::PipelineCache,
        extent: vk::Extent2D,
        color_format: vk::Format,
        depth_format: vk::Format,
        shader_infos: &[&ShaderStageInfo],
        engine_settings: &EngineSettings,
    ) -> Result<(), Box<dyn Error>> {
        let binding_descs = [Vertex::binding_description()];
        let attribute_descs = Vertex::attribute_descriptions();
        self.create_graphics_pipeline_with_vertex_input(
            device,
            pipeline_cache,
            extent,
            color_format,
            depth_format,
            shader_infos,
            engine_settings,
            &binding_descs,
            &attribute_descs,
            vk::CullModeFlags::BACK,
        )
    }

    /// Creates a graphics pipeline with caller-provided vertex and instance bindings.
    ///
    /// Most materials use [`Vertex`]'s single binding through
    /// [`Self::create_graphics_pipeline`].  Instanced renderers such as grass add a
    /// second, per-instance binding while sharing all the usual Vulkan state.
    #[allow(clippy::too_many_arguments)]
    pub fn create_graphics_pipeline_with_vertex_input(
        &mut self,
        device: &ash::Device,
        pipeline_cache: vk::PipelineCache,
        extent: vk::Extent2D,
        color_format: vk::Format,
        depth_format: vk::Format,
        shader_infos: &[&ShaderStageInfo],
        engine_settings: &EngineSettings,
        binding_descs: &[vk::VertexInputBindingDescription],
        attribute_descs: &[vk::VertexInputAttributeDescription],
        cull_mode: vk::CullModeFlags,
    ) -> Result<(), Box<dyn Error>> {
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
            polygon_mode: if engine_settings.wireframe {
                vk::PolygonMode::LINE
            } else {
                vk::PolygonMode::FILL
            },
            line_width: 1.0,
            cull_mode,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            ..Default::default()
        };
        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::from_raw(engine_settings.msaa_samples),
            sample_shading_enable: vk::FALSE,
            ..Default::default()
        };

        // Most imported materials use alpha blending; opaque specialised geometry can skip it.
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
            blend_enable: if self.alpha_blending { vk::TRUE } else { vk::FALSE },
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

        let color_formats = [color_format];
        let rendering_info = vk::PipelineRenderingCreateInfo {
            color_attachment_count: color_formats.len() as u32,
            p_color_attachment_formats: color_formats.as_ptr(),
            depth_attachment_format: depth_format,
            ..Default::default()
        };

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo {
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

        println!("🛠️ Graphics pipeline created with {} stages", pipelines.len());
        self.vk_pipeline = pipelines[0];
        Ok(())
    }

    /// Recreate the pipeline
    /// # Arguments
    /// * `device` - The Vulkan logical device.
    /// * `pipeline_cache` - Shared cache that reuses pipeline compilation work.
    /// * `extent` - The extent of the swapchain.
    /// * `color_format` - The color attachment format.
    /// * `depth_format` - The depth attachment format.
    /// * `shader_infos` - The shader stage information.
    /// * `wireframe` - Whether to enable wireframe mode.
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or an error on failure.
    pub fn recreate(
        &mut self,
        device: &ash::Device,
        pipeline_cache: vk::PipelineCache,
        extent: vk::Extent2D,
        color_format: vk::Format,
        depth_format: vk::Format,
        shader_infos: &[&ShaderStageInfo],
        engine_settings: &EngineSettings
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            // The render graph has already waited for all submitted command buffers.
            device.destroy_pipeline(self.vk_pipeline, None);
        }
        self.create_graphics_pipeline(
            device,
            pipeline_cache,
            extent,
            color_format,
            depth_format,
            shader_infos,
            engine_settings
        )?;
        Ok(())
    }

    /// Recreates a pipeline that uses custom vertex/instance bindings.
    #[allow(clippy::too_many_arguments)]
    pub fn recreate_with_vertex_input(
        &mut self,
        device: &ash::Device,
        pipeline_cache: vk::PipelineCache,
        extent: vk::Extent2D,
        color_format: vk::Format,
        depth_format: vk::Format,
        shader_infos: &[&ShaderStageInfo],
        engine_settings: &EngineSettings,
        binding_descs: &[vk::VertexInputBindingDescription],
        attribute_descs: &[vk::VertexInputAttributeDescription],
        cull_mode: vk::CullModeFlags,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            device.destroy_pipeline(self.vk_pipeline, None);
        }
        self.create_graphics_pipeline_with_vertex_input(
            device,
            pipeline_cache,
            extent,
            color_format,
            depth_format,
            shader_infos,
            engine_settings,
            binding_descs,
            attribute_descs,
            cull_mode,
        )
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
