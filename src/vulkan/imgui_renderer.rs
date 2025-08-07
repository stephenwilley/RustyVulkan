//! --------------------------------------------------------------------------------------
//! ImGui Renderer Module (imgui_renderer.rs)
//_!
//! Created: July 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//_!
//! This module defines `ImGuiRenderer`, which encapsulates the Vulkan setup and rendering
//! logic for Dear ImGui. It handles:
//!   • Uploading the font atlas and creating associated Vulkan resources (image, view, sampler)
//!   • Creating descriptor set layout, pool, and descriptor set for UI textures
//!   • Building the ImGui-specific graphics pipeline and pipeline layout
//!   • Recording draw commands for ImGui draw data into command buffers
//!   • Cleaning up all ImGui-related Vulkan resources on teardown
//_!
//! Usage:
//!   1. `ImGuiRenderer::new(base: &mut VulkanBase, atlas: &FontAtlasTexture) -> Self`  
//!   2. `renderer.render(base: &VulkanBase, cmd_buf: vk::CommandBuffer, draw_data: &imgui::DrawData)`  
//!   3. `renderer.cleanup(base: &mut VulkanBase)`
//! 
//! --------------------------------------------------------------------------------------

use ash::vk;
use std::error::Error;
use crate::graphics::shaders::{ShaderModule, ShaderStageInfo};
use crate::vulkan::base::VulkanBase;
use imgui::{Context as ImGuiContext, FontConfig, FontSource};
use imgui::FontAtlasTexture;
use memoffset::offset_of;
use imgui::DrawVert;
use imgui::DrawData;
use bytemuck;
use imgui::DrawIdx;

pub struct ImGuiRenderer {
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool:       vk::DescriptorPool,
    descriptor_set:        vk::DescriptorSet,
    pipeline_layout:       vk::PipelineLayout,
    pub vk_pipeline:       vk::Pipeline,
    pub font_sampler:      Option<vk::Sampler>,
    pub font_image:        Option<vk::Image>,
    pub font_image_memory: Option<vk::DeviceMemory>,
    pub font_image_view:   Option<vk::ImageView>,
    pub vertex_buffer:        vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub vertex_buffer_size:   vk::DeviceSize,
    pub index_buffer:         vk::Buffer,
    pub index_buffer_memory:  vk::DeviceMemory,
    pub index_buffer_size:    vk::DeviceSize,
    device:                ash::Device,
    mem_props:             vk::PhysicalDeviceMemoryProperties,
    vert_stage:            Option<ShaderStageInfo>,
    frag_stage:            Option<ShaderStageInfo>,
}

impl ImGuiRenderer {
    // ----------------------------------------------------------
    // 1 - Font Atlas Upload
    /// create_staging buffer
    /// fill_staging_buffer
    /// create_font_image
    /// copy_buffer_to_image
    /// create_image_view
    /// create_sampler
    // ----------------------------------------------------------
    fn create_staging_buffer(base: &VulkanBase, atlas: &FontAtlasTexture)
        -> (vk::Buffer, vk::DeviceMemory)
    {
        let device = &base.device;
        let size = (atlas.width * atlas.height * 4) as vk::DeviceSize;
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let staging_buffer = unsafe { device.create_buffer(&buffer_info, None).unwrap() };
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let mem_props = unsafe { base.instance.get_physical_device_memory_properties(base.physical_device) };
        let mem_type_index = Self::find_memory_type(
            mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &mem_props,
        );
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_requirements.size,
            memory_type_index: mem_type_index,
            ..Default::default()
        };
        let staging_buffer_memory = unsafe { device.allocate_memory(&alloc_info, None).unwrap() };
        unsafe { device.bind_buffer_memory(staging_buffer, staging_buffer_memory, 0).unwrap() };

        (staging_buffer, staging_buffer_memory)
    }

    /// Maps that memory and memcpy’s the atlas into it.
    fn fill_staging_buffer(base: &VulkanBase, staging_mem: vk::DeviceMemory, atlas: &FontAtlasTexture) {
        let device = &base.device;
        let size = (atlas.width * atlas.height * 4) as vk::DeviceSize;
        let data_ptr = unsafe {
            device.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty()).unwrap()
        };
        unsafe {
            std::ptr::copy_nonoverlapping(atlas.data.as_ptr(), data_ptr as *mut u8, atlas.data.len());
            device.unmap_memory(staging_mem);
        }
    }

    /// Creates an OPTIMAL‐tiling vk::Image + device‐local memory for the atlas.
    fn create_font_image(base: &VulkanBase, width: u32, height: u32)
        -> (vk::Image, vk::DeviceMemory)
    {
        let device = &base.device;
        let image_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D { width, height, depth: 1 },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let font_image = unsafe { device.create_image(&image_info, None).unwrap() };
        let font_req = unsafe { device.get_image_memory_requirements(font_image) };
        let mem_props = unsafe { base.instance.get_physical_device_memory_properties(base.physical_device) };
        let font_mem_type = Self::find_memory_type(
            font_req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &mem_props,
        );
        let font_alloc = vk::MemoryAllocateInfo {
            allocation_size: font_req.size,
            memory_type_index: font_mem_type,
            ..Default::default()
        };
        let font_image_memory = unsafe { device.allocate_memory(&font_alloc, None).unwrap() };
        unsafe { device.bind_image_memory(font_image, font_image_memory, 0).unwrap() };

        (font_image, font_image_memory)
    }

    /// Records and submits a one-time cmd buffer that:
    /// 1) TRANSITIONS UNDEFINED → TRANSFER_DST_OPTIMAL  
    /// 2) COPIES the staging buffer into the VkImage  
    /// 3) TRANSITIONS → SHADER_READ_ONLY_OPTIMAL  
    fn copy_buffer_to_image(base: &VulkanBase,
                            staging_buffer: vk::Buffer,
                            image: vk::Image,
                            width: u32,
                            height: u32)
    {
        let device = &base.device;
        let font_image = image;
        let cmd_alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: base.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        let cmd_buffer = unsafe { device.allocate_command_buffers(&cmd_alloc_info).unwrap()[0] };
        let cmd_begin = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe { device.begin_command_buffer(cmd_buffer, &cmd_begin).unwrap() };

        // Transition to TRANSFER_DST_OPTIMAL
        let barrier1 = vk::ImageMemoryBarrier {
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: font_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            ..Default::default()
        };
        unsafe {
            device.cmd_pipeline_barrier(
                cmd_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier1],
            );
        }

        // Copy buffer to image
        let copy_region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D { width, height, depth: 1 },
        };
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd_buffer,
                staging_buffer,
                font_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );
        }

        // Transition to SHADER_READ_ONLY_OPTIMAL
        let barrier2 = vk::ImageMemoryBarrier {
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: font_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            ..Default::default()
        };
        unsafe {
            device.cmd_pipeline_barrier(
                cmd_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier2],
            );
        }

        unsafe { device.end_command_buffer(cmd_buffer).unwrap() };

        // Submit and wait
        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd_buffer,
            ..Default::default()
        };
        unsafe {
            device.queue_submit(base.graphics_queue, &[submit], vk::Fence::null()).unwrap();
            device.queue_wait_idle(base.graphics_queue).unwrap();
        }

        // 5) Clean up staging and command buffer
        unsafe {
            device.free_command_buffers(base.command_pool, &[cmd_buffer]);
        }
    }

    /// Make a 2D ImageView for the font atlas.
    fn create_image_view(base: &VulkanBase, image: vk::Image) -> vk::ImageView {
        let device = &base.device;
        let view_info = vk::ImageViewCreateInfo {
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };
        unsafe { device.create_image_view(&view_info, None).unwrap() }
    }

    /// Create the linear, clamp‐to‐edge sampler.
    fn create_sampler(base: &VulkanBase) -> vk::Sampler {
        let device = &base.device;
        let sampler_info = vk::SamplerCreateInfo {
            mag_filter:             vk::Filter::LINEAR,
            min_filter:             vk::Filter::LINEAR,
            address_mode_u:         vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v:         vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w:         vk::SamplerAddressMode::CLAMP_TO_EDGE,
            mipmap_mode:            vk::SamplerMipmapMode::LINEAR,
            mip_lod_bias:           0.0,
            anisotropy_enable:      vk::FALSE,
            max_anisotropy:         1.0,
            min_lod:                0.0,
            max_lod:                1.0,
            border_color:           vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,
            ..Default::default()
        };
        unsafe { device.create_sampler(&sampler_info, None).unwrap() }
    }

    // ----------------------------------------------------------
    // 2 - Descriptor Set Lifecycle
    // create_imgui_descriptor_set_layout
    // create_imgui_descriptor_pool
    // allocate_imgui_descriptor_set
    // write_descriptor_set
    // Aggregated in `init_imgui_descriptor_resources`
    // ----------------------------------------------------------
    /// Helper: create descriptor set layout for ImGui font atlas
    fn create_imgui_descriptor_set_layout(base: &VulkanBase) -> vk::DescriptorSetLayout {
        let bindings = [vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        }];
        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        unsafe { base.device.create_descriptor_set_layout(&layout_info, None).unwrap() }
    }

    /// Helper: create descriptor pool for one combined image sampler
    fn create_imgui_descriptor_pool(base: &VulkanBase) -> vk::DescriptorPool {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo {
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            max_sets: 1,
            ..Default::default()
        };
        unsafe { base.device.create_descriptor_pool(&pool_info, None).unwrap() }
    }

    /// Helper: allocate a descriptor set for ImGui
    fn allocate_imgui_descriptor_set(&self, base: &VulkanBase) -> vk::DescriptorSet {
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };
        unsafe { base.device.allocate_descriptor_sets(&alloc_info).unwrap()[0] }
    }
    
    /// Updates the previously-allocated descriptor set so binding 0 points at our atlas view+sampler.
    fn write_descriptor_set(&self, base: &VulkanBase) {
        let device = &base.device;
        let image_info = vk::DescriptorImageInfo {
            sampler:      self.font_sampler.unwrap(),
            image_view:   self.font_image_view.unwrap(),
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let descriptor_write = vk::WriteDescriptorSet {
            dst_set:           self.descriptor_set,
            dst_binding:       0,
            dst_array_element: 0,
            descriptor_count:  1,
            descriptor_type:   vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info:      &image_info,
            ..Default::default()
        };
        unsafe {
            device.update_descriptor_sets(&[descriptor_write], &[]);
        }
    }

    /// Initialize ImGui descriptor resources
    fn init_imgui_descriptor_resources(&mut self, base: &VulkanBase) {
        // Layout
        self.descriptor_set_layout = Self::create_imgui_descriptor_set_layout(base);
        // Pool
        self.descriptor_pool = Self::create_imgui_descriptor_pool(base);
        // Allocate
        self.descriptor_set = self.allocate_imgui_descriptor_set(base);
        // The write to bind image+sampler will happen later when fonts are uploaded
    }

    // ----------------------------------------------------------
    // 3 - Rendering Pipeline and Dynamic Buffers
    // load_shaders
    // create_pipeline
    // find_memory_type
    // create_buffer
    // update_buffers
    // ----------------------------------------------------------
    /// Load the ImGui shaders
    fn load_shaders(&mut self, base: &mut VulkanBase) -> (ShaderStageInfo, ShaderStageInfo) {
        let entry_name = c"main";
        // Load vertex shader
        let vert_module = ShaderModule::from_spv_file(&base.device, "assets/shaders/spv/imgui.vert.spv")
            .expect("Failed to load imgui.vert.spv");
        // Load fragment shader
        let frag_module = ShaderModule::from_spv_file(&base.device, "assets/shaders/spv/imgui.frag.spv")
            .expect("Failed to load imgui.frag.spv");
        let vert_stage = ShaderStageInfo {
            stage: vk::ShaderStageFlags::VERTEX,
            shader_module: vert_module,
            entry_name,
        };
        let frag_stage = ShaderStageInfo {
            stage: vk::ShaderStageFlags::FRAGMENT,
            shader_module: frag_module,
            entry_name,
        };

        (vert_stage, frag_stage)
    }

    /// Creates the ImGui graphics pipeline with blending and the UI vertex layout.
    fn create_pipeline(
        &mut self,
        base: &VulkanBase,
        extent: vk::Extent2D,
        render_pass: vk::RenderPass,
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
            depth_test_enable:       vk::FALSE,
            depth_write_enable:      vk::FALSE,
            depth_compare_op:        vk::CompareOp::ALWAYS,
            stencil_test_enable:     vk::FALSE,
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

        // Pipeline layout with descriptor set and push constant
        let layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: &self.descriptor_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            ..Default::default()
        };
        self.pipeline_layout = unsafe {
            device.create_pipeline_layout(&layout_info, None).unwrap()
        };

        // Finally create the graphics pipeline
        let pipeline_info = vk::GraphicsPipelineCreateInfo {
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
            layout: self.pipeline_layout,
            render_pass,
            subpass: 0,
            ..Default::default()
        };
        self.vk_pipeline = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                  .map_err(|e| e.1).unwrap()[0]
        };
        Ok(())
    }

    // Helper to choose memory type
    fn find_memory_type(
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
    ) -> u32 {
        for i in 0..mem_props.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && mem_props.memory_types[i as usize].property_flags.contains(properties)
            {
                return i;
            }
        }
        panic!("Failed to find suitable memory type!");
    }

    /// Creates a Vulkan buffer and allocates its memory.
    /// Returns (buffer, buffer_memory).
    fn create_buffer(&self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let device = &self.device;
        // 1) Create the buffer handle
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = unsafe { device.create_buffer(&buffer_info, None).unwrap() };
        // 2) Allocate memory for the buffer
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_type_index = Self::find_memory_type(
            mem_requirements.memory_type_bits,
            properties,
            &self.mem_props,
        );
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_requirements.size,
            memory_type_index: mem_type_index,
            ..Default::default()
        };
        let buffer_memory = unsafe { device.allocate_memory(&alloc_info, None).unwrap() };
        // 3) Bind buffer and memory together
        unsafe {
            device.bind_buffer_memory(buffer, buffer_memory, 0).unwrap();
        }
        (buffer, buffer_memory)
    }

     /// Ensures the vertex and index buffers are large enough and uploads ImGui draw data into them.
    pub fn update_buffers(&mut self, draw_data: &DrawData) {
        let device = &self.device;
        // Total vertex and index data sizes
        let vertex_size = (draw_data.total_vtx_count as usize * std::mem::size_of::<DrawVert>()) as vk::DeviceSize;
        let index_size  = (draw_data.total_idx_count as usize * std::mem::size_of::<DrawIdx>()) as vk::DeviceSize;

        // Resize vertex buffer if needed
        if vertex_size > self.vertex_buffer_size {
            if self.vertex_buffer != vk::Buffer::null() {
                unsafe {
                    device.device_wait_idle().expect("Failed to wait device idle");
                    device.destroy_buffer(self.vertex_buffer, None);
                    device.free_memory(self.vertex_buffer_memory, None);
                }
            }
            let (buf, mem) = self.create_buffer(vertex_size,
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            self.vertex_buffer = buf;
            self.vertex_buffer_memory = mem;
            self.vertex_buffer_size = vertex_size;
        }

        // Resize index buffer if needed
        if index_size > self.index_buffer_size {
            if self.index_buffer != vk::Buffer::null() {
                unsafe {
                    device.destroy_buffer(self.index_buffer, None);
                    device.free_memory(self.index_buffer_memory, None);
                }
            }
            let (buf, mem) = self.create_buffer(index_size,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            self.index_buffer = buf;
            self.index_buffer_memory = mem;
            self.index_buffer_size = index_size;
        }

        // Map and copy vertex data
        unsafe {
            let vtx_ptr = device.map_memory(self.vertex_buffer_memory, 0, vertex_size, vk::MemoryMapFlags::empty()).unwrap();
            let mut offset = 0;
            for draw_list in draw_data.draw_lists() {
                let src = draw_list.vtx_buffer();
                let byte_len = std::mem::size_of_val(src);
                std::ptr::copy_nonoverlapping(
                    src.as_ptr() as *const u8,
                    (vtx_ptr as *mut u8).add(offset),
                    byte_len,
                );
                offset += byte_len;
            }
            device.unmap_memory(self.vertex_buffer_memory);

            // Map and copy index data
            let idx_ptr = device.map_memory(self.index_buffer_memory, 0, index_size, vk::MemoryMapFlags::empty()).unwrap();
            let mut idx_offset = 0;
            for draw_list in draw_data.draw_lists() {
                let src = draw_list.idx_buffer();
                let byte_len = std::mem::size_of_val(src);
                std::ptr::copy_nonoverlapping(
                    src.as_ptr() as *const u8,
                    (idx_ptr as *mut u8).add(idx_offset),
                    byte_len,
                );
                idx_offset += byte_len;
            }
            device.unmap_memory(self.index_buffer_memory);
        }
    }

    pub fn rebuild_pipeline(&mut self, base: &mut VulkanBase) -> Result<(), Box<dyn Error>> {
        unsafe {
            base.device.device_wait_idle().expect("Failed to wait device idle");
            base.device.destroy_pipeline(self.vk_pipeline, None);
            base.device.destroy_pipeline_layout(self.pipeline_layout, None);
            if self.vert_stage.is_some() || self.frag_stage.is_some() {
                self.vert_stage.as_ref().unwrap().shader_module.cleanup();
                self.frag_stage.as_ref().unwrap().shader_module.cleanup();
            }
        }
        let (vert_stage, frag_stage) = Self::load_shaders(self, base);
        let shader_stages = [
            vert_stage.to_create_info(),
            frag_stage.to_create_info(),
        ];

        let extent = base.swapchain.extent;
        let render_pass = base.swapchain.render_pass;
        self.create_pipeline(base, extent, render_pass, &shader_stages)?;

        // 5) Store stages for potential future reload
        self.vert_stage = Some(vert_stage);
        self.frag_stage = Some(frag_stage);
        Ok(())
    }

    /// Creates a new ImGuiRenderer instance.
    /// # Arguments
    /// * `base` - The VulkanBase instance to use for resource creation.
    /// * `imgui` - The ImGui context to initialize the renderer with.
    /// # Returns
    /// A new ImGuiRenderer instance with all resources initialized.
    pub fn new(base: &mut VulkanBase, imgui: &mut ImGuiContext) -> Self {
        let instance        = base.instance.clone();
        let physical_device = base.physical_device;
        let device          = base.device.clone();
        let mem_props = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        // 0) Load default font atlas
        imgui.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(FontConfig {
                rasterizer_multiply: 1.0,
                ..FontConfig::default()
            }),
        }]);
        let atlas = imgui.fonts().build_rgba32_texture();
        // 1) Create staging buffer for font atlas
        let (staging_buffer, staging_buffer_memory) = Self::create_staging_buffer(base, &atlas);
        // 2) Map memory and copy font atlas data into it
        Self::fill_staging_buffer(base, staging_buffer_memory, &atlas);
        // 3) Create font image with device-local memory
        let (font_image, font_image_memory) = Self::create_font_image(base, atlas.width, atlas.height);
        // 4) Bind the staging buffer memory to the font image
        Self::copy_buffer_to_image(base, staging_buffer, font_image, atlas.width, atlas.height);
        // 5) Clean up staging buffer memory
        unsafe {
            base.device.free_memory(staging_buffer_memory, None);
            base.device.destroy_buffer(staging_buffer, None);
        }
        // 6) Create an ImageView for the font atlas
        let font_image_view = Self::create_image_view(base, font_image);
        // 7) Create a Sampler for the font atlas
        let font_sampler = Self::create_sampler(base);

        let mut imgui_renderer = Self {
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            vk_pipeline: vk::Pipeline::null(),
            font_sampler: Some(font_sampler),
            font_image: Some(font_image),
            font_image_memory: Some(font_image_memory),
            font_image_view: Some(font_image_view),
            vertex_buffer: vk::Buffer::null(),
            vertex_buffer_memory: vk::DeviceMemory::null(),
            vertex_buffer_size: 0,
            index_buffer: vk::Buffer::null(),
            index_buffer_memory: vk::DeviceMemory::null(),
            index_buffer_size: 0,
            device,
            mem_props,
            vert_stage: None,
            frag_stage: None,
        };

        // 8) Initialize descriptor layout, pool, and set for ImGui
        imgui_renderer.init_imgui_descriptor_resources(base);
        // 9) Write the descriptor set to bind the font atlas image and sampler
        imgui_renderer.write_descriptor_set(base);

        // 10) Loads the shaders and build the pipeline
        imgui_renderer.rebuild_pipeline(base)
            .expect("Failed to rebuild imgui pipeline");

        imgui_renderer
    }
    
    /// Cleans up ImGui Vulkan resources created by this renderer.
    /// # Arguments
    /// * `base` - The VulkanBase instance to use for resource cleanup.
    ///   This function destroys all Vulkan resources associated with the ImGui renderer,
    ///   including the font image, sampler, descriptor set layout, descriptor pool, and pipeline.
    ///   It should be called when the ImGui renderer is no longer needed.
    pub fn cleanup(&self, base: VulkanBase) {
        unsafe {
            // Destroy dynamic ImGui vertex/index buffers
            if self.vertex_buffer != vk::Buffer::null() {
                base.device.destroy_buffer(self.vertex_buffer, None);
            }
            if self.vertex_buffer_memory != vk::DeviceMemory::null() {
                base.device.free_memory(self.vertex_buffer_memory, None);
            }
            if self.index_buffer != vk::Buffer::null() {
                base.device.destroy_buffer(self.index_buffer, None);
            }
            if self.index_buffer_memory != vk::DeviceMemory::null() {
                base.device.free_memory(self.index_buffer_memory, None);
            }
            // Destroy sampler, view, image, and free memory
            if let Some(sampler) = self.font_sampler {
                base.device.destroy_sampler(sampler, None);
            }
            if let Some(view) = self.font_image_view {
                base.device.destroy_image_view(view, None);
            }
            if let Some(image) = self.font_image {
                base.device.destroy_image(image, None);
            }
            if let Some(mem) = self.font_image_memory {
                base.device.free_memory(mem, None);
            }
            self.vert_stage.as_ref().unwrap().shader_module.cleanup();
            self.frag_stage.as_ref().unwrap().shader_module.cleanup();
            // Destroy descriptor pool and layout
            base.device.destroy_descriptor_pool(self.descriptor_pool, None);
            base.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            // Destroy pipeline and layout when created (skip if null)
            if self.vk_pipeline != vk::Pipeline::null() {
                base.device.destroy_pipeline(self.vk_pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                base.device.destroy_pipeline_layout(self.pipeline_layout, None);
            }
        }
    }

    /// Records ImGui draw commands: bind pipeline, descriptor set, and push constants.
    /// # Arguments
    /// * `device` - The Vulkan device to use for command recording.
    /// * `cmd_buf` - The command buffer to record the ImGui draw commands into.
    /// * `draw_data` - The ImGui draw data containing vertex and index information.
    ///   This function performs the following steps:
    ///   1. Updates the vertex and index buffers with the latest ImGui draw data.
    ///   2. Binds the vertex and index buffers to the command buffer.
    ///   3. Binds the ImGui graphics pipeline and descriptor set.
    ///   4. Sets the dynamic viewport based on the ImGui display size.
    ///   5. Computes the orthographic projection matrix for ImGui.
    ///   6. Iterates through the ImGui draw lists and issues draw calls for each command.
    ///
    ///   It handles scissor rectangles and indexed drawing based on ImGui's clip rects.
    ///   If there is no ImGui draw data (total vertex or index count is zero),
    ///   it simply returns without rendering anything.
    pub fn render(
        &mut self,
        device: &ash::Device,
        cmd_buf: vk::CommandBuffer,
        draw_data: &DrawData,
    ) {
        // If there is nothing to draw, skip UI rendering
        if draw_data.total_vtx_count == 0 || draw_data.total_idx_count == 0 {
            return;
        }
        // Account for HiDPI: logical→physical scale
        let fb_scale = draw_data.framebuffer_scale;
        // 1) Ensure buffers are up-to-date with ImGui draw data
        self.update_buffers(draw_data);

        // 2) Bind vertex and index buffers
        unsafe {
            device.cmd_bind_vertex_buffers(cmd_buf, 0, &[self.vertex_buffer], &[0]);
            device.cmd_bind_index_buffer(cmd_buf, self.index_buffer, 0, vk::IndexType::UINT16);
        }
        // Bind ImGui pipeline and descriptor set
        unsafe {
            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.vk_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
        }
        // Set dynamic viewport for UI
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width:  draw_data.display_size[0] * fb_scale[0],
            height: draw_data.display_size[1] * fb_scale[1],
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe {
            device.cmd_set_viewport(cmd_buf, 0, &[viewport]);
        }
        // Compute orthographic projection matrix for ImGui (matching Vulkan NDC and winit coords)
        let (l, t) = (draw_data.display_pos[0], draw_data.display_pos[1]);
        let (w, h) = (draw_data.display_size[0], draw_data.display_size[1]);
        let r = l + w;
        let b = t + h;
        // clang-format off
        let proj = [
            [ 2.0 / (r - l),    0.0,               0.0, 0.0 ],
            [ 0.0,             2.0 / (b - t),     0.0, 0.0 ],
            [ 0.0,              0.0,              1.0, 0.0 ],
            [ (r + l) / (l - r), (t + b) / (t - b), 0.0, 1.0 ],
        ];
        // clang-format on
        unsafe {
            device.cmd_push_constants(
                cmd_buf,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::cast_slice(&proj),
            );
        }
        // 3) Iterate draw lists and issue draw calls
        unsafe {
            let mut vertex_offset: i32 = 0;
            let mut index_offset: u32 = 0;
            for draw_list in draw_data.draw_lists() {
                for cmd in draw_list.commands() {
                    if let imgui::DrawCmd::Elements { count, cmd_params } = cmd {
                        // Set scissor rectangle from ImGui clip rect
                        let clip = cmd_params.clip_rect;
                        // Convert logical clip rect to physical pixels
                        let scissor = vk::Rect2D {
                            offset: vk::Offset2D {
                                x: (clip[0] * fb_scale[0]).max(0.0) as i32,
                                y: (clip[1] * fb_scale[1]).max(0.0) as i32,
                            },
                            extent: vk::Extent2D {
                                width: ((clip[2] - clip[0]) * fb_scale[0]).max(0.0) as u32,
                                height: ((clip[3] - clip[1]) * fb_scale[1]).max(0.0) as u32,
                            },
                        };
                        device.cmd_set_scissor(cmd_buf, 0, &[scissor]);
                        // Draw indexed
                        device.cmd_draw_indexed(
                            cmd_buf,
                            count as u32,
                            1,
                            index_offset + cmd_params.idx_offset as u32,
                            vertex_offset + cmd_params.vtx_offset as i32,
                            0,
                        );
                    }
                }
                index_offset += draw_list.idx_buffer().len() as u32;
                vertex_offset += draw_list.vtx_buffer().len() as i32;
            }
        }
    }
}