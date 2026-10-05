//! Font-atlas upload and sampled-image descriptor ownership for ImGui.
//!
//! Construction is kept together because the temporary staging allocation is
//! borrowed only during upload, while the returned image resources become fields
//! owned by ImGuiRenderer and are later released by its cleanup method.

use super::{ImGuiRenderer, VulkanBase};
use ash::vk;
use imgui::{Context as ImGuiContext, FontAtlasTexture, FontConfig, FontSource};
use vk_mem::{Alloc, Allocation, Allocator, MemoryUsage};

use crate::graphics::shadow_math::SHADOW_CASCADE_COUNT;

impl ImGuiRenderer {
    // ----------------------------------------------------------
    // 1 - Font Atlas Upload
    /// Creates a staging buffer for font atlas data using VMA.
    /// # Arguments
    /// * `allocator` - Global Vulkan memory allocator.
    /// * `atlas` - The ImGui font atlas texture.
    /// # Returns
    /// * `(vk::Buffer, Allocation)` - The staging buffer and its allocation.
    fn create_staging_buffer(
        allocator: &Allocator,
        atlas: &FontAtlasTexture,
    ) -> (vk::Buffer, Allocation) {
        let size = (atlas.width * atlas.height * 4) as vk::DeviceSize;
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        };
        unsafe {
            allocator
                .create_buffer(&buffer_info, &alloc_info)
                .expect("create staging buffer")
        }
    }

    /// Maps the staging buffer memory and copies the font atlas data into it using VMA.
    /// # Arguments
    /// * `allocator` - Global Vulkan memory allocator.
    /// * `staging_alloc` - Allocation for the staging buffer.
    /// * `atlas` - The ImGui font atlas texture.
    fn fill_staging_buffer(
        allocator: &Allocator,
        staging_alloc: &mut Allocation,
        atlas: &FontAtlasTexture,
    ) {
        unsafe {
            let data_ptr = allocator.map_memory(staging_alloc).expect("map staging");
            std::ptr::copy_nonoverlapping(atlas.data.as_ptr(), data_ptr, atlas.data.len());
            allocator
                .flush_allocation(staging_alloc, 0, atlas.data.len() as u64)
                .expect("flush font staging");
            allocator.unmap_memory(staging_alloc);
        }
    }

    /// Creates an optimal-tiling Vulkan image and device-local memory for the font atlas using VMA.
    /// # Arguments
    /// * `allocator` - Global Vulkan memory allocator.
    /// * `width` - The width of the font atlas.
    /// * `height` - The height of the font atlas.
    /// # Returns
    /// * `(vk::Image, Allocation)` - The font image and its allocation.
    fn create_font_image(
        allocator: &Allocator,
        width: u32,
        height: u32,
    ) -> (vk::Image, Allocation) {
        let image_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            initial_layout: vk::ImageLayout::UNDEFINED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        unsafe {
            allocator
                .create_image(&image_info, &alloc_info)
                .expect("create font image")
        }
    }

    /// Records and submits a one-time command buffer to transition image layouts and copy data from a staging buffer to the image.
    /// # Arguments
    /// * `base` - The VulkanBase instance.
    /// * `staging_buffer` - The staging buffer containing the image data.
    /// * `image` - The destination image.
    /// * `width` - The width of the image.
    /// * `height` - The height of the image.
    fn copy_buffer_to_image(
        base: &VulkanBase,
        staging_buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
    ) {
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
        let barrier1 = vk::ImageMemoryBarrier2 {
            dst_stage_mask: vk::PipelineStageFlags2::COPY,
            dst_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
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
            ..Default::default()
        };
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[barrier1]),
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
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
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
        let barrier2 = vk::ImageMemoryBarrier2 {
            src_stage_mask: vk::PipelineStageFlags2::COPY,
            src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            dst_access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
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
            ..Default::default()
        };
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[barrier2]),
            );
        }

        unsafe { device.end_command_buffer(cmd_buffer).unwrap() };

        // Submit and wait
        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd_buffer)];
        let submit = vk::SubmitInfo2::default().command_buffer_infos(&command_buffers);
        unsafe {
            device
                .queue_submit2(base.graphics_queue, &[submit], vk::Fence::null())
                .unwrap();
            device.queue_wait_idle(base.graphics_queue).unwrap();
        }

        // 5) Clean up staging and command buffer
        unsafe {
            device.free_command_buffers(base.command_pool, &[cmd_buffer]);
        }
    }

    /// Creates a 2D ImageView for the font atlas.
    /// # Arguments
    /// * `base` - The VulkanBase instance.
    /// * `image` - The image to create the view for.
    /// # Returns
    /// * `vk::ImageView` - The created image view.
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

    /// Creates a linear, clamp-to-edge sampler.
    /// # Arguments
    /// * `base` - The VulkanBase instance.
    /// # Returns
    /// * `vk::Sampler` - The created sampler.
    fn create_sampler(base: &VulkanBase) -> vk::Sampler {
        let device = &base.device;
        let sampler_info = vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            mip_lod_bias: 0.0,
            anisotropy_enable: vk::FALSE,
            max_anisotropy: 1.0,
            min_lod: 0.0,
            max_lod: 1.0,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,
            ..Default::default()
        };
        unsafe { device.create_sampler(&sampler_info, None).unwrap() }
    }

    // ----------------------------------------------------------
    // 2 - Descriptors
    // ----------------------------------------------------------
    /// Helper: create the push-descriptor set layout for the texture each draw samples
    /// # Arguments
    /// * `base` - The VulkanBase instance.
    /// # Returns
    /// * `vk::DescriptorSetLayout` - The created descriptor set layout.
    fn create_imgui_descriptor_set_layout(base: &VulkanBase) -> vk::DescriptorSetLayout {
        let binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        };
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(std::slice::from_ref(&binding));
        unsafe {
            base.device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        }
    }

    /// Register or replace an image for displaying with ui.image(...).
    /// Passing the ID returned earlier replaces that image and keeps the ID.
    pub fn ensure_texture(
        &mut self,
        sampler: vk::Sampler,
        view: vk::ImageView,
        existing: Option<imgui::TextureId>,
    ) -> imgui::TextureId {
        let image_info = vk::DescriptorImageInfo {
            sampler,
            image_view: view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        match existing {
            Some(id) if id.id() < self.textures.len() => {
                self.textures[id.id()] = image_info;
                id
            }
            _ => {
                self.textures.push(image_info);
                imgui::TextureId::new(self.textures.len() - 1)
            }
        }
    }

    /// Returns the texture ID that shows one shadow cascade, pointed at this frame's view.
    pub fn shadow_texture_id(
        &mut self,
        cascade_index: usize,
        sampler: vk::Sampler,
        view: vk::ImageView,
    ) -> imgui::TextureId {
        let texture_id = self.ensure_texture(sampler, view, self.shadow_tex_ids[cascade_index]);
        self.shadow_tex_ids[cascade_index] = Some(texture_id);
        texture_id
    }

    // ----------------------------------------------------------
    // 3 - Rendering Pipeline and Dynamic Buffers
    // load_shaders

    /// Creates a new ImGuiRenderer instance.
    /// # Arguments
    /// * `base` - The VulkanBase instance to use for resource creation.
    /// * `imgui` - The ImGui context to initialize the renderer with.
    /// # Returns
    /// A new ImGuiRenderer instance with all resources initialized.
    pub fn new(base: &mut VulkanBase, imgui: &mut ImGuiContext) -> Self {
        let device = base.device.clone();
        let allocator = base.allocator.as_ref().expect("Allocator not initialized");

        // 0) Load default font atlas
        imgui.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(FontConfig {
                rasterizer_multiply: 1.0,
                ..FontConfig::default()
            }),
        }]);
        let atlas = imgui.fonts().build_rgba32_texture();
        // 1) Create staging buffer for font atlas
        let (staging_buffer, mut staging_alloc) = Self::create_staging_buffer(allocator, &atlas);
        // 2) Map memory and copy font atlas data into it
        Self::fill_staging_buffer(allocator, &mut staging_alloc, &atlas);
        // 3) Create font image with device-local memory
        let (font_image, font_allocation) =
            Self::create_font_image(allocator, atlas.width, atlas.height);
        // 4) Copy from staging to the font image
        Self::copy_buffer_to_image(base, staging_buffer, font_image, atlas.width, atlas.height);
        // 5) Clean up staging buffer
        unsafe {
            allocator.destroy_buffer(staging_buffer, &mut staging_alloc);
        }
        // 6) Create an ImageView for the font atlas
        let font_image_view = Self::create_image_view(base, font_image);
        // 7) Create a Sampler for the font atlas
        let font_sampler = Self::create_sampler(base);

        let font_image_info = vk::DescriptorImageInfo {
            sampler: font_sampler,
            image_view: font_image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        let mut imgui_renderer = Self {
            descriptor_set_layout: Self::create_imgui_descriptor_set_layout(base),
            pipeline_layout: vk::PipelineLayout::null(),
            vk_pipeline: vk::Pipeline::null(),
            font_sampler: Some(font_sampler),
            font_image: Some(font_image),
            font_image_allocation: Some(font_allocation),
            font_image_view: Some(font_image_view),
            frame_buffers: Vec::new(),
            device,
            push_descriptor: base.push_descriptor.clone(),
            vert_stage: None,
            frag_stage: None,
            textures: vec![font_image_info],
            shadow_tex_ids: [None; SHADOW_CASCADE_COUNT],
        };

        // 8) Loads the shaders and build the pipeline
        imgui_renderer
            .rebuild_pipeline(base)
            .expect("Failed to rebuild imgui pipeline");

        imgui_renderer
    }
}
