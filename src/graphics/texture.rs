//! --------------------------------------------------------------------------------------
//! Texture Module (texture.rs)
//!
//! Created: July 2025
//! Author: Stephen Willey (with AI-assistant)
//!
//! This module defines `Texture`, responsible for loading image files (e.g. PNGs),
//! creating Vulkan image resources, image views, and samplers. It handles staging
//! buffer uploads to GPU memory and cleanup of those resources.
//!
//! --------------------------------------------------------------------------------------

use ash::vk;
use vk_mem::{Alloc, Allocator, Allocation, MemoryUsage};

/// Holds a GPU texture: image, its allocation, view, and sampler.
pub struct Texture {
    pub image:      vk::Image,
    pub allocation: Allocation,
    pub image_view: vk::ImageView,
    pub sampler:    vk::Sampler,
}

impl Texture {
    /// Creates a new `Texture` by loading image data from the given path,
    /// uploading via a staging buffer, and setting up the image, view, and sampler.
    /// # Arguments
    /// * `device` - The Vulkan device.
    /// * `allocator` - The global VMA allocator.
    /// * `command_pool` - The command pool to use for creating the texture.
    /// * `queue` - The queue to use for submitting the texture creation commands.
    /// * `image_path` - The path to the image file.
    /// # Returns
    /// * `Result<Self, Box<dyn std::error::Error>>` - Returns the initialized `Texture` on success, or an error on failure.
    pub fn new(
        device: &ash::Device,
        allocator: &Allocator,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        image_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Load the PNG file into an RGBA8 image buffer
        let img = image::open(image_path)?.to_rgba8();
        let (width, height) = img.dimensions();
        let pixels = img.into_raw(); // Vec<u8> with RGBA8 data
        let image_size = (width as vk::DeviceSize) * (height as vk::DeviceSize) * 4;

        // Create the staging buffer using VMA
        let buffer_info = vk::BufferCreateInfo {
            size: image_size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::CpuToGpu,
            ..Default::default()
        };
        let (staging_buffer, mut staging_allocation) =
            unsafe { allocator.create_buffer(&buffer_info, &alloc_info)? };

        // Copy pixel data into the staging buffer
        unsafe {
            let data_ptr = allocator.map_memory(&mut staging_allocation)? as *mut u8;
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), data_ptr, pixels.len());
            allocator.unmap_memory(&mut staging_allocation);
        }

        // Create the GPU image in device-local memory using VMA
        let image_create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D { width, height, depth: 1 },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let image_alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::GpuOnly,
            ..Default::default()
        };
        let (image, image_allocation) =
            unsafe { allocator.create_image(&image_create_info, &image_alloc_info)? };

        // Begin one-time command buffer for layout transitions and copy
        let alloc_info = vk::CommandBufferAllocateInfo {
            level: vk::CommandBufferLevel::PRIMARY,
            command_pool,
            command_buffer_count: 1,
            ..Default::default()
        };
        let cmd_buf = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };
        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe {
            device.begin_command_buffer(cmd_buf, &begin_info)?;

            // Transition UNDEFINED -> TRANSFER_DST_OPTIMAL
            let barrier1 = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier1],
            );

            // Copy buffer -> image
            let region = vk::BufferImageCopy {
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
            device.cmd_copy_buffer_to_image(
                cmd_buf,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            // Transition TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
            let barrier2 = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier2],
            );

            device.end_command_buffer(cmd_buf)?;

            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd_buf,
                ..Default::default()
            };
            device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
            device.queue_wait_idle(queue)?;
            device.free_command_buffers(command_pool, &[cmd_buf]);
        }

        // Cleanup staging resources
        unsafe {
            allocator.destroy_buffer(staging_buffer, &mut staging_allocation);
        }

        // Create image view
        let view_info = vk::ImageViewCreateInfo {
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            components: vk::ComponentMapping::default(),
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };
        let image_view = unsafe { device.create_image_view(&view_info, None)? };

        // Create sampler
        let sampler_info = vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            anisotropy_enable: vk::TRUE,
            max_anisotropy: 16.0,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            min_lod: 0.0,
            max_lod: 0.0,
            ..Default::default()
        };
        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        Ok(Texture {
            image,
            allocation: image_allocation,
            image_view,
            sampler,
        })
    }

    /// Cleans up Vulkan resources associated with this texture.
    /// # Arguments
    /// * `device` - The Vulkan device to use for cleanup.
    /// * `allocator` - The global VMA allocator.
    pub fn cleanup(&mut self, device: &ash::Device, allocator: &Allocator) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.image_view, None);
            allocator.destroy_image(self.image, &mut self.allocation);
        }
    }
}