//! One-shot uploads for grass geometry, instances, and the wind texture.
//!
//! The batch owns temporary staging buffers until the transfer fence signals.
//! Consuming finish makes that ownership hand-off explicit: afterwards only
//! device-local buffers retained by GrassRenderer remain.

use super::{WIND_MAP_SIZE, build_wind_map_pixels};
use crate::vulkan::base::VulkanBase;
use ash::vk;
use bytemuck::Pod;
use vk_mem::{Alloc, Allocation, Allocator, MemoryUsage};
pub(super) struct GrassUploadBatch {
    command_buffer: vk::CommandBuffer,
    staging_buffers: Vec<(vk::Buffer, Allocation)>,
}

impl GrassUploadBatch {
    pub(super) fn new(vb: &VulkanBase) -> Result<Self, vk::Result> {
        let allocate_info = vk::CommandBufferAllocateInfo {
            command_pool: vb.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        let command_buffer = unsafe { vb.device.allocate_command_buffers(&allocate_info)?[0] };
        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe {
            vb.device
                .begin_command_buffer(command_buffer, &begin_info)?
        };
        Ok(Self {
            command_buffer,
            staging_buffers: Vec::new(),
        })
    }

    pub(super) fn upload_buffer<T: Pod>(
        &mut self,
        device: &ash::Device,
        allocator: &Allocator,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, Allocation), vk::Result> {
        let byte_size = std::mem::size_of_val(data) as vk::DeviceSize;
        let staging_info = vk::BufferCreateInfo {
            size: byte_size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let staging_alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        };
        let (staging, mut staging_allocation) =
            unsafe { allocator.create_buffer(&staging_info, &staging_alloc_info)? };
        unsafe {
            let destination = allocator.map_memory(&mut staging_allocation)? as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), destination, data.len());
            allocator.flush_allocation(&staging_allocation, 0, byte_size)?;
            allocator.unmap_memory(&mut staging_allocation);
        }

        let device_info = vk::BufferCreateInfo {
            size: byte_size,
            usage: usage | vk::BufferUsageFlags::TRANSFER_DST,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let device_alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        let (buffer, allocation) =
            unsafe { allocator.create_buffer(&device_info, &device_alloc_info)? };
        let copy = vk::BufferCopy {
            size: byte_size,
            ..Default::default()
        };
        unsafe { device.cmd_copy_buffer(self.command_buffer, staging, buffer, &[copy]) };
        self.staging_buffers.push((staging, staging_allocation));
        Ok((buffer, allocation))
    }

    pub(super) fn upload_wind_image(
        &mut self,
        device: &ash::Device,
        allocator: &Allocator,
    ) -> Result<(vk::Image, Allocation), vk::Result> {
        let pixels = build_wind_map_pixels();
        let staging_info = vk::BufferCreateInfo {
            size: pixels.len() as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let staging_alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        };
        let (staging, mut staging_allocation) =
            unsafe { allocator.create_buffer(&staging_info, &staging_alloc_info)? };
        unsafe {
            let destination = allocator.map_memory(&mut staging_allocation)?;
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), destination, pixels.len());
            allocator.flush_allocation(&staging_allocation, 0, pixels.len() as u64)?;
            allocator.unmap_memory(&mut staging_allocation);
        }

        let image_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D {
                width: WIND_MAP_SIZE,
                height: WIND_MAP_SIZE,
                depth: 1,
            },
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
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        let (image, allocation) =
            unsafe { allocator.create_image(&image_info, &image_alloc_info)? };
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let to_transfer = vk::ImageMemoryBarrier2 {
            dst_stage_mask: vk::PipelineStageFlags2::COPY,
            dst_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image,
            subresource_range: range,
            ..Default::default()
        };
        let to_shader = vk::ImageMemoryBarrier2 {
            src_stage_mask: vk::PipelineStageFlags2::COPY,
            src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
            dst_access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image,
            subresource_range: range,
            ..Default::default()
        };
        let copy = vk::BufferImageCopy {
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: vk::Extent3D {
                width: WIND_MAP_SIZE,
                height: WIND_MAP_SIZE,
                depth: 1,
            },
            ..Default::default()
        };
        unsafe {
            device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_transfer]),
            );
            device.cmd_copy_buffer_to_image(
                self.command_buffer,
                staging,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
            device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_shader]),
            );
        }
        self.staging_buffers.push((staging, staging_allocation));
        Ok((image, allocation))
    }

    pub(super) fn finish(
        mut self,
        vb: &VulkanBase,
        allocator: &Allocator,
    ) -> Result<(), vk::Result> {
        unsafe {
            // Make every transfer write visible to later vertex/index fetches. Queue order alone
            // orders execution, but this barrier supplies the required memory dependency.
            let barrier = vk::MemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::COPY,
                src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
                    | vk::PipelineStageFlags2::INDEX_INPUT,
                dst_access_mask: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
                    | vk::AccessFlags2::INDEX_READ,
                ..Default::default()
            };
            vb.device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().memory_barriers(&[barrier]),
            );
            vb.device.end_command_buffer(self.command_buffer)?;
            let fence = vb
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;
            let command_buffers =
                [vk::CommandBufferSubmitInfo::default().command_buffer(self.command_buffer)];
            let submit = vk::SubmitInfo2::default().command_buffer_infos(&command_buffers);
            vb.device
                .queue_submit2(vb.graphics_queue, &[submit], fence)?;
            vb.device.wait_for_fences(&[fence], true, u64::MAX)?;
            vb.device.destroy_fence(fence, None);
            vb.device
                .free_command_buffers(vb.command_pool, &[self.command_buffer]);
            for (buffer, mut allocation) in self.staging_buffers.drain(..) {
                allocator.destroy_buffer(buffer, &mut allocation);
            }
        }
        Ok(())
    }
}
