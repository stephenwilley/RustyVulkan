//! --------------------------------------------------------------------------------------
//! Texture Module (texture.rs)
//!
//! Creates GPU images for material textures.  `TextureCache` keeps one copy of
//! each image and one common sampler, while `TextureUploadBatch` records all
//! startup copies into a single command buffer and submits it once.
//!
//! --------------------------------------------------------------------------------------

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::time::Instant;

use ash::vk;
use vk_mem::{Alloc, Allocation, Allocator, MemoryUsage};

use crate::vulkan::base::VulkanBase;

/// A GPU texture owned exclusively by [`TextureCache`]. Sampling state is also
/// cache-owned, so identical sampler objects are not created for every texture.
pub struct Texture {
    /// Device-local image containing the decoded RGBA pixels.
    pub image: vk::Image,
    /// VMA allocation that backs `image`.
    pub allocation: Allocation,
    /// 2D view used by material descriptor sets.
    pub image_view: vk::ImageView,
}

impl Texture {
    pub fn cleanup(&mut self, device: &ash::Device, allocator: &Allocator) {
        unsafe {
            device.destroy_image_view(self.image_view, None);
            allocator.destroy_image(self.image, &mut self.allocation);
        }
    }
}

/// Host-visible pixels that must survive until the queued copy has finished.
struct StagingBuffer {
    buffer: vk::Buffer,
    allocation: Allocation,
}

/// Records uploads for many textures into a single command buffer.
struct TextureUploadBatch {
    command_buffer: vk::CommandBuffer,
    staging_buffers: Vec<StagingBuffer>,
}

impl TextureUploadBatch {
    fn new(device: &ash::Device, command_pool: vk::CommandPool) -> Result<Self, vk::Result> {
        // Every queued texture copy is recorded into this one one-time buffer.
        let alloc_info = vk::CommandBufferAllocateInfo {
            level: vk::CommandBufferLevel::PRIMARY,
            command_pool,
            command_buffer_count: 1,
            ..Default::default()
        };
        let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };
        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe { device.begin_command_buffer(command_buffer, &begin_info)? };

        Ok(Self {
            command_buffer,
            staging_buffers: Vec::new(),
        })
    }

    fn upload(
        &mut self,
        device: &ash::Device,
        allocator: &Allocator,
        image_path: &str,
    ) -> Result<Texture, Box<dyn Error>> {
        let prepare_start = Instant::now();
        // Decode on the CPU first so every source format becomes RGBA8.
        let img = image::open(image_path)?.to_rgba8();
        let (width, height) = img.dimensions();
        let pixels = img.into_raw();

        // Stage the pixels in host-visible memory; the final image stays device-local.
        let buffer_info = vk::BufferCreateInfo {
            size: pixels.len() as vk::DeviceSize,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        };
        let (staging_buffer, mut staging_allocation) =
            unsafe { allocator.create_buffer(&buffer_info, &alloc_info)? };
        unsafe {
            let data_ptr = allocator.map_memory(&mut staging_allocation)?;
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), data_ptr, pixels.len());
            allocator.flush_allocation(&staging_allocation, 0, pixels.len() as u64)?;
            allocator.unmap_memory(&mut staging_allocation);
        }

        // Create the sampled image now, then record its layout transitions and copy.
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

        let subresource_range = vk::ImageSubresourceRange {
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
            subresource_range,
            ..Default::default()
        };
        let region = vk::BufferImageCopy {
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            ..Default::default()
        };
        let to_shader_read = vk::ImageMemoryBarrier2 {
            src_stage_mask: vk::PipelineStageFlags2::COPY,
            src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            dst_access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image,
            subresource_range,
            ..Default::default()
        };
        unsafe {
            // undefined -> transfer destination -> shader-readable
            device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_transfer]),
            );
            device.cmd_copy_buffer_to_image(
                self.command_buffer,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
            device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_shader_read]),
            );
        }

        let view_info = vk::ImageViewCreateInfo {
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            components: vk::ComponentMapping::default(),
            subresource_range,
            ..Default::default()
        };
        let image_view = unsafe { device.create_image_view(&view_info, None)? };
        // The copy has not run yet, so keep the staging allocation alive until `finish`.
        self.staging_buffers.push(StagingBuffer {
            buffer: staging_buffer,
            allocation: staging_allocation,
        });

        let prepare_ms = prepare_start.elapsed().as_secs_f32() * 1_000.0;
        if prepare_ms >= 100.0 {
            println!("🖼️ Prepared {image_path} ({width}×{height}) in {prepare_ms:.1} ms");
        }

        // The returned image is ready to use only after the enclosing batch is flushed.
        Ok(Texture {
            image,
            allocation,
            image_view,
        })
    }

    fn finish(
        mut self,
        device: &ash::Device,
        allocator: &Allocator,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<(), vk::Result> {
        unsafe {
            // Submit all recorded copies together, then wait once before rendering begins.
            device.end_command_buffer(self.command_buffer)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let command_buffers =
                [vk::CommandBufferSubmitInfo::default().command_buffer(self.command_buffer)];
            let submit_info = vk::SubmitInfo2::default().command_buffer_infos(&command_buffers);
            device.queue_submit2(queue, &[submit_info], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            // The fence guarantees the GPU no longer reads the staging buffers.
            device.destroy_fence(fence, None);
            device.free_command_buffers(command_pool, &[self.command_buffer]);
            for staging in self.staging_buffers.drain(..) {
                let mut allocation = staging.allocation;
                allocator.destroy_buffer(staging.buffer, &mut allocation);
            }
        }
        Ok(())
    }

    fn discard(
        mut self,
        device: &ash::Device,
        allocator: &Allocator,
        command_pool: vk::CommandPool,
    ) {
        unsafe {
            // Error cleanup path: nothing was submitted, so resources are immediately safe.
            device.free_command_buffers(command_pool, &[self.command_buffer]);
            for staging in self.staging_buffers.drain(..) {
                let mut allocation = staging.allocation;
                allocator.destroy_buffer(staging.buffer, &mut allocation);
            }
        }
    }
}

/// Owns deduplicated textures, their common sampler, and a pending upload batch.
pub struct TextureCache {
    /// Canonical file path to the cache-owned texture.
    textures: HashMap<String, Texture>,
    /// All material textures currently share the same sampling settings.
    sampler: vk::Sampler,
    /// Pending uploads recorded while the scene is being constructed.
    upload_batch: Option<TextureUploadBatch>,
}

impl TextureCache {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            sampler: vk::Sampler::null(),
            upload_batch: None,
        }
    }

    fn key_for_path(image_path: &str) -> String {
        // Normalise equivalent paths so they map to the same cached texture.
        std::fs::canonicalize(Path::new(image_path))
            .unwrap_or_else(|_| Path::new(image_path).to_path_buf())
            .to_string_lossy()
            .into_owned()
    }

    fn ensure_sampler(&mut self, device: &ash::Device) -> Result<(), vk::Result> {
        if self.sampler != vk::Sampler::null() {
            return Ok(());
        }
        // One sampler shared by every Texture.
        let sampler_info = vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            anisotropy_enable: vk::FALSE,
            max_anisotropy: 1.0,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            min_lod: 0.0,
            max_lod: 0.0,
            ..Default::default()
        };
        self.sampler = unsafe { device.create_sampler(&sampler_info, None)? };
        Ok(())
    }

    /// Return a cached image view, or create its texture and queue its first upload.
    ///
    /// `vk::ImageView` is a small, `Copy` Vulkan handle, not an owning Rust
    /// reference.  A material can store this handle in a descriptor while this
    /// cache remains the one Rust owner responsible for destroying the image.
    pub fn load(
        &mut self,
        vb: &VulkanBase,
        image_path: &str,
    ) -> Result<vk::ImageView, Box<dyn Error>> {
        let key = Self::key_for_path(image_path);
        // No decode, allocation, or GPU work for a texture already used by a material.
        if let Some(texture) = self.textures.get(&key) {
            return Ok(texture.image_view);
        }

        self.ensure_sampler(&vb.device)?;
        // Start the batch lazily so untextured scenes do not allocate a command buffer.
        if self.upload_batch.is_none() {
            self.upload_batch = Some(TextureUploadBatch::new(&vb.device, vb.command_pool)?);
        }
        let texture = self
            .upload_batch
            .as_mut()
            .expect("texture upload batch")
            .upload(
                &vb.device,
                vb.allocator.as_ref().expect("allocator"),
                image_path,
            )?;
        let image_view = texture.image_view;
        self.textures.insert(key, texture);
        Ok(image_view)
    }

    /// Sampler paired with every texture descriptor created by this cache.
    pub fn sampler(&self) -> vk::Sampler {
        debug_assert_ne!(self.sampler, vk::Sampler::null());
        self.sampler
    }

    /// Submit all pending texture copies and wait once before the first draw.
    pub fn flush(&mut self, vb: &VulkanBase) -> Result<(), Box<dyn Error>> {
        if let Some(batch) = self.upload_batch.take() {
            let upload_count = batch.staging_buffers.len();
            batch.finish(
                &vb.device,
                vb.allocator.as_ref().expect("allocator"),
                vb.command_pool,
                vb.graphics_queue,
            )?;
            println!(
                "🖼️ Uploaded {} unique textures in one GPU submission",
                upload_count
            );
        }
        Ok(())
    }

    /// Destroy queued staging resources, cached images, and the common sampler.
    pub fn cleanup(
        &mut self,
        device: &ash::Device,
        allocator: &Allocator,
        command_pool: vk::CommandPool,
    ) {
        // A failed/aborted load may have a recording batch that was never submitted.
        if let Some(batch) = self.upload_batch.take() {
            batch.discard(device, allocator, command_pool);
        }
        // Materials hold only copied image-view handles, so the cache has the
        // sole `Texture` ownership and can destroy every image deterministically.
        for (_, mut texture) in self.textures.drain() {
            texture.cleanup(device, allocator);
        }
        unsafe {
            if self.sampler != vk::Sampler::null() {
                device.destroy_sampler(self.sampler, None);
                self.sampler = vk::Sampler::null();
            }
        }
    }
}

impl Default for TextureCache {
    fn default() -> Self {
        Self::new()
    }
}
