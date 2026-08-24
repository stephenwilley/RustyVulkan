//! Wind-map image ownership and deterministic noise generation.
//!
//! The descriptor set, sampler, view, image, and allocation are one resource
//! family, so WindMap destroys them together in the reverse of creation.

use super::{WIND_MAP_SIZE, generation::hash01};
use ash::vk;
use vk_mem::{Allocation, Allocator};

/// Grass-owned wind image and descriptor resources.
pub(super) struct WindMap {
    pub(super) image: vk::Image,
    pub(super) allocation: Allocation,
    pub(super) view: vk::ImageView,
    pub(super) sampler: vk::Sampler,
    pub(super) descriptor_pool: vk::DescriptorPool,
    pub(super) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) descriptor_set: vk::DescriptorSet,
}

impl WindMap {
    pub(super) fn cleanup(&mut self, device: &ash::Device, allocator: &Allocator) {
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            allocator.destroy_image(self.image, &mut self.allocation);
        }
    }
}

pub(super) fn create_wind_map_resources(
    device: &ash::Device,
    image: vk::Image,
    allocation: Allocation,
) -> Result<WindMap, vk::Result> {
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let view_info = vk::ImageViewCreateInfo {
        image,
        view_type: vk::ImageViewType::TYPE_2D,
        format: vk::Format::R8G8B8A8_UNORM,
        subresource_range: range,
        ..Default::default()
    };
    let view = unsafe { device.create_image_view(&view_info, None)? };
    let sampler_info = vk::SamplerCreateInfo {
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        address_mode_u: vk::SamplerAddressMode::REPEAT,
        address_mode_v: vk::SamplerAddressMode::REPEAT,
        address_mode_w: vk::SamplerAddressMode::REPEAT,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        max_lod: 0.0,
        ..Default::default()
    };
    let sampler = unsafe { device.create_sampler(&sampler_info, None)? };
    let binding = vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::VERTEX,
        ..Default::default()
    };
    let layout_info = vk::DescriptorSetLayoutCreateInfo {
        binding_count: 1,
        p_bindings: &binding,
        ..Default::default()
    };
    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };
    let pool_size = vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
    };
    let pool_info = vk::DescriptorPoolCreateInfo {
        max_sets: 1,
        pool_size_count: 1,
        p_pool_sizes: &pool_size,
        ..Default::default()
    };
    let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
    let allocate_info = vk::DescriptorSetAllocateInfo {
        descriptor_pool,
        descriptor_set_count: 1,
        p_set_layouts: &descriptor_set_layout,
        ..Default::default()
    };
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&allocate_info)?[0] };
    let image_info = vk::DescriptorImageInfo {
        sampler,
        image_view: view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };
    let write = vk::WriteDescriptorSet {
        dst_set: descriptor_set,
        dst_binding: 0,
        descriptor_count: 1,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        p_image_info: &image_info,
        ..Default::default()
    };
    unsafe { device.update_descriptor_sets(&[write], &[]) };
    Ok(WindMap {
        image,
        allocation,
        view,
        sampler,
        descriptor_pool,
        descriptor_set_layout,
        descriptor_set,
    })
}

/// Generates two seamless value-noise channels: broad gusts in red and fine motion in green.
pub(super) fn build_wind_map_pixels() -> Vec<u8> {
    let mut pixels = Vec::with_capacity((WIND_MAP_SIZE * WIND_MAP_SIZE * 4) as usize);
    for y in 0..WIND_MAP_SIZE {
        for x in 0..WIND_MAP_SIZE {
            let broad = tileable_value_noise(x, y, 8, 41);
            let detail = tileable_value_noise(x, y, 23, 97);
            pixels.extend_from_slice(&[
                (broad * 255.0).round() as u8,
                (detail * 255.0).round() as u8,
                0,
                255,
            ]);
        }
    }
    pixels
}

fn tileable_value_noise(x: u32, y: u32, cells: u32, seed: u32) -> f32 {
    let grid_x = x as f32 / WIND_MAP_SIZE as f32 * cells as f32;
    let grid_y = y as f32 / WIND_MAP_SIZE as f32 * cells as f32;
    let cell_x = grid_x.floor() as u32;
    let cell_y = grid_y.floor() as u32;
    let mut fraction_x = grid_x.fract();
    let mut fraction_y = grid_y.fract();
    fraction_x = fraction_x * fraction_x * (3.0 - 2.0 * fraction_x);
    fraction_y = fraction_y * fraction_y * (3.0 - 2.0 * fraction_y);
    let sample = |offset_x: u32, offset_y: u32| {
        let wrapped_x = (cell_x + offset_x) % cells;
        let wrapped_y = (cell_y + offset_y) % cells;
        hash01(wrapped_y * cells + wrapped_x, seed)
    };
    let bottom = sample(0, 0) + (sample(1, 0) - sample(0, 0)) * fraction_x;
    let top = sample(0, 1) + (sample(1, 1) - sample(0, 1)) * fraction_x;
    bottom + (top - bottom) * fraction_y
}
