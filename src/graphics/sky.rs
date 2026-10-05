//! --------------------------------------------------------------------------------------
//! Panorama Sky (sky.rs)
//!
//! Draws an equirectangular photograph behind the scene with one fullscreen triangle.
//! The panorama image remains owned by `MaterialManager`'s texture cache; this renderer
//! owns only its sampler, descriptor set layout, shader code, and pipeline.
//!
//! --------------------------------------------------------------------------------------

use std::error::Error;

use ash::vk;
use cgmath::{Matrix4, SquareMatrix, Vector4};

use crate::graphics::camera::Camera;
use crate::graphics::materialmanager::MaterialManager;
use crate::graphics::pipeline::Pipeline;
use crate::graphics::shaders::ShaderStageInfo;
use crate::vulkan::base::VulkanBase;

const SKY_TEXTURE_PATH: &str = "assets/textures/sky/kloppenheim_05_puresky.jpg";
/// Aligns the panorama's photographed sun horizontally with the scene's directional light.
const PANORAMA_YAW_RADIANS: f32 = -1.20;

/// Owns the Vulkan state specific to the fullscreen panorama pass.
pub struct SkyRenderer {
    pipeline: Pipeline,
    shaders: SkyShaders,
    descriptor_set_layout: vk::DescriptorSetLayout,
    image_info: vk::DescriptorImageInfo,
}

impl SkyRenderer {
    /// Queues the panorama through the shared texture batch and creates its draw state.
    pub fn new(
        vb: &VulkanBase,
        material_manager: &mut MaterialManager,
    ) -> Result<Self, Box<dyn Error>> {
        let (image_view, _shared_material_sampler) =
            material_manager.load_shared_texture(vb, SKY_TEXTURE_PATH)?;
        // Longitude repeats at the left/right seam, but latitude must stop at the poles.
        // Material textures repeat both axes, so the sky needs this one specialised sampler.
        let sampler = create_sampler(&vb.device)?;
        let descriptor_set_layout = create_descriptor_set_layout(&vb.device)?;
        let image_info = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        let shaders = SkyShaders::load()?;
        let mut pipeline = Pipeline::new_background(&vb.device, &[descriptor_set_layout])?;
        pipeline.create_graphics_pipeline_with_vertex_input(
            &vb.device,
            vb.pipeline_cache,
            vb.swapchain.extent,
            vb.swapchain.color_format,
            vb.swapchain.depth_format,
            &[&shaders.vertex, &shaders.fragment],
            &vb.engine_settings,
            &[],
            &[],
            vk::CullModeFlags::NONE,
        )?;

        Ok(Self {
            pipeline,
            shaders,
            descriptor_set_layout,
            image_info,
        })
    }

    /// Rebuilds the swapchain-dependent pipeline.
    pub fn recreate_pipeline(&mut self, vb: &VulkanBase) -> Result<(), Box<dyn Error>> {
        self.pipeline.recreate_with_vertex_input(
            &vb.device,
            vb.pipeline_cache,
            vb.swapchain.extent,
            vb.swapchain.color_format,
            vb.swapchain.depth_format,
            &[&self.shaders.vertex, &self.shaders.fragment],
            &vb.engine_settings,
            &[],
            &[],
            vk::CullModeFlags::NONE,
        )
    }

    /// Records one three-vertex draw before scene geometry overwrites the background.
    pub fn draw(&self, vb: &VulkanBase, cmd: vk::CommandBuffer, camera: &Camera) {
        let device = &vb.device;
        let push_constants = sky_push_constants(camera);
        let write = vk::WriteDescriptorSet::default()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&self.image_info));
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.vk_pipeline,
            );
            vb.push_descriptor.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.vk_layout,
                0,
                &[write],
            );
            device.cmd_push_constants(
                cmd,
                self.pipeline.vk_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &push_constants,
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }

    /// Releases the sampler, layout, and pipeline before the cache-owned image is destroyed.
    pub fn cleanup(&mut self, device: &ash::Device) {
        self.pipeline.cleanup(device);
        unsafe {
            device.destroy_sampler(self.image_info.sampler, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

fn create_sampler(device: &ash::Device) -> Result<vk::Sampler, vk::Result> {
    let sampler_info = vk::SamplerCreateInfo {
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        address_mode_u: vk::SamplerAddressMode::REPEAT,
        address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        max_lod: 0.0,
        ..Default::default()
    };
    unsafe { device.create_sampler(&sampler_info, None) }
}

fn create_descriptor_set_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout, vk::Result> {
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
    unsafe { device.create_descriptor_set_layout(&layout_info, None) }
}

/// Packs the inverse projection and rotation-only view matrix into GLSL column-major order.
fn sky_push_constants(camera: &Camera) -> [u8; 80] {
    let mut rotation_only_view = *camera.get_view();
    rotation_only_view.w = Vector4::new(0.0, 0.0, 0.0, 1.0);
    let inverse_view_projection = (*camera.get_projection() * rotation_only_view)
        .invert()
        .unwrap_or_else(Matrix4::identity);

    let mut bytes = [0_u8; 80];
    let columns = [
        inverse_view_projection.x,
        inverse_view_projection.y,
        inverse_view_projection.z,
        inverse_view_projection.w,
    ];
    let mut offset = 0;
    for column in columns {
        for value in [column.x, column.y, column.z, column.w] {
            bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
            offset += 4;
        }
    }
    for value in [PANORAMA_YAW_RADIANS, 0.0, 0.0, 0.0] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
        offset += 4;
    }
    debug_assert_eq!(offset, bytes.len());
    bytes
}

/// SPIR-V kept for rebuilding the pipeline after a swapchain change.
struct SkyShaders {
    vertex: ShaderStageInfo,
    fragment: ShaderStageInfo,
}

impl SkyShaders {
    fn load() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            vertex: ShaderStageInfo::load(vk::ShaderStageFlags::VERTEX, "sky.vert.spv")?,
            fragment: ShaderStageInfo::load(vk::ShaderStageFlags::FRAGMENT, "sky.frag.spv")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{PANORAMA_YAW_RADIANS, sky_push_constants};
    use crate::graphics::camera::Camera;

    #[test]
    fn sky_push_constants_contain_a_matrix_and_explicit_panorama_rotation() {
        let bytes = sky_push_constants(&Camera::new());
        assert_eq!(bytes.len(), 80);
        assert_eq!(
            f32::from_ne_bytes(bytes[64..68].try_into().unwrap()),
            PANORAMA_YAW_RADIANS
        );
    }
}
