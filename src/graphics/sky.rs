//! --------------------------------------------------------------------------------------
//! Panorama Sky (sky.rs)
//!
//! Draws an equirectangular photograph behind the scene with one fullscreen triangle.
//! The panorama image remains owned by `MaterialManager`'s texture cache; this renderer
//! owns only its descriptor objects, shader modules, and pipeline.
//!
//! --------------------------------------------------------------------------------------

use std::error::Error;

use ash::vk;
use cgmath::{Matrix4, SquareMatrix, Vector4};

use crate::graphics::camera::Camera;
use crate::graphics::materialmanager::MaterialManager;
use crate::graphics::pipeline::Pipeline;
use crate::graphics::shaders::{ShaderModule, ShaderStageInfo};
use crate::vulkan::base::VulkanBase;

const SKY_TEXTURE_PATH: &str = "assets/textures/sky/kloppenheim_05_puresky.jpg";
/// Aligns the panorama's photographed sun horizontally with the scene's directional light.
const PANORAMA_YAW_RADIANS: f32 = -1.20;

/// Owns the Vulkan state specific to the fullscreen panorama pass.
pub struct SkyRenderer {
    pipeline: Pipeline,
    shaders: SkyShaders,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    sampler: vk::Sampler,
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
        let (descriptor_pool, descriptor_set) =
            create_descriptor_set(&vb.device, descriptor_set_layout)?;

        let image_info = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let descriptor_write = vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info: &image_info,
            ..Default::default()
        };
        unsafe { vb.device.update_descriptor_sets(&[descriptor_write], &[]) };

        let shaders = SkyShaders::load(&vb.device)?;
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
            descriptor_pool,
            descriptor_set,
            sampler,
        })
    }

    /// Rebuilds the swapchain-dependent pipeline while retaining the panorama descriptor.
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
    pub fn draw(&self, device: &ash::Device, cmd: vk::CommandBuffer, camera: &Camera) {
        let push_constants = sky_push_constants(camera);
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.vk_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.vk_layout,
                0,
                &[self.descriptor_set],
                &[],
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

    /// Releases descriptor and pipeline handles before the cache-owned image is destroyed.
    pub fn cleanup(&mut self, device: &ash::Device) {
        self.pipeline.cleanup(device);
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_sampler(self.sampler, None);
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
    let layout_info = vk::DescriptorSetLayoutCreateInfo {
        binding_count: 1,
        p_bindings: &binding,
        ..Default::default()
    };
    unsafe { device.create_descriptor_set_layout(&layout_info, None) }
}

fn create_descriptor_set(
    device: &ash::Device,
    layout: vk::DescriptorSetLayout,
) -> Result<(vk::DescriptorPool, vk::DescriptorSet), vk::Result> {
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
        p_set_layouts: &layout,
        ..Default::default()
    };
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&allocate_info)?[0] };
    Ok((descriptor_pool, descriptor_set))
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

/// Shader modules stay alive because pipelines may be recreated after a swapchain change.
struct SkyShaders {
    vertex: ShaderStageInfo,
    fragment: ShaderStageInfo,
}

impl SkyShaders {
    fn load(device: &ash::Device) -> Result<Self, Box<dyn Error>> {
        let entry = c"main";
        Ok(Self {
            vertex: ShaderStageInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                shader_module: ShaderModule::from_spv_file(
                    device,
                    "assets/shaders/spv/sky.vert.spv",
                )?,
                entry_name: entry,
            },
            fragment: ShaderStageInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                shader_module: ShaderModule::from_spv_file(
                    device,
                    "assets/shaders/spv/sky.frag.spv",
                )?,
                entry_name: entry,
            },
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
