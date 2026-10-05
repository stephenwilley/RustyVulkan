//! Grass shader ownership, vertex layout, and pipeline push constants.
//!
//! GrassShaders keeps each stage's SPIR-V so pipelines can be rebuilt later.

use super::{GrassInstance, GrassVertex};
use crate::graphics::gpu_data::GrassPushConstants;
use crate::graphics::shaders::ShaderStageInfo;
use ash::vk;
use std::error::Error;
use std::mem::offset_of;
pub(super) struct GrassShaders {
    pub(super) vertex: ShaderStageInfo,
    pub(super) fragment: ShaderStageInfo,
    pub(super) mid_fragment: ShaderStageInfo,
}

impl GrassShaders {
    pub(super) fn load() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            vertex: ShaderStageInfo::load(vk::ShaderStageFlags::VERTEX, "grass.vert.spv")?,
            fragment: ShaderStageInfo::load(vk::ShaderStageFlags::FRAGMENT, "grass.frag.spv")?,
            mid_fragment: ShaderStageInfo::load(
                vk::ShaderStageFlags::FRAGMENT,
                "grass_mid.frag.spv",
            )?,
        })
    }
}

pub(super) fn vertex_input_descriptions() -> (
    [vk::VertexInputBindingDescription; 2],
    [vk::VertexInputAttributeDescription; 5],
) {
    let bindings = [
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<GrassVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        },
        vk::VertexInputBindingDescription {
            binding: 1,
            stride: std::mem::size_of::<GrassInstance>() as u32,
            input_rate: vk::VertexInputRate::INSTANCE,
        },
    ];
    let attributes = [
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(GrassVertex, local_position) as u32,
        },
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 1,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(GrassVertex, local_normal) as u32,
        },
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 2,
            format: vk::Format::R32_SFLOAT,
            offset: offset_of!(GrassVertex, flower_head) as u32,
        },
        vk::VertexInputAttributeDescription {
            binding: 1,
            location: 3,
            format: vk::Format::R16G16B16A16_UNORM,
            offset: offset_of!(GrassInstance, position_height) as u32,
        },
        vk::VertexInputAttributeDescription {
            binding: 1,
            location: 4,
            format: vk::Format::R16G16B16A16_UNORM,
            offset: offset_of!(GrassInstance, rotation_width_tint_phase) as u32,
        },
    ];
    (bindings, attributes)
}

/// Supplies the named grass parameters using the same matrix convention as scene objects.
pub(super) fn grass_push_constants(
    camera: &crate::graphics::camera::Camera,
    time_seconds: f32,
    lod: f32,
) -> GrassPushConstants {
    GrassPushConstants::new(camera, time_seconds, 0.16, lod)
}
