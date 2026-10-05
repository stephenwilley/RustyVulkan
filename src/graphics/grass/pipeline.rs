//! Grass shader ownership, vertex layout, and pipeline push constants.
//!
//! GrassShaders keeps each stage's SPIR-V so pipelines can be rebuilt later.

use super::{GrassInstance, GrassVertex};
use crate::graphics::shaders::ShaderStageInfo;
use crate::vulkan::main_pass::compute_push_constant_per_obj;
use ash::vk;
use cgmath::{Matrix4, SquareMatrix};
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
            vertex: ShaderStageInfo::load(
                vk::ShaderStageFlags::VERTEX,
                "assets/shaders/spv/grass.vert.spv",
            )?,
            fragment: ShaderStageInfo::load(
                vk::ShaderStageFlags::FRAGMENT,
                "assets/shaders/spv/grass.frag.spv",
            )?,
            mid_fragment: ShaderStageInfo::load(
                vk::ShaderStageFlags::FRAGMENT,
                "assets/shaders/spv/grass_mid.frag.spv",
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

/// Reuses the application's standard push-constant packing and fills the grass-only LOD slot.
pub(super) fn grass_push_constants(
    camera: &crate::graphics::camera::Camera,
    time_seconds: f32,
    lod: f32,
) -> [u8; 144] {
    let mut bytes =
        compute_push_constant_per_obj(camera, &Matrix4::identity(), [time_seconds, 0.16]);
    bytes[136..140].copy_from_slice(&lod.to_ne_bytes());
    bytes
}
