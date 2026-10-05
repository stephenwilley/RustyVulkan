//! CPU data whose byte layout is shared with GLSL. Arrays store matrix columns.
//! `Pod` rejects hidden padding, so `bytes_of` can expose the entire initialized value.

use crate::graphics::camera::Camera;
use bytemuck::{Pod, Zeroable};
use cgmath::Matrix4;

/// main.vert: two column-major mat4s and a vec4 (xy are UV tiling, zw are padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ScenePushConstants {
    pub mvp: [[f32; 4]; 4],
    pub mv: [[f32; 4]; 4],
    pub uv_tiling: [f32; 4],
}

impl ScenePushConstants {
    pub fn new(camera: &Camera, model: &Matrix4<f32>, uv_tiling: [f32; 2]) -> Self {
        let mv = camera.get_view() * model;
        let mvp = camera.get_projection() * mv;
        Self {
            mvp: mvp.into(),
            mv: mv.into(),
            uv_tiling: [uv_tiling[0], uv_tiling[1], 0.0, 0.0],
        }
    }
}

/// grass.vert: the final vec4 carries time, wind strength, LOD, and explicit padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GrassPushConstants {
    pub mvp: [[f32; 4]; 4],
    pub mv: [[f32; 4]; 4],
    pub time_seconds: f32,
    pub wind_strength: f32,
    pub lod: f32,
    pub _pad: f32,
}

impl GrassPushConstants {
    pub fn new(camera: &Camera, time_seconds: f32, wind_strength: f32, lod: f32) -> Self {
        let mv = *camera.get_view();
        Self {
            mvp: (camera.get_projection() * mv).into(),
            mv: mv.into(),
            time_seconds,
            wind_strength,
            lod,
            _pad: 0.0,
        }
    }
}

/// shadow_depth.vert: light-space projection multiplied by the model matrix.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ShadowPushConstants {
    pub mvp: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct GpuLight {
    pub position: [f32; 3],
    pub intensity: f32, // packs with position to 16 bytes
    pub color: [f32; 3],
    pub _pad: f32, // pad to 16 bytes
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct GpuDirLight {
    pub direction: [f32; 3],
    pub intensity: f32, // keep std140 friendly packing
    pub color: [f32; 3],
    pub _pad: f32,
}

/// Global (per-frame/per-image) uniform buffer object shared across all pipelines via **descriptor set 0, binding 0**.
/// There is **one buffer per swapchain image** so the CPU can update the UBO while another image is still in-flight.
/// Keep fields 16-byte aligned for std140. Field order must match every GLSL `GlobalUBO`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct GlobalUbo {
    pub dir_light: GpuDirLight,
    /// One view-to-light-clip matrix for each camera-depth slice.
    pub light_vp: [[[f32; 4]; 4]; crate::graphics::shadow_math::SHADOW_CASCADE_COUNT],
    /// Positive view-space far depth for each cascade.
    pub cascade_splits: [f32; crate::graphics::shadow_math::SHADOW_CASCADE_COUNT],
    // Then the array of point lights (std140 array of structs)
    pub lights: [GpuLight; crate::app::app::MAX_LIGHTS], // array of point lights
    pub light_count: u32,                                // number of active point lights
    pub _pad0: [u32; 3],                                 // three GLSL uints, never a uvec3 (std140)
}

// Vulkan/GLSL offsets are a contract, independent of Rust's chosen default layout.
const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<ScenePushConstants>() == 144);
    assert!(offset_of!(ScenePushConstants, mv) == 64);
    assert!(offset_of!(ScenePushConstants, uv_tiling) == 128);
    assert!(size_of::<GrassPushConstants>() == 144);
    assert!(offset_of!(GrassPushConstants, mv) == 64);
    assert!(offset_of!(GrassPushConstants, time_seconds) == 128);
    assert!(offset_of!(GrassPushConstants, wind_strength) == 132);
    assert!(offset_of!(GrassPushConstants, lod) == 136);
    assert!(size_of::<ShadowPushConstants>() == 64);
    assert!(size_of::<GpuLight>() == 32);
    assert!(size_of::<GpuDirLight>() == 32);
    assert!(size_of::<GlobalUbo>() == 576);
    assert!(offset_of!(GlobalUbo, light_vp) == 32);
    assert!(offset_of!(GlobalUbo, cascade_splits) == 288);
    assert!(offset_of!(GlobalUbo, lights) == 304);
    assert!(offset_of!(GlobalUbo, light_count) == 560);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_matrix_bytes_preserve_column_order() {
        let matrix = Matrix4::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let push = ShadowPushConstants { mvp: matrix.into() };
        let values: Vec<f32> = bytemuck::bytes_of(&push)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_ne_bytes(*bytes))
            .collect();
        assert_eq!(values, (1..=16).map(|n| n as f32).collect::<Vec<_>>());
    }

    #[test]
    fn named_grass_fields_match_the_shader_vec4() {
        let camera = Camera::new();
        let push = GrassPushConstants::new(&camera, 2.0, 0.16, 0.5);
        let params: Vec<f32> = bytemuck::bytes_of(&push)[128..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_ne_bytes(*bytes))
            .collect();
        assert_eq!(params, [2.0, 0.16, 0.5, 0.0]);
    }
}
