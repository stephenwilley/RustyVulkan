//! Shadow math helpers for computing a tight light projection that follows
//! the camera frustum. Kept small and well-documented for learning.

use cgmath::{EuclideanSpace, InnerSpace, SquareMatrix};
use cgmath::{Matrix4, Point3, Vector3};

use crate::graphics::camera::Camera;

/// Output matrices for a tight directional-light fit.
/// - `world_to_light_clip`: Use for rendering the shadow map (MVP = world_to_light_clip * model)
/// - `view_to_light_clip`:  Use in main pass when transforming view-space positions
///   directly (e.g. vFragPosView) to light clip space for shadow lookups.
pub struct TightLightMats {
    pub world_to_light_clip: Matrix4<f32>,
    pub view_to_light_clip: Matrix4<f32>,
}

/// Compute a tight-fitting directional light projection around the camera frustum.
///
/// Steps:
/// 1) Reconstruct the 8 frustum corners by inverting the camera's view-projection and
///    transforming the Vulkan NDC cube corners (x,y∈[-1,1], z∈[0,1]).
/// 2) Build a light view looking from `sun_dir` toward the frustum center. To keep
///    it stable, choose an alternate up vector if the direction is nearly vertical.
/// 3) Transform corners into light view space and compute an axis-aligned bounding box (AABB).
/// 4) Optionally snap the AABB center to the shadow texel grid to reduce shimmering.
/// 5) Build an orthographic projection from the AABB and correct for Vulkan depth range.
pub fn compute_tight_light_mats(
    camera: &Camera,
    sun_dir: [f32; 3],
    shadow_res: u32,
    shadow_distance: f32,
) -> TightLightMats {
    // 1) Build a camera-aligned frustum slice using FOV/aspect and a limited far distance.
    // This avoids wasting shadow resolution on distant, irrelevant regions.
    let view = *camera.get_view();
    let inv_view = view.invert().unwrap_or(Matrix4::identity());
    let cam_pos = Point3::new(inv_view.w.x, inv_view.w.y, inv_view.w.z);
    let right = Vector3::new(inv_view.x.x, inv_view.x.y, inv_view.x.z);
    let up = Vector3::new(inv_view.y.x, inv_view.y.y, inv_view.y.z);
    let forward = -Vector3::new(inv_view.z.x, inv_view.z.y, inv_view.z.z);

    let fov_rad = camera.get_fov_deg().to_radians();
    let aspect = camera.get_aspect();
    let cam_near = camera.get_near();
    let cam_far = camera.get_far();
    // Limit far slice by a practical shadow distance
    let slice_far = (cam_near + shadow_distance).min(cam_far);

    let corners_world: [Point3<f32>; 8] = {
        // Half‑sizes of the frustum at the near and far slice planes
        let near_half_height = (fov_rad * 0.5).tan() * cam_near;
        let near_half_width = near_half_height * aspect;
        let far_half_height = (fov_rad * 0.5).tan() * slice_far;
        let far_half_width = far_half_height * aspect;

        // Centers of the near and far planes in world space
        let near_center = cam_pos + forward * cam_near;
        let far_center = cam_pos + forward * slice_far;

        [
            // Near plane (left/right × down/up)
            near_center - right * near_half_width - up * near_half_height, // left  - down
            near_center + right * near_half_width - up * near_half_height, // right - down
            near_center + right * near_half_width + up * near_half_height, // right - up
            near_center - right * near_half_width + up * near_half_height, // left  - up
            // Far plane (left/right × down/up)
            far_center - right * far_half_width - up * far_half_height,
            far_center + right * far_half_width - up * far_half_height,
            far_center + right * far_half_width + up * far_half_height,
            far_center - right * far_half_width + up * far_half_height,
        ]
    };

    // Center and radius of the frustum
    let mut center = Vector3::new(0.0, 0.0, 0.0);
    for c in &corners_world {
        center += c.to_vec();
    }
    center /= 8.0;
    let center_p = Point3::from_vec(center);
    let mut radius: f32 = 0.0;
    for c in &corners_world {
        radius = radius.max((c - center_p).magnitude());
    }

    // 2) Light view matrix looking toward the frustum center
    // Robust normalization (no magic fallback vector):
    let mut dir = Vector3::new(sun_dir[0], sun_dir[1], sun_dir[2]);
    let len = dir.magnitude().max(1e-6);
    dir /= len;
    let eye = center_p - dir * (radius * 2.0 + 1.0);
    // Pick an up vector roughly perpendicular to dir
    let mut up = Vector3::new(0.0, 1.0, 0.0);
    if dir.dot(up).abs() > 0.99 {
        up = Vector3::new(0.0, 0.0, 1.0);
    }
    let light_view = Matrix4::look_at_rh(eye, center_p, up);

    // 3) Light-space AABB of frustum corners
    let mut min_l = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max_l = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for c in &corners_world {
        let v = light_view * c.to_homogeneous();
        let v3 = Vector3::new(v.x, v.y, v.z);
        min_l.x = min_l.x.min(v3.x);
        min_l.y = min_l.y.min(v3.y);
        min_l.z = min_l.z.min(v3.z);
        max_l.x = max_l.x.max(v3.x);
        max_l.y = max_l.y.max(v3.y);
        max_l.z = max_l.z.max(v3.z);
    }

    // Slight padding to reduce clipping risk
    let pad_xy = 0.01 * (max_l.x - min_l.x + max_l.y - min_l.y).max(1e-3);
    let pad_z = 0.5;
    min_l.x -= pad_xy;
    max_l.x += pad_xy;
    min_l.y -= pad_xy;
    max_l.y += pad_xy;
    min_l.z -= pad_z;
    max_l.z += pad_z;

    // 4) Texel snapping: align the ortho center to the shadow map grid to reduce shimmering
    let res_f = shadow_res as f32;
    let width = max_l.x - min_l.x;
    let height = max_l.y - min_l.y;
    let units_per_texel_x = (width / res_f).max(1e-6);
    let units_per_texel_y = (height / res_f).max(1e-6);
    let mut center_x = 0.5 * (min_l.x + max_l.x);
    let mut center_y = 0.5 * (min_l.y + max_l.y);
    center_x = (center_x / units_per_texel_x).floor() * units_per_texel_x;
    center_y = (center_y / units_per_texel_y).floor() * units_per_texel_y;
    let half_w = 0.5 * width;
    let half_h = 0.5 * height;
    let left = center_x - half_w;
    let right = center_x + half_w;
    let bottom = center_y - half_h;
    let top = center_y + half_h;
    // Convert light-view z (negative in front) to positive near/far distances as
    // expected by a right-handed orthographic projection (like OpenGL).
    // Points in front: z_l in [-far_z, -near_z].
    // IMPORTANT: fix near to a small constant to avoid near-plane clipping of casters
    // when the distribution of corners makes max_l.z quite negative.
    let min_z = min_l.z; // most negative (furthest forward along light look direction)
    let eps = 1e-3;
    let near = eps; // keep tiny near; ortho precision impact is negligible
    let mut far = (-min_z).max(near + eps);
    // A little extra padding to be safe on far
    far += 0.5;

    // 5) Build Vulkan-corrected orthographic projection
    let proj_gl = cgmath::ortho(left, right, bottom, top, near, far);
    const OPENGL_TO_VULKAN_MATRIX: Matrix4<f32> = Matrix4::new(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
    );
    let proj_vk = OPENGL_TO_VULKAN_MATRIX * proj_gl;

    // World-space and view-space variants
    let world_to_light_clip = proj_vk * light_view;
    let view_to_light_clip = world_to_light_clip * view.invert().unwrap_or(Matrix4::identity());

    TightLightMats {
        world_to_light_clip,
        view_to_light_clip,
    }
}
