//! Camera-frustum fitting for cascaded directional shadows.

use cgmath::{EuclideanSpace, InnerSpace, Matrix4, Point3, SquareMatrix, Vector3};

use crate::graphics::camera::Camera;

/// Four cascades are a useful balance between directional-shadow quality and cost.
pub const SHADOW_CASCADE_COUNT: usize = 4;

/// Matrices and camera-space depth range belonging to one cascade.
#[derive(Clone, Copy)]
pub struct ShadowCascade {
    pub world_to_light_clip: Matrix4<f32>,
    pub view_to_light_clip: Matrix4<f32>,
    pub far_distance: f32,
}

/// Split the shadow range between linear and logarithmic spacing. The logarithmic
/// part concentrates detail near the camera without starving the distant view.
fn cascade_splits(camera: &Camera, shadow_distance: f32) -> [f32; SHADOW_CASCADE_COUNT] {
    let near = camera.get_near().max(0.001);
    let far = (near + shadow_distance.max(0.001)).min(camera.get_far());
    let lambda = 0.65;
    std::array::from_fn(|index| {
        let fraction = (index + 1) as f32 / SHADOW_CASCADE_COUNT as f32;
        let logarithmic = near * (far / near).powf(fraction);
        let linear = near + (far - near) * fraction;
        lambda * logarithmic + (1.0 - lambda) * linear
    })
}

/// Build the same cascade set used by the depth pass and the receiving shaders.
pub fn compute_shadow_cascades(
    camera: &Camera,
    sun_dir: [f32; 3],
    shadow_res: u32,
    shadow_distance: f32,
) -> [ShadowCascade; SHADOW_CASCADE_COUNT] {
    let splits = cascade_splits(camera, shadow_distance);
    let mut slice_near = camera.get_near();
    std::array::from_fn(|index| {
        let slice_far = splits[index];
        let cascade = fit_frustum_slice(camera, sun_dir, shadow_res, slice_near, slice_far);
        slice_near = slice_far;
        cascade
    })
}

/// Fit one orthographic light camera around one slice of the view frustum.
fn fit_frustum_slice(
    camera: &Camera,
    sun_dir: [f32; 3],
    shadow_res: u32,
    slice_near: f32,
    slice_far: f32,
) -> ShadowCascade {
    let view = *camera.get_view();
    let inv_view = view.invert().unwrap_or(Matrix4::identity());
    let camera_pos = Point3::new(inv_view.w.x, inv_view.w.y, inv_view.w.z);
    let right = Vector3::new(inv_view.x.x, inv_view.x.y, inv_view.x.z);
    let up = Vector3::new(inv_view.y.x, inv_view.y.y, inv_view.y.z);
    let forward = -Vector3::new(inv_view.z.x, inv_view.z.y, inv_view.z.z);

    let corners_world = frustum_corners(
        camera_pos,
        right,
        up,
        forward,
        (camera.get_fov_deg().to_radians() * 0.5).tan(),
        camera.get_aspect(),
        slice_near,
        slice_far,
    );

    let mut center = Vector3::new(0.0, 0.0, 0.0);
    for corner in &corners_world {
        center += corner.to_vec();
    }
    let center = Point3::from_vec(center / corners_world.len() as f32);
    let radius = corners_world
        .iter()
        .map(|corner| (corner - center).magnitude())
        .fold(0.0_f32, f32::max);

    let mut direction = Vector3::new(sun_dir[0], sun_dir[1], sun_dir[2]);
    direction /= direction.magnitude().max(1e-6);
    let light_up = if direction.y.abs() > 0.99 {
        Vector3::unit_z()
    } else {
        Vector3::unit_y()
    };
    // Orientation only: a light view that followed the slice would move the texel grid
    // with the camera, defeating the snapping below.
    let light_view = Matrix4::look_at_rh(Point3::origin(), Point3::from_vec(direction), light_up);

    let mut min_light = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max_light = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for corner in &corners_world {
        let point = light_view * corner.to_homogeneous();
        min_light.x = min_light.x.min(point.x);
        min_light.y = min_light.y.min(point.y);
        min_light.z = min_light.z.min(point.z);
        max_light.x = max_light.x.max(point.x);
        max_light.y = max_light.y.max(point.y);
        max_light.z = max_light.z.max(point.z);
    }

    // A square projection changes less as the camera rotates than a tight rectangle.
    let half_extent =
        ((max_light.x - min_light.x).max(max_light.y - min_light.y) * 0.51).max(0.001);
    let units_per_texel = (half_extent * 2.0 / shadow_res.max(1) as f32).max(1e-6);
    let center_x = ((min_light.x + max_light.x) * 0.5 / units_per_texel).floor() * units_per_texel;
    let center_y = ((min_light.y + max_light.y) * 0.5 / units_per_texel).floor() * units_per_texel;

    // Casters up to r + 1 sunward of the slice's nearest corner still land in the map;
    // the small margin beyond the far side keeps receivers at the slice's edge inside it.
    let near = -max_light.z - (radius + 1.0);
    let far = -min_light.z + radius * 0.25 + 0.5;
    let projection_gl = cgmath::ortho(
        center_x - half_extent,
        center_x + half_extent,
        center_y - half_extent,
        center_y + half_extent,
        near,
        far,
    );
    const OPENGL_TO_VULKAN: Matrix4<f32> = Matrix4::new(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
    );
    let world_to_light_clip = OPENGL_TO_VULKAN * projection_gl * light_view;

    ShadowCascade {
        world_to_light_clip,
        view_to_light_clip: world_to_light_clip * inv_view,
        far_distance: slice_far,
    }
}

#[allow(clippy::too_many_arguments)]
fn frustum_corners(
    camera_pos: Point3<f32>,
    right: Vector3<f32>,
    up: Vector3<f32>,
    forward: Vector3<f32>,
    half_tangent: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> [Point3<f32>; 8] {
    let plane = |distance: f32| {
        let half_height = half_tangent * distance;
        let half_width = half_height * aspect;
        let center = camera_pos + forward * distance;
        [
            center - right * half_width - up * half_height,
            center + right * half_width - up * half_height,
            center + right * half_width + up * half_height,
            center - right * half_width + up * half_height,
        ]
    };
    let near_corners = plane(near);
    let far_corners = plane(far);
    [
        near_corners[0],
        near_corners[1],
        near_corners[2],
        near_corners[3],
        far_corners[0],
        far_corners[1],
        far_corners[2],
        far_corners[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascade_splits_are_ordered_and_end_at_shadow_distance() {
        let camera = Camera::new();
        let splits = cascade_splits(&camera, 75.0);

        assert!(splits.windows(2).all(|pair| pair[0] < pair[1]));
        assert!((splits[SHADOW_CASCADE_COUNT - 1] - 75.5).abs() < 0.001);
    }

    #[test]
    fn moving_the_camera_shifts_cascades_by_whole_texels() {
        let res = 2048;
        let mut camera = Camera::new();
        camera.set_view_yxz(Point3::new(1.0, 2.0, 3.0), 30.0, -10.0, 0.0);
        let before = compute_shadow_cascades(&camera, [0.3, -1.0, 0.2], res, 75.0);
        camera.set_view_yxz(Point3::new(1.37, 2.0, 3.21), 30.0, -10.0, 0.0);
        let after = compute_shadow_cascades(&camera, [0.3, -1.0, 0.2], res, 75.0);

        // Orthographic: clip.x = row0 · p + w.x, so a change in w.x is a shift of
        // (res / 2) texels.
        for (a, b) in before.iter().zip(&after) {
            for row in 0..2 {
                let shift_texels = (b.world_to_light_clip.w[row] - a.world_to_light_clip.w[row])
                    * res as f32
                    / 2.0;
                assert!(
                    (shift_texels - shift_texels.round()).abs() < 0.05,
                    "cascade shifted by {shift_texels} texels"
                );
            }
        }
    }

    #[test]
    fn each_cascade_contains_its_view_slice() {
        let camera = Camera::new();
        let cascades = compute_shadow_cascades(&camera, [0.3, -1.0, 0.2], 2048, 75.0);
        let inv_view = camera.get_view().invert().unwrap();
        let splits = cascade_splits(&camera, 75.0);
        let mut slice_near = camera.get_near();
        for (cascade, slice_far) in cascades.iter().zip(splits) {
            let corners = frustum_corners(
                Point3::new(inv_view.w.x, inv_view.w.y, inv_view.w.z),
                inv_view.x.truncate(),
                inv_view.y.truncate(),
                -inv_view.z.truncate(),
                (camera.get_fov_deg().to_radians() * 0.5).tan(),
                camera.get_aspect(),
                slice_near,
                slice_far,
            );
            for corner in corners {
                let clip = cascade.world_to_light_clip * corner.to_homogeneous();
                assert!(
                    clip.x.abs() <= 1.0 && clip.y.abs() <= 1.0,
                    "corner outside map"
                );
                assert!((0.0..=1.0).contains(&clip.z), "corner outside depth range");
            }
            slice_near = slice_far;
        }
    }

    #[test]
    fn fitted_cascade_matrices_are_finite() {
        let camera = Camera::new();
        let cascades = compute_shadow_cascades(&camera, [0.3, -1.0, 0.2], 2048, 75.0);

        for cascade in cascades {
            let matrix: &[[f32; 4]; 4] = cascade.world_to_light_clip.as_ref();
            assert!(matrix.iter().flatten().all(|value| value.is_finite()));
        }
    }
}
