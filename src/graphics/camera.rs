//! --------------------------------------------------------------------------------------
//! Camera Module (camera.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Provides a simple perspective camera used by the renderer.
//!
//! --------------------------------------------------------------------------------------

use cgmath::{Matrix4, Point3, Vector3, Deg, Rad, perspective};
use cgmath::prelude::*;

/// A simple camera with perspective projection and Y–X–Z Euler view control.
pub struct Camera {
    projection: Matrix4<f32>,
    view:       Matrix4<f32>,
    position:   Point3<f32>,
    yaw:        f32, // degrees
    pitch:      f32,
    roll:       f32,
    // Persisted projection parameters so callers can adjust aspect without
    // clobbering near/far or fov.
    fov_deg:    f32,
    aspect:     f32,
    near:       f32,
    far:        f32,
}

impl Camera {
    /// Creates a new camera with a default perspective and view.
    ///
    /// - Default projection: `fov=45°`, `aspect=16:9`, `near=0.5`, `far=5.0`.
    /// - Projection is Vulkan-corrected (Y flipped, depth range 0..1).
    /// - Default view looks toward the origin from `(-2, 3, 8)`.
    ///
    /// Use [`set_perspective_projection`] to change fov/near/far/aspect or
    /// [`set_aspect`] during window resizes to preserve fov/near/far.
    pub fn new() -> Self {
        let mut cam = Camera {
            projection: Matrix4::identity(),
            view:       Matrix4::identity(),
            position:   Point3::new(-2.0, 3.5, 8.0),
            yaw:        12.0,
            pitch:      -15.0,
            roll:       0.0,
            fov_deg:    45.0,
            aspect:     16.0/9.0,
            near:       0.5,
            far:        75.0,
        };
        cam.rebuild_projection();
        cam.set_view_yxz(cam.position, cam.yaw, cam.pitch, cam.roll);
        cam
    }

    /// Updates only the aspect ratio, preserving `fov/near/far`.
    ///
    /// Prefer calling this from a window-resize handler so your chosen
    /// `near`/`far` are not overwritten.
    pub fn set_aspect(&mut self, aspect: f32) {
        // Minimized windows report a zero height.  Keeping the last valid
        // projection avoids NaNs propagating through the renderer.
        if aspect.is_finite() && aspect > 0.0 {
            self.aspect = aspect;
            self.rebuild_projection();
        }
    }

    /// Rebuilds the projection matrix from stored parameters.
    fn rebuild_projection(&mut self) {
        let mut proj = perspective(Deg(self.fov_deg), self.aspect, self.near, self.far);
        // Flip Y to match Vulkan framebuffer coordinate conventions used here
        proj.y.y *= -1.0;
        // Map GL clip-space Z [-1,1] to Vulkan's [0,1]
        pub const OPENGL_TO_VULKAN_MATRIX: Matrix4<f32> = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0,
        );
        self.projection = OPENGL_TO_VULKAN_MATRIX * proj;
    }

    /// Defines the camera’s position and orientation using yaw–pitch in degrees.
    ///
    /// Builds a right-handed view matrix via `look_at_rh` (roll is currently ignored).
    ///
    /// Arguments
    /// - `position`: World position.
    /// - `yaw_deg`:  Yaw angle in degrees (rotation around +Y).
    /// - `pitch_deg`: Pitch angle in degrees (rotation around +X).
    /// - `roll_deg`:  Roll angle in degrees (currently unused).
    pub fn set_view_yxz(
        &mut self,
        position: Point3<f32>,
        yaw_deg: f32,
        pitch_deg: f32,
        _roll_deg: f32,
    ) {
        self.position = position;
        self.yaw      = yaw_deg;
        self.pitch    = pitch_deg;

        // Compute forward direction from yaw & pitch:
        let yaw_rad: Rad<f32>   = Deg(yaw_deg).into();
        let pitch_rad: Rad<f32> = Deg(pitch_deg).into();        
        let forward = Vector3 {
            x: yaw_rad.sin() * pitch_rad.cos(),
            y: pitch_rad.sin(),
            z: -yaw_rad.cos() * pitch_rad.cos(),
        }.normalize();

        // Look from `position` toward `position + forward`, with +Y up:
        self.view = Matrix4::look_at_rh(
            position,
            position + forward,
            Vector3::unit_y(),
        );
    }

    /// Rotates the camera by yaw/pitch (degrees) without changing position.
    ///
    /// Pitch is clamped to `[-89°, 89°]` to avoid gimbal lock.
    ///
    /// Arguments
    /// - `delta_yaw`:   Added to current yaw, in degrees.
    /// - `delta_pitch`: Added to current pitch, in degrees.
    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw   += delta_yaw;
        self.pitch  = (self.pitch + delta_pitch).clamp(-89.0, 89.0);
        // Recompute the view matrix with the new angles:
        self.set_view_yxz(self.position, self.yaw, self.pitch, self.roll);
    }

    /// Moves the camera in the horizontal X/Z plane using its yaw only.
    ///
    /// Arguments
    /// - `forward_distance`: > 0 moves forward; < 0 moves back.
    /// - `right_distance`:   > 0 strafes right; < 0 strafes left.
    ///
    /// Looking up or down therefore never makes first-person movement fly or sink.
    pub fn translate_horizontal(&mut self, forward_distance: f32, right_distance: f32) {
        let yaw_rad: Rad<f32> = Deg(self.yaw).into();
        // Discard pitch: WASD movement stays level while the view can look anywhere.
        let forward_dir = Vector3 {
            x: yaw_rad.sin(),
            y: 0.0,
            z: -yaw_rad.cos(),
        };
        let right_dir = forward_dir.cross(Vector3::unit_y()).normalize();

        self.position += (forward_dir * forward_distance) + (right_dir * right_distance);

        self.set_view_yxz(self.position, self.yaw, self.pitch, self.roll);
    }

    /// Raises the camera to a fixed eye height above a sampled terrain height.
    pub fn set_height_above_ground(&mut self, ground_height: f32, eye_height: f32) {
        self.position.y = ground_height + eye_height;
        self.set_view_yxz(self.position, self.yaw, self.pitch, self.roll);
    }

    /// Returns the current world-space camera position.
    pub fn position(&self) -> Point3<f32> {
        self.position
    }

    /// Returns the Vulkan-corrected projection matrix.
    ///
    /// The matrix has Y flipped and maps clip-space Z to Vulkan’s `[0, 1]`.
    pub fn get_projection(&self) -> &Matrix4<f32> {
        &self.projection
    }

    /// Returns the right-handed view matrix (world → view).
    pub fn get_view(&self) -> &Matrix4<f32> {
        &self.view
    }

    /// Returns the vertical field of view in degrees.
    pub fn get_fov_deg(&self) -> f32 { self.fov_deg }

    /// Returns the current aspect ratio (width/height).
    pub fn get_aspect(&self) -> f32 { self.aspect }

    /// Returns the near clipping plane distance.
    pub fn get_near(&self) -> f32 { self.near }

    /// Returns the far clipping plane distance.
    pub fn get_far(&self) -> f32 { self.far }
}

impl Default for Camera {
    fn default() -> Self {
        Camera::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Camera;

    #[test]
    fn invalid_aspect_does_not_replace_the_last_valid_projection() {
        let mut camera = Camera::new();
        camera.set_aspect(4.0 / 3.0);
        let projection = *camera.get_projection();

        camera.set_aspect(0.0);
        assert_eq!(camera.get_aspect(), 4.0 / 3.0);
        assert_eq!(*camera.get_projection(), projection);

        camera.set_aspect(f32::NAN);
        assert_eq!(camera.get_aspect(), 4.0 / 3.0);
        assert_eq!(*camera.get_projection(), projection);
    }

    #[test]
    fn horizontal_motion_and_ground_lock_keep_the_expected_height() {
        let mut camera = Camera::new();
        let original_height = camera.position().y;

        camera.translate_horizontal(2.0, 1.0);
        assert_eq!(camera.position().y, original_height);

        camera.set_height_above_ground(0.25, 1.8);
        assert_eq!(camera.position().y, 2.05);
    }
}
