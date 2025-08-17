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
}

impl Camera {
    /// Create a new Camera with default FOV/aspect and looking at the origin.
    pub fn new() -> Self {
        let mut cam = Camera {
            projection: Matrix4::identity(),
            view:       Matrix4::identity(),
            position:   Point3::new(-2.0, 3.0, 8.0),
            yaw:        12.0,
            pitch:      -15.0,
            roll:       0.0,
        };
        cam.set_perspective_projection(45.0, 16.0/9.0, 0.1, 100.0);
        cam.set_view_yxz(cam.position, cam.yaw, cam.pitch, cam.roll);
        cam
    }

    /// Set up a perspective projection.
    /// # Arguments
    /// * `fov_deg` - vertical field of view in degrees  
    /// * `aspect` - width/height ratio  
    /// * `near` - near clipping plane  
    /// * `far` - far clipping plane
    pub fn set_perspective_projection(
        &mut self,
        fov_deg: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) {
        let mut proj = perspective(Deg(fov_deg), aspect, near, far);
        // Vulkan’s NDC has Y pointing down, so flip it
        proj.y.y *= -1.0;
        self.projection = proj;
    }

    /// Define the camera’s position & orientation using yaw & pitch (degrees).
    /// We ignore roll for now and build the view via `look_at_rh`.
    /// # Arguments
    /// * `position` - The position of the camera.
    /// * `yaw_deg` - The yaw of the camera in degrees.
    /// * `pitch_deg` - The pitch of the camera in degrees.
    /// * `roll_deg` - The roll of the camera in degrees.
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
            x: yaw_rad.sin()  * pitch_rad.cos(),
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

    /// Rotate the camera by the given yaw and pitch angles (in degrees).
    /// The camera's position will not be changed.
    /// # Arguments
    /// * `delta_yaw` - The change in yaw in degrees.
    /// * `delta_pitch` - The change in pitch in degrees.
    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw   += delta_yaw;
        self.pitch  = (self.pitch + delta_pitch).clamp(-89.0, 89.0);
        // Recompute the view matrix with the new angles:
        self.set_view_yxz(self.position, self.yaw, self.pitch, self.roll);
    }

    /// Move the camera along its local forward and right axes.
    /// # Arguments
    /// * `forward_amt` - The amount to move forward. `forward_amt` > 0 moves you “into” the scene, `<0` moves you back.
    /// * `right_amt` - The amount to move right. `right_amt` > 0 strafes you right,  `<0` strafes you left.
    pub fn translate(&mut self, forward_amt: f32, right_amt: f32) {
        let yaw_rad: Rad<f32> = Deg(self.yaw).into();
        let pitch_rad: Rad<f32> = Deg(self.pitch).into();
        // Forward vector in XZ plane
        let forward_dir = Vector3 {
            x: yaw_rad.sin(),
            y: pitch_rad.sin(),
            z: -yaw_rad.cos(),
        }
        .normalize();
        // Right is cross(forward, up)
        let right_dir = forward_dir.cross(Vector3::unit_y()).normalize();

        // Move position
        self.position += (forward_dir * forward_amt) + (right_dir * right_amt);

        // Rebuild view matrix at new position
        self.set_view_yxz(self.position, self.yaw, self.pitch, self.roll);
    }

    /// Get the projection matrix (for your MVP computation).
    pub fn get_projection(&self) -> &Matrix4<f32> {
        &self.projection
    }

    /// Get the view matrix (for your MVP computation).
    pub fn get_view(&self) -> &Matrix4<f32> {
        &self.view
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera::new()
    }
}