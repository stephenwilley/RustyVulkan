// src/scene.rs

use cgmath::{Matrix4, Vector3, Quaternion, Deg, Rotation3};

/// A simple 3D transform: position, rotation (as a quaternion), uniform scale.
#[derive(Clone, Copy)]
pub struct Transform {
    pub translation: Vector3<f32>,
    pub rotation:    Quaternion<f32>,
    pub scale:       f32,
}

impl Transform {
    /// Creates an identity transform (no translation, no rotation, scale 1.0).
    pub fn identity() -> Self {
        Self {
            translation: Vector3::new(0.0, 0.0, 0.0),
            rotation:    Quaternion::new(1.0, 0.0, 0.0, 0.0),
            scale:       1.0,
        }
    }

    /// Creates a new Transform from Euler angles (degrees) about X, Y, and Z axes.
    /// # Arguments
    /// * `translation` - The position of the transform.
    /// * `euler_deg` - Contains rotation angles in degrees (X, Y, Z).
    /// * `scale` - A uniform scale factor.
    pub fn from_euler(
        translation: Vector3<f32>,
        euler_deg: Vector3<f32>,
        scale: f32,
    ) -> Self {
        // build each axis rotation
        let qx = Quaternion::from_angle_x(Deg(euler_deg.x));
        let qy = Quaternion::from_angle_y(Deg(euler_deg.y));
        let qz = Quaternion::from_angle_z(Deg(euler_deg.z));
        // combine them in Y→X→Z order (you can tweak order if you prefer)
        let rotation = qz * qx * qy;

        Transform { translation, rotation, scale }
    }

    /// Builds a 4x4 model matrix (Translation * Rotation * Scale).
    pub fn model_matrix(&self) -> Matrix4<f32> {
        let t = Matrix4::from_translation(self.translation);
        let r = Matrix4::from(self.rotation);
        let s = Matrix4::from_scale(self.scale);
        t * r * s
    }
}

/// One thing in your world that you can draw:
pub struct SceneObject {
    pub transform: Transform,
    pub material_id: usize,
    pub mesh_id: usize,
}

/// A container holding *all* the drawables:
#[derive(Default)]
pub struct Scene {
    pub objects: Vec<SceneObject>,
}

impl Scene {
    /// Creates a new empty `Scene`.
    pub fn new() -> Self {
        Scene { objects: Vec::new() }
    }
    /// Adds a `SceneObject` to the scene.
    /// # Arguments
    /// * `obj` - The `SceneObject` to add.
    pub fn add(&mut self, obj: SceneObject) {
        self.objects.push(obj);
    }
}