//! --------------------------------------------------------------------------------------
//! Scene Module (scene.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Defines basic scene graph structures and transformations for objects.
//!
//! --------------------------------------------------------------------------------------

use cgmath::{Deg, Matrix4, Quaternion, Rotation3, Vector3};

/// A simple 3D transform: position, rotation (as a quaternion), uniform scale.
/// Shaders rely on the scale being uniform: they transform normals with `mat3(mv)`.
#[derive(Clone, Copy)]
pub struct Transform {
    pub translation: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub scale: f32,
}

impl Transform {
    /// Creates an identity transform (no translation, no rotation, scale 1.0).
    pub fn identity() -> Self {
        Self {
            translation: Vector3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            scale: 1.0,
        }
    }

    /// Creates a new Transform from Euler angles (degrees) about X, Y, and Z axes.
    /// # Arguments
    /// * `translation` - The position of the transform.
    /// * `euler_deg` - Contains rotation angles in degrees (X, Y, Z).
    /// * `scale` - A uniform scale factor.
    pub fn from_euler(translation: Vector3<f32>, euler_deg: Vector3<f32>, scale: f32) -> Self {
        // build each axis rotation
        let qx = Quaternion::from_angle_x(Deg(euler_deg.x));
        let qy = Quaternion::from_angle_y(Deg(euler_deg.y));
        let qz = Quaternion::from_angle_z(Deg(euler_deg.z));
        // combine them in Y→X→Z order (you can tweak order if you prefer)
        let rotation = qz * qx * qy;

        Transform {
            translation,
            rotation,
            scale,
        }
    }

    /// Sets Y so this uniformly scaled, upright object rests on a terrain height.
    ///
    /// `local_base_y` is the geometry's lowest object-local Y coordinate.  It is
    /// known to be `-1.0` for the unit cube; imported multipart objects calculate it
    /// from retained mesh bounds.  This is deliberately not physics or a raycast.
    pub fn place_on_ground(&mut self, ground_y: f32, local_base_y: f32) {
        self.translation.y = ground_y - local_base_y * self.scale;
    }

    /// Builds a 4x4 model matrix (Translation * Rotation * Scale).
    pub fn model_matrix(&self) -> Matrix4<f32> {
        let t = Matrix4::from_translation(self.translation);
        let r = Matrix4::from(self.rotation);
        let s = Matrix4::from_scale(self.scale);
        t * r * s
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

/// A drawable sub-part of a SceneObject (e.g., a glTF primitive).
///
/// Parts are cheap handles plus a transform, so cloning them shares the manager-owned
/// GPU mesh and material rather than copying either resource.
#[derive(Clone, Copy)]
pub struct ScenePart {
    /// Transform in the object's local/model space
    pub transform: Transform,
    pub material_id: usize,
    pub mesh_id: usize,
}

/// One thing in your world that you can draw. Holds a world-space transform and N parts.
#[derive(Clone)]
pub struct SceneObject {
    /// World/scene-space transform for the object root
    pub transform: Transform,
    /// Collection of draw parts (mesh+material) with optional local transforms
    pub parts: Vec<ScenePart>,
    /// Whether this object should be drawn
    pub visible: bool,
}

// No special constructor helpers; create SceneObject with parts explicitly.

/// A container holding *all* the drawables:
#[derive(Default)]
pub struct Scene {
    pub objects: Vec<SceneObject>,
}

impl Scene {
    /// Creates a new empty `Scene`.
    pub fn new() -> Self {
        Scene {
            objects: Vec::new(),
        }
    }
    /// Adds a `SceneObject` to the scene.
    /// # Arguments
    /// * `obj` - The `SceneObject` to add.
    pub fn add(&mut self, obj: SceneObject) {
        self.objects.push(obj);
    }
}
