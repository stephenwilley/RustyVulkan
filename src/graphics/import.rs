//! --------------------------------------------------------------------------------------
//! Unified Import (import.rs)
//!
//! Created: September 2025
//! Author: Stephen Willey
//!
//! Single entry-point to import a model into a `SceneObject`, selecting the appropriate
//! backend based on file extension and enabled features.
//!
//! - russimp/Assimp importer (behind `russimp` feature)
//!
//! This keeps the rest of the app agnostic to the source format while preserving
//! your scene, materials, and procedural primitives like the infinite plane.
//!
//! --------------------------------------------------------------------------------------

use std::error::Error;
use crate::vulkan::base::VulkanBase;
use crate::graphics::meshmanager::MeshManager;
use crate::graphics::materialmanager::MaterialManager;
use crate::app::scene::SceneObject;

/// Import a model at `path` into a single `SceneObject` containing one part per primitive,
/// using the russimp/Assimp importer.
pub fn import_model_as_object(
    path: &str,
    vb: &VulkanBase,
    meshes: &mut MeshManager,
    mats: &mut MaterialManager,
) -> Result<SceneObject, Box<dyn Error>> {
    crate::graphics::assimp_loader::import_model_as_object(path, vb, meshes, mats)
}
