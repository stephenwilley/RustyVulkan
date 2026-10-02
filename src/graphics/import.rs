//! --------------------------------------------------------------------------------------
//! Unified Import (import.rs)
//!
//! Created: September 2025
//! Author: Stephen Willey
//!
//! Single entry-point to import a model into a `SceneObject`, selecting the appropriate
//! backend based on file extension and enabled features.
//!
//! - russimp-ng/Assimp importer (always enabled)
//!
//! This keeps the rest of the app agnostic to the source format while preserving
//! your scene, materials, and procedural primitives like the infinite plane.
//!
//! --------------------------------------------------------------------------------------

use crate::app::scene::SceneObject;
use crate::graphics::materialmanager::MaterialManager;
use crate::graphics::meshmanager::MeshManager;
use crate::vulkan::base::VulkanBase;
use std::error::Error;

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

/// Imports one member of a model pack while reusing a specialised textured material.
pub fn import_model_as_object_with_shared_textured_material(
    path: &str,
    shared_material_name: &str,
    textured_shader_paths: [&str; 2],
    vb: &VulkanBase,
    meshes: &mut MeshManager,
    mats: &mut MaterialManager,
) -> Result<SceneObject, Box<dyn Error>> {
    crate::graphics::assimp_loader::import_model_as_object_with_shared_textured_material(
        path,
        shared_material_name,
        textured_shader_paths,
        vb,
        meshes,
        mats,
    )
}
