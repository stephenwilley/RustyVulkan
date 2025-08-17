//! --------------------------------------------------------------------------------------
//! Mesh Manager (meshmanager.rs)
//!
//! Created: July 2025  
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Objects can request a mesh.  If it already exists, the manager just returns an
//! index into the vector, otherwise it creates the Mesh, pushes it onto the Vec and
//! then returns the index
//!
//! --------------------------------------------------------------------------------------

use crate::vulkan::base::VulkanBase;
use crate::graphics::mesh::LoadedMesh;
use crate::graphics::mesh::Mesh;
use std::error::Error;
use vk_mem::Allocator;

/// Manages meshes, ensuring that each mesh is only created once.
pub struct MeshManager {
    pub meshes: Vec<LoadedMesh>,
}

impl MeshManager {
    /// Creates a new `MeshManager`.
    pub fn new() -> MeshManager {
        MeshManager {
            meshes: Vec::new(),
        }
    }

    /// Requests a cube mesh. If the mesh already exists, it returns the index of the existing mesh.
    /// Otherwise, it creates a new cube mesh and returns its index.
    /// # Arguments
    /// * `vb` - The VulkanBase struct.
    /// # Returns
    /// * `Result<usize, Box<dyn Error>>` - The index of the cube mesh on success, or an error on failure.
    pub fn request_cube(
        &mut self,
        vb: &VulkanBase
    ) -> Result<usize, Box<dyn Error>> {

        if let Some(idx) = self.meshes.iter().position(|m| m.name == "Cube") {
            Ok(idx)
        } else {
            let mesh = LoadedMesh::cube(
                "Cube".into(),
                vb.allocator.as_ref().unwrap(),
            )?;
            self.meshes.push(mesh);
            Ok(self.meshes.len() - 1)
        }
    }

    /// Requests a unit plane mesh. If the mesh already exists, it returns the index of the existing mesh.
    /// Otherwise, it creates a new unit plane mesh and returns its index.
    /// # Arguments
    /// * `vb` - The VulkanBase struct.
    /// # Returns
    /// * `Result<usize, Box<dyn Error>>` - The index of the unit plane mesh on success, or an error on failure.
    pub fn request_unit_plane(
        &mut self,
        vb: &VulkanBase
    ) -> Result<usize, Box<dyn Error>> {

        if let Some(idx) = self.meshes.iter().position(|m| m.name == "UnitPlane") {
            Ok(idx)
        } else {
            let mesh = LoadedMesh::unit_plane(
                "UnitPlane".into(),
                vb.allocator.as_ref().unwrap(),
            )?;
            self.meshes.push(mesh);
            Ok(self.meshes.len() - 1)
        }
    }

    /// Uploads a CPU-side mesh directly into a LoadedMesh and returns its ID.
    /// # Arguments
    /// * `name` - The name of the mesh.
    /// * `vb` - The VulkanBase struct.
    /// * `mesh` - The CPU-side mesh data.
    /// # Returns
    /// * `Result<usize, Box<dyn Error>>` - The index of the loaded mesh on success, or an error on failure.
    pub fn request_mesh_from_cpu(
        &mut self,
        name: String,
        vb: &VulkanBase,
        mesh: Mesh,
    ) -> Result<usize, Box<dyn Error>> {
        // If we already have one by this name, return it.
        if let Some(idx) = self.meshes.iter().position(|m| m.name == name) {
            return Ok(idx);
        }
        // Otherwise upload to GPU...
        let loaded = crate::graphics::mesh::LoadedMesh::load(
            name.clone(),
            vb.allocator.as_ref().unwrap(),
            &mesh,
        )?;
        self.meshes.push(loaded);
        Ok(self.meshes.len() - 1)
    }

    /// Cleans up all meshes.
    /// # Arguments
    /// * `device` - The Vulkan device.
    pub fn cleanup(&mut self, allocator: &Allocator) {
        for mesh in &mut self.meshes {
            println!("🗑️ Cleaning up mesh {}", mesh.name);
            mesh.cleanup(allocator);
        }
        self.meshes.clear();
        println!("🗑️ MeshManager cleaned up");
    }
}