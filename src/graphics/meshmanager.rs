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

use ash::Device;
use crate::vulkan::base::VulkanBase;
use crate::graphics::mesh::LoadedMesh;
use crate::graphics::mesh::Mesh;

pub struct MeshManager {
    pub meshes: Vec<LoadedMesh>,
}

impl MeshManager {
    pub fn new() -> MeshManager {
        MeshManager {
            meshes: Vec::new(),
        }
    }

    // Cube
    pub fn request_cube(
        &mut self,
        vb: &VulkanBase
    ) -> usize {

        if let Some(idx) = self.meshes.iter().position(|m| m.name == "Cube") {
            idx
        } else {
            let mesh = LoadedMesh::cube(
                "Cube".into(),
                &vb.instance,
                &vb.device,
                vb.physical_device,
            ).expect("Failed to load mesh");
            self.meshes.push(mesh);
            self.meshes.len() - 1
        }
    }

    // Unit Plane
    pub fn request_unit_plane(
        &mut self,
        vb: &VulkanBase
    ) -> usize {

        if let Some(idx) = self.meshes.iter().position(|m| m.name == "UnitPlane") {
            idx
        } else {
            let mesh = LoadedMesh::unit_plane(
                "UnitPlane".into(),
                &vb.instance,
                &vb.device,
                vb.physical_device,
            ).expect("Failed to load mesh");
            self.meshes.push(mesh);
            self.meshes.len() - 1
        }
    }

    /// Uploads a CPU-side mesh directly into a LoadedMesh and returns its ID.
    pub fn request_mesh_from_cpu(
        &mut self,
        name: String,
        vb: &VulkanBase,
        mesh: Mesh,
    ) -> usize {
        // If we already have one by this name, return it.
        if let Some(idx) = self.meshes.iter().position(|m| m.name == name) {
            return idx;
        }
        // Otherwise upload to GPU...
        let loaded = crate::graphics::mesh::LoadedMesh::load(
            name.clone(),
            &vb.instance,
            &vb.device,
            vb.physical_device,
            &mesh,
        ).expect("Failed to upload CPU mesh");
        self.meshes.push(loaded);
        self.meshes.len() - 1
    }

    pub fn cleanup(&mut self, device: &Device) {
        for mesh in &mut self.meshes {
            println!("🗑️ Cleaning up mesh {}", mesh.name);
            mesh.cleanup(device);
        }
        self.meshes.clear();
        println!("🗑️ MeshManager cleaned up");
    }
}