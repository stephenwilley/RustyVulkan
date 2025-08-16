//! --------------------------------------------------------------------------------------
//! Material Manager (materialmanager.rs)
//!
//! Created: July 2025  
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Objects can request a material.  If it already exists, the manager just returns an
//! index into the vector, otherwise it creates the Material, pushes it onto the Vec and
//! then returns the index
//!
//! --------------------------------------------------------------------------------------

use ash::Device;
use crate::graphics::material::Material;
use crate::vulkan::base::VulkanBase;
use vk_mem::Allocator;

/// Manages materials, ensuring that each material is only created once.
pub struct MaterialManager {
    pub materials: Vec<Material>,
}

/// Properties for creating a new material.
pub struct MaterialProperties {
    pub name: String,
    pub vs_path: String,
    pub fs_path: String,
    pub diffuse_texture_path: Option<String>,
    pub normalmap_texture_path: Option<String>,
    pub depth_write: bool,
}

impl MaterialManager {
    /// Creates a new `MaterialManager`.
    pub fn new() -> MaterialManager {
        MaterialManager {
            materials: Vec::new(),
        }
    }

    /// Recreates the pipelines for all materials.
    /// # Arguments
    /// * `vb` - The VulkanBase struct.
    /// # Returns
    /// * `Result<(), Box<dyn std::error::Error>>` - The result of the pipeline recreation.
    pub fn recreate_pipelines(
        &mut self,
        vb: &VulkanBase
    ) -> Result<(), Box<dyn std::error::Error>> {
        for material in &mut self.materials {
            if let Err(e) = material.recreate_pipeline(vb) {
                eprintln!("Failed to recreate pipeline: {}", e);
            }
        }
        Ok(())
    }

    /// Requests a material. If the material already exists, it returns the index of the existing material.
    /// Otherwise, it creates a new material and returns its index.
    /// # Arguments
    /// * `vb` - The VulkanBase struct.
    /// * `props` - The properties of the material to create.
    /// # Returns
    /// * `usize` - The index of the material.
    pub fn request_material(
        &mut self,
        vb: &VulkanBase,
        props: MaterialProperties,
    ) -> usize {
        if let Some(idx) = self.materials.iter().position(|m| m.name == props.name) {
            idx
        } else {
            let mat = Material::new(
                props.name,
                vb,
                props.vs_path,
                props.fs_path,
                props.diffuse_texture_path,
                props.normalmap_texture_path,
                props.depth_write
            ).unwrap();
            self.materials.push(mat);
            self.materials.len() - 1
        }
    }

    /// Cleans up all materials.
    /// # Arguments
    /// * `device` - The Vulkan device.
    /// * `allocator` - The global VMA allocator.
    pub fn cleanup(&mut self, device: &Device, allocator: &Allocator) {
        for material in &mut self.materials {
            println!("🗑️ Cleaning up material {}", material.name);
            material.cleanup(device, allocator);
        }
        self.materials.clear();
        println!("🗑️ MaterialManager cleaned up");
    }
}