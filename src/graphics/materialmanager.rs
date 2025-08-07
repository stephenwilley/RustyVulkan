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

pub struct MaterialManager {
    pub materials: Vec<Material>,
}

impl MaterialManager {
    pub fn new() -> MaterialManager {
        MaterialManager {
            materials: Vec::new(),
        }
    }

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

    pub fn request_material(
        &mut self,
        name: String,
        vb: &VulkanBase,
        vs_path: String,
        fs_path: String,
        diffuse_texture_path: Option<String>,
        normalmap_texture_path: Option<String>,
        depth_write: bool
    ) -> usize {
        if let Some(idx) = self.materials.iter().position(|m| m.name == name) {
            idx
        } else {
            let mat = Material::new(
                name,
                vb,
                vs_path,
                fs_path,
                diffuse_texture_path,
                normalmap_texture_path,
                depth_write
            ).unwrap();
            self.materials.push(mat);
            self.materials.len() - 1
        }
    }

    pub fn cleanup(&mut self, device: &Device) {
        for material in &mut self.materials {
            println!("🗑️ Cleaning up material {}", material.name);
            material.cleanup(device);
        }
        self.materials.clear();
        println!("🗑️ MaterialManager cleaned up");
    }
}