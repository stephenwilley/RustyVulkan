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

use crate::graphics::material::Material;
use crate::vulkan::base::VulkanBase;
use ash::Device;
use std::error::Error;

pub struct MaterialManager {
    pub materials: Vec<Material>,
}

impl MaterialManager {
    pub fn new() -> MaterialManager {
        MaterialManager {
            materials: Vec::new(),
        }
    }

    pub fn recreate_pipelines(&mut self, vb: &VulkanBase) -> Result<(), Box<dyn Error>> {
        let mut first_error: Option<Box<dyn Error>> = None;
        for material in &mut self.materials {
            if let Err(e) = material.recreate_pipeline(vb) {
                if first_error.is_none() {
                    first_error = Some(e);
                }
            }
        }
        if let Some(err) = first_error {
            Err(err)
        } else {
            Ok(())
        }
    }

    pub fn request_material(
        &mut self,
        name: String,
        vb: &VulkanBase,
        vs_path: String,
        fs_path: String,
        diffuse_texture_path: Option<String>,
        normalmap_texture_path: Option<String>,
        depth_write: bool,
    ) -> Result<usize, Box<dyn Error>> {
        if let Some(idx) = self.materials.iter().position(|m| m.name == name) {
            Ok(idx)
        } else {
            let mat = Material::new(
                name,
                vb,
                vs_path,
                fs_path,
                diffuse_texture_path,
                normalmap_texture_path,
                depth_write,
            )?;
            self.materials.push(mat);
            Ok(self.materials.len() - 1)
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
