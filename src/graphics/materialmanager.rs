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

use ash::{Device, vk};
use crate::graphics::material::Material;
use crate::graphics::texture::TextureCache;
use crate::vulkan::base::VulkanBase;
use std::error::Error;
use vk_mem::Allocator;

/// Manages materials, ensuring that each material is only created once.
pub struct MaterialManager {
    pub materials: Vec<Material>,
    /// Owns shared texture images and accumulates their initial GPU uploads.
    texture_cache: TextureCache,
}

/// Properties for creating a new material.
pub struct MaterialProperties {
    pub name: String,
    pub vs_path: String,
    pub fs_path: String,
    pub diffuse_texture_path: Option<String>,
    pub normalmap_texture_path: Option<String>,
    pub depth_write: bool,
    // Optional per-material UV tiling factor (x, y). Defaults to [1,1] if None.
    pub uv_tiling: Option<[f32; 2]>,
}

impl MaterialManager {
    /// Creates a new `MaterialManager`.
    pub fn new() -> MaterialManager {
        MaterialManager {
            materials: Vec::new(),
            texture_cache: TextureCache::new(),
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
    ) -> Result<(), Box<dyn Error>> {
        for material in &mut self.materials {
            material.recreate_pipeline(vb)?;
        }
        Ok(())
    }

    /// Requests a material. If the material already exists, it returns the index of the existing material.
    /// Otherwise, it creates a new material and returns its index.
    /// # Arguments
    /// * `vb` - The VulkanBase struct.
    /// * `props` - The properties of the material to create.
    /// # Returns
    /// * `Result<usize, Box<dyn Error>>` - The material index, or the creation error.
    pub fn request_material(
        &mut self,
        vb: &VulkanBase,
        props: MaterialProperties,
    ) -> Result<usize, Box<dyn Error>> {
        if let Some(idx) = self.materials.iter().position(|m| m.name == props.name) {
            Ok(idx)
        } else {
            // `props` is consumed here: its owned Strings move into `Material::new`
            // without cloning. New textures are queued and submitted after scene setup.
            let mat = Material::new(
                props.name,
                vb,
                &mut self.texture_cache,
                props.vs_path,
                props.fs_path,
                props.diffuse_texture_path,
                props.normalmap_texture_path,
                props.depth_write,
                props.uv_tiling.unwrap_or([1.0, 1.0]),
            )?;
            self.materials.push(mat);
            Ok(self.materials.len() - 1)
        }
    }

    /// Queues a texture for a specialised renderer while keeping the cache as its owner.
    ///
    /// The copied Vulkan handles remain valid until [`Self::cleanup`].  This lets renderers
    /// such as the sky share the material upload batch without owning a duplicate image.
    pub fn load_shared_texture(
        &mut self,
        vb: &VulkanBase,
        image_path: &str,
    ) -> Result<(vk::ImageView, vk::Sampler), Box<dyn Error>> {
        let image_view = self.texture_cache.load(vb, image_path)?;
        Ok((image_view, self.texture_cache.sampler()))
    }

    /// Submit all texture uploads recorded while materials were created.
    /// This is called once after scene construction, before any material is drawn.
    pub fn finish_loading(&mut self, vb: &VulkanBase) -> Result<(), Box<dyn Error>> {
        self.texture_cache.flush(vb)
    }

    /// Cleans up all materials.
    /// # Arguments
    /// * `device` - The Vulkan device.
    /// * `allocator` - The global VMA allocator.
    /// * `command_pool` - Frees an unsubmitted texture-upload command buffer, if any.
    pub fn cleanup(
        &mut self,
        device: &Device,
        allocator: &Allocator,
        command_pool: vk::CommandPool,
    ) {
        for material in &mut self.materials {
            println!("🗑️ Cleaning up material {}", material.name);
            material.cleanup(device);
        }
        // The ordering is a Vulkan lifetime rule that raw handles cannot encode:
        // materials use image views, so destroy their descriptors before the cache images.
        self.materials.clear();
        self.texture_cache.cleanup(device, allocator, command_pool);
        println!("🗑️ MaterialManager cleaned up");
    }
}
