//! --------------------------------------------------------------------------------------
//! Graphics Material (material.rs)
//!
//! Created: July 2025  
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module will hold the various bits and pieces requires to build a 'Material'
//! such as the shaders, textures, and it'll hold the pipeline layout and pipeline itself
//!
//! --------------------------------------------------------------------------------------

use crate::graphics::pipeline::Pipeline;
use crate::graphics::shaders::ShaderStageInfo;
use crate::graphics::texture::TextureCache;
use crate::vulkan::base::VulkanBase;
use ash::Device;
use ash::vk;
use std::error::Error;
use std::fmt;

/// Neutral albedo used when a material supplies only a normal map.
const DEFAULT_DIFFUSE_TEXTURE_PATH: &str = "assets/textures/default_diffuse.png";
/// Flat normal used when a material supplies only a diffuse map.
const DEFAULT_NORMAL_TEXTURE_PATH: &str = "assets/textures/default_normal.png";

/// The Material struct holds the information required to create a material
pub struct Material {
    pub name: String,
    pub pipeline: Pipeline,
    texture_set_layout: vk::DescriptorSetLayout,
    shaders: LoadedShaders,
    /// Diffuse (binding 0) and normal map (binding 1), pushed as set 1 at draw time.
    pub textures: Option<[vk::DescriptorImageInfo; 2]>,
    pub uv_tiling: [f32; 2],
}

impl Material {
    /// Recreates the materials pipeline by passing on the information to the Pipeline struct
    /// # Arguments
    /// * `vb` - The VulkanBase struct
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - The result of the pipeline recreation
    pub fn recreate_pipeline(&mut self, vb: &VulkanBase) -> Result<(), Box<dyn Error>> {
        self.pipeline.recreate(
            &vb.device,
            vb.pipeline_cache,
            vb.swapchain.extent,
            vb.swapchain.color_format,
            vb.swapchain.depth_format,
            &[&self.shaders.vertex, &self.shaders.fragment],
            &vb.engine_settings,
        )?;
        println!("🛠️ Recreated pipeline for material {}", self.name);
        Ok(())
    }

    fn create_texture_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, vk::Result> {
        let binding = |binding| vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        };
        let bindings = [binding(0), binding(1)];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(&bindings);
        unsafe { device.create_descriptor_set_layout(&layout_info, None) }
    }

    /// Create a new Material
    /// # Arguments
    /// * `name` - The name of the material.
    /// * `vb` - The VulkanBase struct.
    /// * `texture_cache` - Shares decoded textures and records their startup uploads.
    /// * `vertex_shader` - The embedded vertex shader name.
    /// * `fragment_shader` - The embedded fragment shader name.
    /// * `diffuse_texture_path` - The path to the diffuse texture.
    /// * `normalmap_texture_path` - The path to the normalmap texture.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - Returns the initialized `Material` on success, or an error on failure.
    // Material creation mirrors the independent shader, texture, and pipeline
    // inputs. The manager presents the friendlier `MaterialProperties` API.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        vb: &VulkanBase,
        texture_cache: &mut TextureCache,
        vertex_shader: String,
        fragment_shader: String,
        diffuse_texture_path: Option<String>,
        normalmap_texture_path: Option<String>,
        depth_write: bool,
        uv_tiling: [f32; 2],
    ) -> Result<Self, Box<dyn Error>> {
        // A material with either map is textured and binds both; the missing one gets a neutral default.
        let textures = if diffuse_texture_path.is_some() || normalmap_texture_path.is_some() {
            let diffuse_path =
                diffuse_texture_path.unwrap_or_else(|| DEFAULT_DIFFUSE_TEXTURE_PATH.to_string());
            let normal_path =
                normalmap_texture_path.unwrap_or_else(|| DEFAULT_NORMAL_TEXTURE_PATH.to_string());
            // The cache records copies now; MaterialManager submits the batch after scene setup.
            let diffuse_view = texture_cache.load(vb, &diffuse_path)?;
            let normal_view = texture_cache.load(vb, &normal_path)?;
            let image_info = |image_view| vk::DescriptorImageInfo {
                sampler: texture_cache.sampler(),
                image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            Some([image_info(diffuse_view), image_info(normal_view)])
        } else {
            None
        };

        let texture_set_layout = if textures.is_some() {
            Self::create_texture_set_layout(&vb.device)?
        } else {
            vk::DescriptorSetLayout::null()
        };

        // Pipeline layout must be contiguous sets starting at 0.
        // Always include the global set=0 layout first, then set=1 texture layout if used.
        let layouts = if textures.is_some() {
            vec![vb.set0_global_layout, texture_set_layout] // set 0, set 1
        } else {
            vec![vb.set0_global_layout] // set 0 only
        };

        let pipeline = Pipeline::new(&vb.device, &layouts, depth_write)?;

        let shaders = LoadedShaders::load(vertex_shader, fragment_shader)?;

        let mut material = Self {
            name,
            pipeline,
            texture_set_layout,
            shaders,
            textures,
            uv_tiling,
        };

        material.pipeline.create_graphics_pipeline(
            &vb.device,
            vb.pipeline_cache,
            vb.swapchain.extent,
            vb.swapchain.color_format,
            vb.swapchain.depth_format,
            &[&material.shaders.vertex, &material.shaders.fragment],
            &vb.engine_settings,
        )?;

        Ok(material)
    }

    /// Records this material's textures as set 1 for the draws that follow.
    pub fn push_textures(&self, vb: &VulkanBase, cmd: vk::CommandBuffer) {
        let Some(textures) = &self.textures else {
            return;
        };
        let writes = [0, 1].map(|binding| {
            vk::WriteDescriptorSet::default()
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&textures[binding as usize]))
        });
        unsafe {
            vb.push_descriptor.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.vk_layout,
                1,
                &writes,
            );
        }
    }

    /// Cleans up Vulkan resources for the material.
    /// This should be called when the material is no longer needed.
    /// # Arguments
    /// * `device` - The Vulkan device to use for cleanup.
    pub fn cleanup(&mut self, device: &Device) {
        // The texture images belong to TextureCache, which outlives every material.
        self.pipeline.cleanup(device);
        unsafe { device.destroy_descriptor_set_layout(self.texture_set_layout, None) };
    }
}

impl fmt::Display for Material {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// A struct to hold the loaded shaders
pub struct LoadedShaders {
    pub vertex: ShaderStageInfo,
    pub fragment: ShaderStageInfo,
}

impl LoadedShaders {
    /// Loads embedded SPIR-V into ShaderStageInfo structs
    /// # Arguments
    /// * `vertex_shader` - The embedded vertex shader name.
    /// * `fragment_shader` - The embedded fragment shader name.
    /// # Returns
    /// * `Result<Self, Box<dyn std::error::Error>>` - Returns the loaded shaders on success, or an error on failure.
    pub fn load(
        vertex_shader: String,
        fragment_shader: String,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(LoadedShaders {
            vertex: ShaderStageInfo::load(vk::ShaderStageFlags::VERTEX, &vertex_shader)?,
            fragment: ShaderStageInfo::load(vk::ShaderStageFlags::FRAGMENT, &fragment_shader)?,
        })
    }
}
