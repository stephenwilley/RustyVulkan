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

use ash::Device;
use ash::vk;
use crate::vulkan::base::VulkanBase;
use crate::graphics::pipeline::Pipeline;
use crate::graphics::texture::Texture;
use crate::graphics::shaders::{ShaderModule,ShaderStageInfo};
use std::error::Error;
use std::fmt;
use vk_mem::Allocator;

/// The Material struct holds the information required to create a material
pub struct Material {
    pub name: String,
    pub pipeline: Pipeline,
    pub texture_descriptor_set_layout: vk::DescriptorSetLayout,
    pub texture_descriptor_pool: vk::DescriptorPool,
    pub texture_descriptor_set: vk::DescriptorSet,
    shaders: LoadedShaders,
    pub textures: Option<LoadedTextures>,
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
            vb.swapchain.extent,
            vb.swapchain.color_format,
            vb.swapchain.depth_format,
            &[ &self.shaders.vertex,
               &self.shaders.fragment ],
            &vb.engine_settings,
        )?;
        println!("🛠️ Recreated pipeline for material {}", self.name);
        Ok(())
    }

    fn setup_texture_descriptors(
        vb: &VulkanBase
    ) -> Result<(
        vk::DescriptorSetLayout,
        vk::DescriptorSet,
        vk::DescriptorPool
    ), Box<dyn Error>> {
        let layout_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            }
        ];
        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: 2,
            p_bindings: layout_bindings.as_ptr(),
            ..Default::default()
        };
        let texture_descriptor_set_layout = unsafe {
            vb.device.create_descriptor_set_layout(&layout_info, None)?
        };

        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 2,
        };
        let pool_info = vk::DescriptorPoolCreateInfo {
            flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
            max_sets: 1,
            ..Default::default()
        };
        let texture_descriptor_pool = unsafe {
            vb.device.create_descriptor_pool(&pool_info, None)?
        };

        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: texture_descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &texture_descriptor_set_layout,
            ..Default::default()
        };
        let texture_descriptor_set = unsafe {
            vb.device.allocate_descriptor_sets(&alloc_info)?[0]
        };
        Ok((texture_descriptor_set_layout, texture_descriptor_set, texture_descriptor_pool))
    }

    /// Create a new Material
    /// # Arguments
    /// * `name` - The name of the material.
    /// * `vb` - The VulkanBase struct.
    /// * `vs_path` - The path to the vertex shader.
    /// * `fs_path` - The path to the fragment shader.
    /// * `diffuse_texture_path` - The path to the diffuse texture.
    /// * `normalmap_texture_path` - The path to the normalmap texture.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - Returns the initialized `Material` on success, or an error on failure.
    pub fn new(
        name: String,
        vb: &VulkanBase,
        vs_path: String,
        fs_path: String,
        diffuse_texture_path: Option<String>,
        normalmap_texture_path: Option<String>,
        depth_write: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let texturing_enabled = diffuse_texture_path.is_some();

        // If texturing is enabled but no normal map was provided, use a default normal map
        let normalmap_texture_path = if texturing_enabled && normalmap_texture_path.is_none() {
            Some("assets/textures/default_normal.png".to_string())
        } else {
            normalmap_texture_path
        };

        // Only create all the texture descriptor stuff if texture paths are there
        let (texture_descriptor_set_layout,
            texture_descriptor_set,
            texture_descriptor_pool
            ) = if texturing_enabled {
                Self::setup_texture_descriptors(vb)?
            } else {
                (vk::DescriptorSetLayout::null(), vk::DescriptorSet::null(), vk::DescriptorPool::null())
            };

        // Pipeline layout must be contiguous sets starting at 0.
        // Always include the global set=0 layout first, then set=1 texture layout if used.
        let layouts = if texturing_enabled {
            vec![vb.set0_global_layout, texture_descriptor_set_layout] // set 0, set 1
        } else {
            vec![vb.set0_global_layout] // set 0 only
        };

        let pipeline = Pipeline::new(
            &vb.device,
            &layouts,
            depth_write,
        )?;
        
        let shaders = LoadedShaders::load(
            &vb.device,
            vs_path,
            fs_path
        )?;

        let mut textures = None;
        if texturing_enabled {
            textures = Some(LoadedTextures::load(
                &vb.device,
                vb.allocator.as_ref().unwrap(),
                vb.command_pool,
                vb.graphics_queue,
                diffuse_texture_path.unwrap(),
                normalmap_texture_path.unwrap()
            ).expect("Failed to load textures"));
        }

        let mut material = Self {
            name,
            pipeline,
            texture_descriptor_set_layout,
            texture_descriptor_pool,
            texture_descriptor_set,
            shaders,
            textures,
        };

        if texturing_enabled {
            let tex_diffuse = &material.textures.as_ref().unwrap().diffuse;
            // Prepare image info pointing to our texture’s sampler and view
            let diffuse_info = vk::DescriptorImageInfo {
                sampler:     tex_diffuse.sampler,
                image_view:  tex_diffuse.image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            // Write it into binding 0 of set 1
            let write_diffuse = vk::WriteDescriptorSet {
                dst_set:           texture_descriptor_set,
                dst_binding:       0,
                dst_array_element: 0,
                descriptor_count:  1,
                descriptor_type:   vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info:      &diffuse_info,
                ..Default::default()
            };
            let tex_normal = &material.textures.as_ref().unwrap().normalmap;
            // Prepare image info pointing to our texture’s sampler and view
            let normal_info = vk::DescriptorImageInfo {
                sampler:     tex_normal.sampler,
                image_view:  tex_normal.image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            // Write it into binding 1 of set 1
            let write_normal = vk::WriteDescriptorSet {
                dst_set:           texture_descriptor_set,
                dst_binding:       1,
                dst_array_element: 0,
                descriptor_count:  1,
                descriptor_type:   vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info:      &normal_info,
                ..Default::default()
            };
            unsafe {
                vb.device.update_descriptor_sets(&[write_diffuse], &[]);
                vb.device.update_descriptor_sets(&[write_normal], &[]);
            }
        }

        material.pipeline.create_graphics_pipeline(
            &vb.device,
            vb.swapchain.extent,
            vb.swapchain.color_format,
            vb.swapchain.depth_format,
            &[ &material.shaders.vertex,
               &material.shaders.fragment ],
            &vb.engine_settings
        )?;

        Ok(material)
    }

    /// Cleans up Vulkan resources for the material.
    /// This should be called when the material is no longer needed.
    /// # Arguments
    /// * `device` - The Vulkan device to use for cleanup.
    /// * `allocator` - The global VMA allocator.
    pub fn cleanup(&mut self, device: &Device, allocator: &Allocator) {
        self.shaders.cleanup();
        if let Some(textures) = self.textures.as_mut() {
            textures.cleanup(device, allocator);
            unsafe {
                device.free_descriptor_sets(
                    self.texture_descriptor_pool,
                    &[self.texture_descriptor_set],
                ).expect("Failed to free descriptor set");
                device.destroy_descriptor_set_layout(self.texture_descriptor_set_layout, None);
                device.destroy_descriptor_pool(self.texture_descriptor_pool, None);
            }
        }
        self.pipeline.cleanup(device);
    }
}

impl fmt::Display for Material {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// A struct to hold the loaded shaders
pub struct LoadedShaders {
    pub vertex:   ShaderStageInfo,
    pub fragment: ShaderStageInfo,
}

impl LoadedShaders {
    /// Loads SPIR-V files into ShaderStageInfo structs
    /// # Arguments
    /// * `device` - The Vulkan device to use for creating the shader modules.
    /// * `vs_path` - The path to the vertex shader.
    /// * `fs_path` - The path to the fragment shader.
    /// # Returns
    /// * `Result<Self, Box<dyn std::error::Error>>` - Returns the loaded shaders on success, or an error on failure.
    pub fn load(
        device: &Device,
        vs_path: String,
        fs_path: String,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let vs = ShaderModule::from_spv_file(device, vs_path)?;
        let fs = ShaderModule::from_spv_file(device, fs_path)?;
        let entry = c"main";

        Ok(LoadedShaders {
            vertex:   ShaderStageInfo { stage: vk::ShaderStageFlags::VERTEX,   shader_module: vs, entry_name: entry },
            fragment: ShaderStageInfo { stage: vk::ShaderStageFlags::FRAGMENT, shader_module: fs, entry_name: entry },
        })
    }

    /// Frees the Vulkan shader modules.
    pub fn cleanup(&self) {
        self.vertex.shader_module.cleanup();
        self.fragment.shader_module.cleanup();
    }
}

/// A struct to hold the loaded textures
pub struct LoadedTextures {
    pub diffuse: Texture,
    pub normalmap: Texture,
}

impl LoadedTextures {
    /// Loads the diffuse PNG into a GPU-backed Texture.
    /// # Arguments
    /// * `device` - The Vulkan device.
    /// * `allocator` - The global VMA allocator.
    /// * `command_pool` - The command pool to use for creating the texture.
    /// * `queue` - The queue to use for submitting the texture creation commands.
    /// * `diffuse_texture_path` - The path to the diffuse texture.
    /// * `normalmap_texture_path` - The path to the normalmap texture.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - Returns the loaded textures on success, or an error on failure.
    pub fn load(
        device: &Device,
        allocator: &Allocator,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        diffuse_texture_path: String,
        normalmap_texture_path: String,
    ) -> Result<Self, Box<dyn Error>> {
        let diffuse = Texture::new(
            device,
            allocator,
            command_pool,
            queue,
            diffuse_texture_path.as_str(),
        )?;
        let normalmap = Texture::new(
            device,
            allocator,
            command_pool,
            queue,
            normalmap_texture_path.as_str(),
        )?;
        Ok(LoadedTextures { diffuse, normalmap })
    }

    /// Cleans up Vulkan resources for the texture.
    /// # Arguments
    /// * `device` - The Vulkan device to use for cleanup.
    /// * `allocator` - The global VMA allocator.
    pub fn cleanup(&mut self, device: &Device, allocator: &Allocator) {
        self.diffuse.cleanup(device, allocator);
        self.normalmap.cleanup(device, allocator);
    }
}