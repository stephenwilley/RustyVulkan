//! --------------------------------------------------------------------------------------
//! Shader Module (shaders.rs)
//!
//! Created: July 2025  
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module provides abstractions for loading and managing Vulkan shader modules.
//! It defines:
//!   • `ShaderModule` - a wrapper around a VkShaderModule handle that automatically
//!     destroys the module when dropped.
//!   • `ShaderStageInfo` - pairs a shader stage flag (vertex/fragment) with its module
//!     and entry-point C string.
//!   • `load_default_stages` - convenience function to load both the vertex and
//!     fragment shaders in one call.
//!
//! Usage: 
//!   1. Call `ShaderModule::from_spv_file(device, path)` to read a SPIR-V file and create  
//!      a module.  
//!   2. Wrap it in `ShaderStageInfo { stage, shader_module, entry_name }`.  
//!   3. Use `ShaderStageInfo::to_create_info()` in your pipeline builder.  
//!   4. Or simply call `load_default_stages(device)` to get both stages at once.
//! --------------------------------------------------------------------------------------

use ash::vk;
use ash::Device;
use std::error::Error;
use std::ffi::CStr;
use std::path::Path;

/// Represents a Vulkan shader module, encapsulating the shader code and its creation.
/// It is used to create shader modules for the graphics pipeline.
/// This struct is used by the `Pipeline` to set up the rendering pipeline.
/// It provides methods to create shader modules from SPIR-V files and clean up resources.
pub struct ShaderModule {
    device: Device,
    pub vk_shader_module: vk::ShaderModule,
}

impl ShaderModule {
    /// Loads a SPIR-V blob from disk and creates a Vulkan shader module.
    /// # Arguments
    /// * `device` - The Vulkan logical device to use for creating the shader module.
    /// * `path` - The path to the SPIR-V file.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - Returns the created `ShaderModule` on
    ///   success, or an error on failure.
    pub fn from_spv_file(device: &Device, path: impl AsRef<Path>) 
        -> Result<Self, Box<dyn Error>>
    {
        let bytes = std::fs::read(path)?;
        // SPIR-V words are u32, not u8, so we cast here:
        let code = bytemuck::cast_slice::<u8, u32>(&bytes);
        let create_info = vk::ShaderModuleCreateInfo::default()
            .code(code);

        let vk_shader_module = unsafe { device.create_shader_module(&create_info, None)? };
        Ok(Self { 
            device: device.clone(),
            vk_shader_module
        })
    }

    /// Clean up the shader module by destroying it
    /// # Arguments
    /// * `self` - The shader module to clean up.
    pub fn cleanup(&self) {
        unsafe { self.device.destroy_shader_module(self.vk_shader_module, None) };
    }
}

/// Describes one programmable stage (vertex or fragment).
pub struct ShaderStageInfo {
    pub stage: vk::ShaderStageFlags,
    pub shader_module: ShaderModule,
    pub entry_name: &'static CStr,
}

impl ShaderStageInfo {
    /// Converts this shader stage info into a Vulkan pipeline shader stage create info.
    /// # Returns
    /// * `vk::PipelineShaderStageCreateInfo` - The Vulkan structure ready to be
    ///   used in pipeline creation.
    pub fn to_create_info(&self) -> vk::PipelineShaderStageCreateInfo<'_> {
        vk::PipelineShaderStageCreateInfo {
            stage: self.stage,
            module: self.shader_module.vk_shader_module,
            p_name: self.entry_name.as_ptr(),
            ..Default::default()
        }
    }
}