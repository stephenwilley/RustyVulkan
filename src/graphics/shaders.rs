//! --------------------------------------------------------------------------------------
//! Shader Module (shaders.rs)
//!
//! Created: July 2025  
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module provides abstractions for loading and managing Vulkan shader modules.
//! It defines:
//!   • `ShaderModule` - an owning wrapper around a `VkShaderModule` handle that
//!     destroys the handle automatically when the Rust value is dropped.
//!   • `ShaderStageInfo` - pairs a shader stage flag (vertex/fragment) with its module
//!     and entry-point C string.
//!
//! Usage:
//!   1. Call `ShaderModule::from_spv_file(device, path)` to read a SPIR-V file and create
//!      a module.
//!   2. Wrap it in `ShaderStageInfo { stage, shader_module, entry_name }`.
//!   3. Use `ShaderStageInfo::to_create_info()` in your pipeline builder.
//!      (`material::LoadedShaders::load` does steps 1-2 for a vertex/fragment pair.)
//! --------------------------------------------------------------------------------------

use ash::Device;
use ash::vk;
use std::error::Error;
use std::ffi::CStr;
use std::path::Path;

/// Owns one Vulkan shader-module handle and destroys it in [`Drop`].
///
/// `ash::Device::clone()` copies Ash's lightweight Rust wrapper; it does not
/// create a second Vulkan logical device.  Keeping that wrapper here avoids a
/// borrowed `&Device` field, which would make this type lifetime-parameterised.
/// The application still explicitly chooses teardown order because Vulkan raw
/// handles cannot express “the device must outlive this module” to Rust. That
/// order guarantees this type's `Drop` runs while its device is still valid.
pub struct ShaderModule {
    device: Device,
    vk_shader_module: vk::ShaderModule,
}

impl ShaderModule {
    /// Loads a SPIR-V blob from disk and creates a Vulkan shader module.
    /// # Arguments
    /// * `device` - The Vulkan logical device to use for creating the shader module.
    /// * `path` - The path to the SPIR-V file.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - Returns the created `ShaderModule` on
    ///   success, or an error on failure.
    pub fn from_spv_file(device: &Device, path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let mut file = std::fs::File::open(path)?;
        // A byte buffer need not be u32-aligned. Ash also validates the magic
        // number and handles byte-swapped SPIR-V, returning an I/O error for
        // malformed input instead of panicking during a slice cast.
        let code = ash::util::read_spv(&mut file)?;
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code);

        let vk_shader_module = unsafe { device.create_shader_module(&create_info, None)? };
        Ok(Self {
            device: device.clone(),
            vk_shader_module,
        })
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        // This value uniquely owns the Vulkan handle, so its destructor runs
        // exactly once when a scope ends or an owning field is replaced.
        unsafe {
            self.device
                .destroy_shader_module(self.vk_shader_module, None)
        };
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
