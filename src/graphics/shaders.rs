//! --------------------------------------------------------------------------------------
//! Shader Module (shaders.rs)
//!
//! Created: July 2025  
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module loads SPIR-V for pipeline creation. With VK_KHR_maintenance5, each
//! pipeline stage takes its SPIR-V directly through a chained
//! `VkShaderModuleCreateInfo`, so no `VkShaderModule` objects are created.
//!
//! Usage:
//!   1. Call `ShaderStageInfo::load(stage, path)` to read a SPIR-V file.
//!   2. In a pipeline builder, take each stage's `module_create_info()` and pass it to
//!      `to_create_info()`. Both must stay alive until the pipeline has been created.
//!      (`material::LoadedShaders::load` does step 1 for a vertex/fragment pair.)
//! --------------------------------------------------------------------------------------

use ash::vk;
use std::error::Error;
use std::ffi::CStr;
use std::path::Path;

/// Every shader in the project names its entry point `main`.
const ENTRY_POINT: &CStr = c"main";

/// One programmable stage (vertex or fragment) and its SPIR-V.
pub struct ShaderStageInfo {
    pub stage: vk::ShaderStageFlags,
    code: Vec<u32>,
}

impl ShaderStageInfo {
    /// Reads a SPIR-V file for one stage.
    /// # Arguments
    /// * `stage` - The pipeline stage this shader runs in.
    /// * `path` - The path to the SPIR-V file.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - The loaded stage, or an I/O error.
    pub fn load(
        stage: vk::ShaderStageFlags,
        path: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut file = std::fs::File::open(path)?;
        // A byte buffer need not be u32-aligned. Ash also validates the magic
        // number and handles byte-swapped SPIR-V, returning an I/O error for
        // malformed input instead of panicking during a slice cast.
        let code = ash::util::read_spv(&mut file)?;
        Ok(Self { stage, code })
    }

    /// Describes this stage's SPIR-V, ready to chain into its pipeline stage info.
    pub fn module_create_info(&self) -> vk::ShaderModuleCreateInfo<'_> {
        vk::ShaderModuleCreateInfo::default().code(&self.code)
    }

    /// Builds the pipeline stage info, chaining the SPIR-V from [`Self::module_create_info`].
    pub fn to_create_info<'a>(
        &self,
        module_info: &'a mut vk::ShaderModuleCreateInfo<'_>,
    ) -> vk::PipelineShaderStageCreateInfo<'a> {
        vk::PipelineShaderStageCreateInfo::default()
            .stage(self.stage)
            .name(ENTRY_POINT)
            .push_next(module_info)
    }
}
