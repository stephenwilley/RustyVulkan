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
//!   1. Call `ShaderStageInfo::load(stage, name)` to load embedded SPIR-V.
//!   2. In a pipeline builder, take each stage's `module_create_info()` and pass it to
//!      `to_create_info()`. Both must stay alive until the pipeline has been created.
//!      (`material::LoadedShaders::load` does step 1 for a vertex/fragment pair.)
//! --------------------------------------------------------------------------------------

use ash::vk;
use std::error::Error;
use std::ffi::CStr;
use std::io::Cursor;

// Cargo generates this table from the same manifest used to compile the GLSL.
include!(concat!(env!("OUT_DIR"), "/shader_assets.rs"));

/// Every shader in the project names its entry point `main`.
const ENTRY_POINT: &CStr = c"main";

/// One programmable stage (vertex or fragment) and its SPIR-V.
pub struct ShaderStageInfo {
    pub stage: vk::ShaderStageFlags,
    code: Vec<u32>,
}

impl ShaderStageInfo {
    /// Loads one stage by its embedded shader name, for example `main.vert.spv`.
    /// # Arguments
    /// * `stage` - The pipeline stage this shader runs in.
    /// * `name` - The name in build.rs SHADERS, with `.spv` appended.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - The loaded stage, or an unknown-name or SPIR-V decoding error.
    pub fn load(stage: vk::ShaderStageFlags, name: &str) -> Result<Self, Box<dyn Error>> {
        let bytes = EMBEDDED_SHADERS
            .iter()
            .find(|(shader_name, _)| *shader_name == name)
            .map(|(_, bytes)| *bytes)
            .ok_or_else(|| format!("unknown embedded shader: {name}"))?;
        let mut file = Cursor::new(bytes);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_embedded_shader_can_be_decoded_without_a_file() {
        assert!(!EMBEDDED_SHADERS.is_empty());
        for (name, _) in EMBEDDED_SHADERS {
            let shader = ShaderStageInfo::load(vk::ShaderStageFlags::VERTEX, name).unwrap();
            assert_eq!(shader.code[0], 0x0723_0203, "SPIR-V header for {name}");
            assert!(shader.code.len() >= 5);
        }
    }
}
