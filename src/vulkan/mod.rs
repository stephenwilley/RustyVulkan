//! --------------------------------------------------------------------------------------
//! Vulkan Module (mod.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Exposes Vulkan-related submodules such as base setup, swapchain, and ImGui renderer.
//!
//! --------------------------------------------------------------------------------------

pub mod base;
pub mod swapchain;
pub mod imgui_renderer;
pub mod render_graph;
pub mod main_pass;
pub mod ui_pass;
pub mod attachments;
pub mod shadow_pass;