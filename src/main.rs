//! --------------------------------------------------------------------------------------
//! Application Entry Point (main.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Initializes subsystems and starts the main application loop.
//!
//! --------------------------------------------------------------------------------------

mod app;
mod graphics;
mod vulkan;

use app::app::App;
use std::path::PathBuf;

/// Finder starts an application with no useful working directory. When the
/// executable lives in a bundle, point relative asset paths at its Resources
/// directory and tell the bundled Vulkan loader where MoltenVK lives.
fn configure_app_bundle() -> std::io::Result<()> {
    let executable = std::env::current_exe()?;
    let Some(contents) = executable
        .parent()
        .and_then(|macos| macos.parent())
        .filter(|path| path.file_name().is_some_and(|name| name == "Contents"))
    else {
        return Ok(()); // Normal `cargo run`: keep using the repository root.
    };

    let resources = contents.join("Resources");
    std::env::set_current_dir(&resources)?;

    let driver_manifest: PathBuf = resources.join("vulkan/icd.d/MoltenVK_icd.json");
    if driver_manifest.is_file() {
        // This runs before winit or Vulkan starts any threads, so no other code
        // can concurrently read the process environment while it is changed.
        unsafe {
            std::env::set_var("VK_DRIVER_FILES", driver_manifest);
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = configure_app_bundle() {
        eprintln!("Could not configure application resources: {error}");
        return;
    }
    if let Err(e) = App::new().run() {
        eprintln!("Application error: {}", e);
    }
}
