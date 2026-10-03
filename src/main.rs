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
use std::path::Path;
#[cfg(target_os = "macos")]
use std::path::PathBuf;

/// Finder starts an application with no useful working directory. When the
/// executable lives in a bundle, point relative asset paths at its Resources
/// directory. On macOS, also select KosmicKrisp before Vulkan is initialized.
fn configure_runtime() -> std::io::Result<()> {
    let executable = std::env::current_exe()?;
    let bundle_resources = executable
        .parent()
        .and_then(|macos| macos.parent())
        .filter(|path| path.file_name().is_some_and(|name| name == "Contents"))
        .map(|contents| contents.join("Resources"));

    if let Some(resources) = &bundle_resources {
        std::env::set_current_dir(resources)?;
    }

    configure_kosmickrisp(bundle_resources.as_deref())?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn configure_kosmickrisp(bundle_resources: Option<&Path>) -> std::io::Result<()> {
    if std::env::var_os("VK_DRIVER_FILES").is_some() {
        return Ok(());
    }

    let mut candidates = Vec::new();
    if let Some(resources) = bundle_resources {
        candidates.push(resources.join("vulkan/icd.d/libkosmickrisp_icd.json"));
    } else {
        if let Some(sdk) = std::env::var_os("VULKAN_SDK") {
            candidates.push(PathBuf::from(sdk).join("share/vulkan/icd.d/libkosmickrisp_icd.json"));
        }
        candidates.push(PathBuf::from(
            "/usr/local/share/vulkan/icd.d/libkosmickrisp_icd.json",
        ));
    }

    let driver_manifest = candidates
        .into_iter()
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "KosmicKrisp ICD not found. Install the macOS Vulkan SDK with System Global Installation enabled, or set VK_DRIVER_FILES explicitly.",
            )
        })?;

    // This runs before winit or Vulkan starts any threads, so no other code can
    // concurrently read the process environment while it is changed.
    unsafe {
        std::env::set_var("VK_DRIVER_FILES", &driver_manifest);
    }
    println!(
        "🌌 Using KosmicKrisp Vulkan driver: {}",
        driver_manifest.display()
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn configure_kosmickrisp(_bundle_resources: Option<&Path>) -> std::io::Result<()> {
    Ok(())
}

fn main() -> std::process::ExitCode {
    if let Err(error) = configure_runtime() {
        eprintln!("Could not configure application runtime: {error}");
        return std::process::ExitCode::FAILURE;
    }
    if let Err(e) = App::new().run() {
        eprintln!("Application error: {}", e);
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}
