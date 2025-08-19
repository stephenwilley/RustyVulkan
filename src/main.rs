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
mod vulkan;
mod graphics;

use app::app::App;

fn main() {
    if let Err(e) = App::new().run() {
        eprintln!("Application error: {}", e);
    }
}

