//! --------------------------------------------------------------------------------------
//! App Module (mod.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Re-exports and submodule declarations for the `app` package.
//!
//! --------------------------------------------------------------------------------------

// Keeping the main application in `app::app` makes its relationship to the
// smaller input and scene modules explicit at call sites.
#[allow(clippy::module_inception)]
pub mod app;
pub mod input;
pub mod scene;
