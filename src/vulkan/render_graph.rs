//! --------------------------------------------------------------------------------------
//! Render Graph Code (render_graph.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Used for constructing the path through the render engine
//!
//! --------------------------------------------------------------------------------------

use std::error::Error;

use crate::app::render::draw_frame;
use crate::app::app::App;

/// Render pass enum
/// Will eventually contain things like
/// Shadow, UI
pub enum RenderPass {
    Main,
}

/// RenderGraph
/// Contains a list of render passes
pub struct RenderGraph {
    render_passes: Vec<RenderPass>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            render_passes: Vec::new(),
        }
    }

    pub fn execute(&self, app: &mut App) -> Result<(), Box<dyn Error>> {
        draw_frame(app)?;
        Ok(())
    }

    pub fn add(&mut self, render_pass: RenderPass) {
        self.render_passes.push(render_pass);
    }
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}