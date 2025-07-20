//! --------------------------------------------------------------------------------------
//! Project Assets (assets.rs)
//!
//! Created: July 2025  
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module manages project assets, including loading and storing meshes, vertex data,
//! shaders, textures, and other resources needed for rendering.
//! The idea is to keep this logic separate from the Vulkan-specific code
//!
//! --------------------------------------------------------------------------------------

use ash::vk;
use ash::Device;
use std::error::Error;

use crate::graphics::mesh::{Vertex, VertexBuffer, IndexBuffer};
use crate::graphics::shaders::{ShaderModule, ShaderStageInfo};

/// This module defines the `ProjectAssets` struct which will hold the various assets
/// used in the project, such as meshes, shaders, and textures.
pub struct ProjectAssets {
    cube_vertices: [Vertex; 24],
    cube_indices: [u16; 36],
    mesh_vs_path: String,
    mesh_fs_path: String,
}

impl ProjectAssets {
    pub fn new() -> Self {
        Self {
            cube_vertices: [
                // +Z face (blue), normal [0,0,1]
                Vertex{ pos:[-1.,-1., 1.], normal:[0.,0.,1.], color:[0.,0.,1.] },
                Vertex{ pos:[ 1.,-1., 1.], normal:[0.,0.,1.], color:[0.,0.,1.] },
                Vertex{ pos:[ 1., 1., 1.], normal:[0.,0.,1.], color:[0.,0.,1.] },
                Vertex{ pos:[-1., 1., 1.], normal:[0.,0.,1.], color:[0.,0.,1.] },

                // -Z face (green), normal [0,0,-1]
                Vertex{ pos:[ 1.,-1.,-1.], normal:[0.,0.,-1.], color:[0.,1.,0.] },
                Vertex{ pos:[-1.,-1.,-1.], normal:[0.,0.,-1.], color:[0.,1.,0.] },
                Vertex{ pos:[-1., 1.,-1.], normal:[0.,0.,-1.], color:[0.,1.,0.] },
                Vertex{ pos:[ 1., 1.,-1.], normal:[0.,0.,-1.], color:[0.,1.,0.] },

                // +Y face (red), normal [0,1,0]
                Vertex{ pos:[-1., 1., 1.], normal:[0.,1.,0.], color:[1.,0.,0.] },
                Vertex{ pos:[ 1., 1., 1.], normal:[0.,1.,0.], color:[1.,0.,0.] },
                Vertex{ pos:[ 1., 1.,-1.], normal:[0.,1.,0.], color:[1.,0.,0.] },
                Vertex{ pos:[-1., 1.,-1.], normal:[0.,1.,0.], color:[1.,0.,0.] },

                // -Y face (yellow), normal [0,-1,0]
                Vertex{ pos:[-1.,-1.,-1.], normal:[0.,-1.,0.], color:[1.,1.,0.] },
                Vertex{ pos:[ 1.,-1.,-1.], normal:[0.,-1.,0.], color:[1.,1.,0.] },
                Vertex{ pos:[ 1.,-1., 1.], normal:[0.,-1.,0.], color:[1.,1.,0.] },
                Vertex{ pos:[-1.,-1., 1.], normal:[0.,-1.,0.], color:[1.,1.,0.] },

                // +X face (magenta), normal [1,0,0]
                Vertex{ pos:[ 1.,-1., 1.], normal:[1.,0.,0.], color:[1.,0.,1.] },
                Vertex{ pos:[ 1.,-1.,-1.], normal:[1.,0.,0.], color:[1.,0.,1.] },
                Vertex{ pos:[ 1., 1.,-1.], normal:[1.,0.,0.], color:[1.,0.,1.] },
                Vertex{ pos:[ 1., 1., 1.], normal:[1.,0.,0.], color:[1.,0.,1.] },

                // -X face (cyan), normal [-1,0,0]
                Vertex{ pos:[-1.,-1.,-1.], normal:[-1.,0.,0.], color:[0.,1.,1.] },
                Vertex{ pos:[-1.,-1., 1.], normal:[-1.,0.,0.], color:[0.,1.,1.] },
                Vertex{ pos:[-1., 1., 1.], normal:[-1.,0.,0.], color:[0.,1.,1.] },
                Vertex{ pos:[-1., 1.,-1.], normal:[-1.,0.,0.], color:[0.,1.,1.] },
            ],
            cube_indices: [
                 0, 1, 2,  2, 3, 0,     // +Z
                 4, 5, 6,  6, 7, 4,     // -Z
                 8, 9,10, 10,11, 8,     // +Y
                12,13,14, 14,15,12,     // -Y
                16,17,18, 18,19,16,     // +X
                20,21,22, 22,23,20,     // -X
            ],
            mesh_vs_path: "assets/shaders/spv/passthrough.vert.spv".into(),
            mesh_fs_path: "assets/shaders/spv/lambert.frag.spv".into(),
        }
    }
}

/// Holds GPU‐ready mesh buffers
pub struct LoadedMeshes {
    pub v_buffer: VertexBuffer,
    pub i_buffer: IndexBuffer,
}

impl LoadedMeshes {
    /// Uploads the vertices and indices into a GPU buffer.
    pub fn load(
        instance: &ash::Instance,
        device:   &Device,
        phys:     vk::PhysicalDevice,
        assets:   &ProjectAssets,
    ) -> Result<Self, Box<dyn Error>> {
        // Grab the raw vertex array from the pure data:
        let verts = &assets.cube_vertices;
        let indices = &assets.cube_indices;

        // Use the mesh helper to allocate & fill a VertexBuffer
        let vb = VertexBuffer::new(
            instance,
            device,
            phys,
            verts,
        )?;
         let ib = IndexBuffer::new(
            instance,
            device,
            phys,
            indices,
        )?;

        Ok(LoadedMeshes { v_buffer: vb, i_buffer: ib })
    }

    /// Record only the indexed draw commands into the given secondary CB
    pub fn record(&self, device: &Device, cmd_buf: vk::CommandBuffer) {
        // Bind & draw
        unsafe {
            device.cmd_bind_vertex_buffers(cmd_buf, 0, &[self.v_buffer.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd_buf, self.i_buffer.buffer, 0, vk::IndexType::UINT16);
            device.cmd_draw_indexed(cmd_buf, self.i_buffer.count, 1, 0, 0, 0);
        }
    }

    /// Frees the GPU buffer and its memory.
    pub fn cleanup(&self, device: &Device) {
        self.v_buffer.cleanup(device);
        self.i_buffer.cleanup(device);
    }
}

/// Holds the actual GPU‐ready shader stages
pub struct LoadedShaders {
    pub vertex:   ShaderStageInfo,
    pub fragment: ShaderStageInfo,
}

impl LoadedShaders {
    /// Loads SPIR-V files into ShaderStageInfo structs
    pub fn load(
        device: &ash::Device,
        paths:  &ProjectAssets,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let vs = ShaderModule::from_spv_file(device, &paths.mesh_vs_path)?;
        let fs = ShaderModule::from_spv_file(device, &paths.mesh_fs_path)?;
        let entry = std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap();

        Ok(LoadedShaders {
            vertex:   ShaderStageInfo { stage: vk::ShaderStageFlags::VERTEX,   shader_module: vs, entry_name: entry },
            fragment: ShaderStageInfo { stage: vk::ShaderStageFlags::FRAGMENT, shader_module: fs, entry_name: entry },
        })
    }

    /// Frees the Vulkan shader modules.
    pub fn cleanup(&self) {
        self.vertex.shader_module.cleanup();
        self.fragment.shader_module.cleanup();
    }
}