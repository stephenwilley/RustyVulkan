//! --------------------------------------------------------------------------------------
//! Graphics Mesh Abstractions (mesh.rs)
//!
//! Created: July 2025  
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module defines simple `Vertex`, `VertexBuffer` and `IndexBuffer` types
//! for uploading data to the GPU.
//!
//! --------------------------------------------------------------------------------------

use ash::vk;
use ash::{Instance, Device};
use std::error::Error;
use bytemuck::{Pod, Zeroable, offset_of};


/// A single vertex: 3D position + normal, color, uv, tangent, bitangent.
#[repr(C)]
#[derive(Default, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos:   [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub uv:    [f32; 2],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
}

impl Vertex {
    /// Returns the binding description for this vertex format.
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    /// Returns the attribute descriptions (position @ location 0, color @ location 1).
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 6] {
        [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, normal) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, color) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 3,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Vertex, uv) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 4,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, tangent) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 5,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, bitangent) as u32,
            },
        ]
    }
}

/// A GPU-resident vertex buffer holding one mesh’s vertices.
pub struct VertexBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
}

impl VertexBuffer {
    /// Create a new vertex buffer, upload `data` (slice of Vertex)
    /// using HOST_VISIBLE | HOST_COHERENT memory properties.
    pub fn new(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        data: &[Vertex],
    ) -> Result<Self, Box<dyn Error>> {
        let size = std::mem::size_of_val(data) as vk::DeviceSize;

        // 1) Create the buffer
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        // 2) Allocate memory
        let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_type_index = find_memory_type(
            instance,
            physical_device,
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_req.size,
            memory_type_index: mem_type_index,
            ..Default::default()
        };
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };

        // 3) Bind and copy data
        unsafe {
            device.bind_buffer_memory(buffer, memory, 0)?;
            let ptr = device.map_memory(memory, 0, size, Default::default())?;
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const _,
                ptr.cast(),
                data.len(),
            );
            device.unmap_memory(memory);
        }

        Ok(VertexBuffer { buffer, memory })
    }

    /// Frees the Vulkan buffer and its backing memory.
    pub fn cleanup(&self, device: &Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// A GPU-resident index buffer for indexed drawing.
pub struct IndexBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub count:  u32,
}

impl IndexBuffer {
    /// Create a new index buffer, uploading `indices` (slice of u32).
    /// # Arguments
    /// * `device` - The Vulkan device to use for creating the buffer.
    /// * `physical_device` - The physical device to query memory properties from.
    /// * `indices` - The indices to upload to the buffer.
    /// # Returns
    /// * `Result<Self, vk::Result>` - Returns the initialized `IndexBuffer`
    pub fn new(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        data: &[u32],
    ) -> Result<Self, vk::Result> {
        let size = std::mem::size_of_val(data) as vk::DeviceSize;

        // 1) create buffer with usage INDEX_BUFFER
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::INDEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        // 2) Allocate memory
        let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_type_index = find_memory_type(
            instance,
            physical_device,
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_req.size,
            memory_type_index: mem_type_index,
            ..Default::default()
        };
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };

        // 3) Bind and copy data
        unsafe {
            device.bind_buffer_memory(buffer, memory, 0)?;
            let ptr = device.map_memory(memory, 0, size, Default::default())?;
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const _,
                ptr.cast(),
                data.len(),
            );
            device.unmap_memory(memory);
        }

        Ok(IndexBuffer { buffer, memory, count: data.len() as u32 })
    }

    pub fn cleanup(&self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Helper to find a memory type index on the GPU.
/// 
/// # Arguments
/// * `instance`     – your Vulkan `Instance` (so you can query properties)
/// * `phys_device`  – the `PhysicalDevice` you picked earlier
/// * `type_filter`  – bitmask from `vkGetBufferMemoryRequirements(...).memoryTypeBits`
/// * `properties`   – desired flags, e.g. HOST_VISIBLE | HOST_COHERENT
pub fn find_memory_type(
    instance: &ash::Instance,
    phys_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    // Query all memory types & heaps on this GPU:
    let mem_props = unsafe {
        instance.get_physical_device_memory_properties(phys_device)
    };

    // Scan through each memory type index:
    for (i, mem_type) in mem_props.memory_types.iter().enumerate() {
        let bit = 1 << i;
        // 1) Is this type allowed by the bitmask?
        // 2) Does it include *all* the flags we asked for?
        if (type_filter & bit) != 0 
            && mem_type.property_flags.contains(properties)
        {
            return i as u32;
        }
    }

    panic!("Failed to find suitable memory type!");
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn unit_plane() -> Self {
        Self {
            vertices: vec![
                Vertex {
                    pos:       [-1.0,  0.0,  1.0],
                    normal:    [ 0.0,  1.0,  0.0],
                    color:     [ 1.0,  1.0,  1.0],
                    uv:        [ 0.0,  0.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  0.0, -1.0],
                },
                Vertex {
                    pos:       [ 1.0,  0.0,  1.0],
                    normal:    [ 0.0,  1.0,  0.0],
                    color:     [ 0.0,  0.0,  1.0],
                    uv:        [ 1.0,  0.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  0.0, -1.0],
                },
                Vertex {
                    pos:       [ 1.0,  0.0, -1.0],
                    normal:    [ 0.0,  1.0,  0.0],
                    color:     [ 0.0,  1.0,  0.0],
                    uv:        [ 1.0,  1.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  0.0, -1.0],
                },
                Vertex {
                    pos:       [-1.0,  0.0, -1.0],
                    normal:    [ 0.0,  1.0,  0.0],
                    color:     [ 1.0,  0.0,  0.0],
                    uv:        [ 0.0,  1.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  0.0, -1.0],
                },
            ],
            indices: vec![
                0, 1, 2,  2, 3, 0,
            ],
        }
    }

    pub fn cube() -> Self {
        Self {
            vertices: vec![
                // +Z face (blue), normal [0,0,1]
                Vertex {
                    pos:       [-1.0, -1.0,  1.0],
                    normal:    [ 0.0,  0.0,  1.0],
                    color:     [ 1.0,  1.0,  1.0],
                    uv:        [ 0.0,  0.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                Vertex {
                    pos:       [ 1.0, -1.0,  1.0],
                    normal:    [ 0.0,  0.0,  1.0],
                    color:     [ 0.0,  0.0,  1.0],
                    uv:        [ 1.0,  0.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                Vertex {
                    pos:       [ 1.0,  1.0,  1.0],
                    normal:    [ 0.0,  0.0,  1.0],
                    color:     [ 0.0,  1.0,  0.0],
                    uv:        [ 1.0,  1.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                Vertex {
                    pos:       [-1.0,  1.0,  1.0],
                    normal:    [ 0.0,  0.0,  1.0],
                    color:     [ 1.0,  0.0,  0.0],
                    uv:        [ 0.0,  1.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                // -Z face (green), normal [0,0,-1]
                Vertex {
                    pos:       [ 1.0, -1.0, -1.0],
                    normal:    [ 0.0,  0.0, -1.0],
                    color:     [ 1.0,  1.0,  1.0],
                    uv:        [ 0.0,  0.0],
                    tangent:   [ -1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                Vertex {
                    pos:       [-1.0, -1.0, -1.0],
                    normal:    [ 0.0,  0.0, -1.0],
                    color:     [ 0.0,  0.0,  1.0],
                    uv:        [ 1.0,  0.0],
                    tangent:   [ -1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                Vertex {
                    pos:       [-1.0,  1.0, -1.0],
                    normal:    [ 0.0,  0.0, -1.0],
                    color:     [ 0.0,  1.0,  0.0],
                    uv:        [ 1.0,  1.0],
                    tangent:   [ -1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                Vertex {
                    pos:       [ 1.0,  1.0, -1.0],
                    normal:    [ 0.0,  0.0, -1.0],
                    color:     [ 1.0,  0.0,  0.0],
                    uv:        [ 0.0,  1.0],
                    tangent:   [ -1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                // +Y face (red), normal [0,1,0]
                Vertex {
                    pos:       [-1.0,  1.0,  1.0],
                    normal:    [ 0.0,  1.0,  0.0],
                    color:     [ 1.0,  1.0,  1.0],
                    uv:        [ 0.0,  0.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  0.0, -1.0],
                },
                Vertex {
                    pos:       [ 1.0,  1.0,  1.0],
                    normal:    [ 0.0,  1.0,  0.0],
                    color:     [ 0.0,  0.0,  1.0],
                    uv:        [ 1.0,  0.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  0.0, -1.0],
                },
                Vertex {
                    pos:       [ 1.0,  1.0, -1.0],
                    normal:    [ 0.0,  1.0,  0.0],
                    color:     [ 0.0,  1.0,  0.0],
                    uv:        [ 1.0,  1.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  0.0, -1.0],
                },
                Vertex {
                    pos:       [-1.0,  1.0, -1.0],
                    normal:    [ 0.0,  1.0,  0.0],
                    color:     [ 1.0,  0.0,  0.0],
                    uv:        [ 0.0,  1.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  0.0, -1.0],
                },
                // -Y face (yellow), normal [0,-1,0]
                Vertex {
                    pos:       [-1.0, -1.0, -1.0],
                    normal:    [ 0.0, -1.0,  0.0],
                    color:     [ 1.0,  1.0,  1.0],
                    uv:        [ 0.0,  0.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  0.0,  1.0],
                },
                Vertex {
                    pos:       [ 1.0, -1.0, -1.0],
                    normal:    [ 0.0, -1.0,  0.0],
                    color:     [ 0.0,  0.0,  1.0],
                    uv:        [ 1.0,  0.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  0.0,  1.0],
                },
                Vertex {
                    pos:       [ 1.0, -1.0,  1.0],
                    normal:    [ 0.0, -1.0,  0.0],
                    color:     [ 0.0,  1.0,  0.0],
                    uv:        [ 1.0,  1.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  0.0,  1.0],
                },
                Vertex {
                    pos:       [-1.0, -1.0,  1.0],
                    normal:    [ 0.0, -1.0,  0.0],
                    color:     [ 1.0,  0.0,  0.0],
                    uv:        [ 0.0,  1.0],
                    tangent:   [ 1.0,  0.0,  0.0],
                    bitangent: [ 0.0,  0.0,  1.0],
                },
                // +X face (magenta), normal [1,0,0]
                Vertex {
                    pos:       [ 1.0, -1.0,  1.0],
                    normal:    [ 1.0,  0.0,  0.0],
                    color:     [ 1.0,  1.0,  1.0],
                    uv:        [ 0.0,  0.0],
                    tangent:   [ 0.0,  0.0, -1.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                Vertex {
                    pos:       [ 1.0, -1.0, -1.0],
                    normal:    [ 1.0,  0.0,  0.0],
                    color:     [ 0.0,  0.0,  1.0],
                    uv:        [ 1.0,  0.0],
                    tangent:   [ 0.0,  0.0, -1.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                Vertex {
                    pos:       [ 1.0,  1.0, -1.0],
                    normal:    [ 1.0,  0.0,  0.0],
                    color:     [ 0.0,  1.0,  0.0],
                    uv:        [ 1.0,  1.0],
                    tangent:   [ 0.0,  0.0, -1.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                Vertex {
                    pos:       [ 1.0,  1.0,  1.0],
                    normal:    [ 1.0,  0.0,  0.0],
                    color:     [ 1.0,  0.0,  0.0],
                    uv:        [ 0.0,  1.0],
                    tangent:   [ 0.0,  0.0, -1.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                // -X face (cyan), normal [-1,0,0]
                Vertex {
                    pos:       [-1.0, -1.0, -1.0],
                    normal:    [-1.0,  0.0,  0.0],
                    color:     [ 1.0,  1.0,  1.0],
                    uv:        [ 0.0,  0.0],
                    tangent:   [ 0.0,  0.0,  1.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                Vertex {
                    pos:       [-1.0, -1.0,  1.0],
                    normal:    [-1.0,  0.0,  0.0],
                    color:     [ 0.0,  0.0,  1.0],
                    uv:        [ 1.0,  0.0],
                    tangent:   [ 0.0,  0.0,  1.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                Vertex {
                    pos:       [-1.0,  1.0,  1.0],
                    normal:    [-1.0,  0.0,  0.0],
                    color:     [ 0.0,  1.0,  0.0],
                    uv:        [ 1.0,  1.0],
                    tangent:   [ 0.0,  0.0,  1.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
                Vertex {
                    pos:       [-1.0,  1.0, -1.0],
                    normal:    [-1.0,  0.0,  0.0],
                    color:     [ 1.0,  0.0,  0.0],
                    uv:        [ 0.0,  1.0],
                    tangent:   [ 0.0,  0.0,  1.0],
                    bitangent: [ 0.0,  1.0,  0.0],
                },
            ],
            indices: vec![
                 0, 1, 2,  2, 3, 0,     // +Z
                 4, 5, 6,  6, 7, 4,     // -Z
                 8, 9,10, 10,11, 8,     // +Y
                12,13,14, 14,15,12,     // -Y
                16,17,18, 18,19,16,     // +X
                20,21,22, 22,23,20,     // -X
            ],
        }
    }
}

/// Holds GPU‐ready mesh buffers
pub struct LoadedMesh {
    pub name: String,
    pub v_buffer: VertexBuffer,
    pub i_buffer: IndexBuffer,
}

impl LoadedMesh {
    /// Uploads the vertices and indices into a GPU buffer.
    pub fn load(
        name: String,
        instance: &ash::Instance,
        device:   &Device,
        phys:     vk::PhysicalDevice,
        mesh:     &Mesh,
    ) -> Result<Self, Box<dyn Error>> {
        // Grab the raw vertex array from the pure data:
        let verts = &mesh.vertices;
        let indices = &mesh.indices;

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

        Ok(LoadedMesh { name, v_buffer: vb, i_buffer: ib })
    }

    pub fn unit_plane(
        name: String,
        instance: &ash::Instance,
        device:   &Device,
        phys:     vk::PhysicalDevice,
    ) -> Result<Self, Box<dyn Error>> {
        let mesh = Mesh::unit_plane();
        Self::load(name, instance, device, phys, &mesh)
    }

    pub fn cube(
        name: String,
        instance: &ash::Instance,
        device:   &Device,
        phys:     vk::PhysicalDevice,
    ) -> Result<Self, Box<dyn Error>> {
        let mesh = Mesh::cube();
        Self::load(name, instance, device, phys, &mesh)
    }

    /// Record only the indexed draw commands into the given secondary CB
    pub fn record(&self, device: &Device, cmd_buf: vk::CommandBuffer) {
        // Bind & draw
        unsafe {
            device.cmd_bind_vertex_buffers(cmd_buf, 0, &[self.v_buffer.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd_buf, self.i_buffer.buffer, 0, vk::IndexType::UINT32);
            device.cmd_draw_indexed(cmd_buf, self.i_buffer.count, 1, 0, 0, 0);
        }
    }

    /// Frees the GPU buffer and its memory.
    pub fn cleanup(&self, device: &Device) {
        self.v_buffer.cleanup(device);
        self.i_buffer.cleanup(device);
    }
}