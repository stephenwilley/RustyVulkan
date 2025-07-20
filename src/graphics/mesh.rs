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

/// A single vertex: 2D position + RGB color.
#[repr(C)]
#[derive(Default, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos:   [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
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
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
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
        let size = (std::mem::size_of::<Vertex>() * data.len()) as vk::DeviceSize;

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
    /// Create a new index buffer, uploading `indices` (slice of u16).
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
        data: &[u16],
    ) -> Result<Self, vk::Result> {
        let size = (std::mem::size_of::<u16>() * data.len()) as vk::DeviceSize;

        // 1) create buffer with usage INDEX_BUFFER
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::INDEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        // 2) allocate & bind DEVICE_LOCAL memory
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

        // 3) map staging, copy data, submit staging→device_local
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