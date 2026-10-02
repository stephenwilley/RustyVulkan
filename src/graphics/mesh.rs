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

use ash::Device;
use ash::vk;
use bytemuck::{Pod, Zeroable, offset_of};
use cgmath::{Matrix4, Vector4};
use std::error::Error;
use vk_mem::{Alloc, Allocation, Allocator, MemoryUsage};

/// A single vertex: 3D position + normal, color, uv, tangent, bitangent.
#[repr(C)]
#[derive(Default, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub uv: [f32; 2],
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
    pub allocation: Allocation,
}

impl VertexBuffer {
    /// Create a new vertex buffer, uploading any POD vertex or instance data.
    /// using host-visible memory selected by VMA.
    /// # Arguments
    /// * `allocator` - Global Vulkan memory allocator.
    /// * `data` - The vertex data to upload.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - Returns the initialized `VertexBuffer` on success, or an error on failure.
    pub fn new<T: Pod>(allocator: &Allocator, data: &[T]) -> Result<Self, Box<dyn Error>> {
        let size = std::mem::size_of_val(data) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };
        let (buffer, mut allocation) =
            unsafe { allocator.create_buffer(&buffer_info, &alloc_info)? };

        unsafe {
            let ptr = allocator.map_memory(&mut allocation)? as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            // Host-visible memory is not guaranteed to be coherent on every GPU.
            allocator.flush_allocation(&allocation, 0, size)?;
            allocator.unmap_memory(&mut allocation);
        }

        Ok(VertexBuffer { buffer, allocation })
    }

    /// Frees the Vulkan buffer and its backing allocation.
    pub fn cleanup(&mut self, allocator: &Allocator) {
        unsafe {
            allocator.destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}

/// A GPU-resident index buffer for indexed drawing.
pub struct IndexBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub count: u32,
}

impl IndexBuffer {
    /// Create a new index buffer, uploading `indices` (slice of u32).
    /// # Arguments
    /// * `allocator` - Global Vulkan memory allocator.
    /// * `data` - The index data to upload.
    /// # Returns
    /// * `Result<Self, vk::Result>` - Returns the initialized `IndexBuffer` on success, or an error on failure.
    pub fn new(allocator: &Allocator, data: &[u32]) -> Result<Self, vk::Result> {
        let size = std::mem::size_of_val(data) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::INDEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };
        let (buffer, mut allocation) =
            unsafe { allocator.create_buffer(&buffer_info, &alloc_info)? };

        unsafe {
            let ptr = allocator.map_memory(&mut allocation)? as *mut u32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            allocator.flush_allocation(&allocation, 0, size)?;
            allocator.unmap_memory(&mut allocation);
        }

        Ok(IndexBuffer {
            buffer,
            allocation,
            count: data.len() as u32,
        })
    }

    /// Frees the Vulkan buffer and its backing allocation.
    pub fn cleanup(&mut self, allocator: &Allocator) {
        unsafe {
            allocator.destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}

/// A CPU-side representation of a mesh, containing vertices and indices.
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// Axis-aligned limits retained after CPU vertices have been uploaded and discarded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshBounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl MeshBounds {
    fn from_vertices(vertices: &[Vertex]) -> Option<Self> {
        let first = vertices.first()?.pos;
        let mut bounds = Self {
            min: first,
            max: first,
        };
        for vertex in &vertices[1..] {
            for axis in 0..3 {
                bounds.min[axis] = bounds.min[axis].min(vertex.pos[axis]);
                bounds.max[axis] = bounds.max[axis].max(vertex.pos[axis]);
            }
        }
        Some(bounds)
    }

    /// Finds the lowest Y reached by this box after a part-local transform.
    /// Testing all eight corners also handles rotated imported parts correctly.
    pub fn transformed_min_y(&self, transform: Matrix4<f32>) -> f32 {
        let mut minimum_y = f32::INFINITY;
        for x in [self.min[0], self.max[0]] {
            for y in [self.min[1], self.max[1]] {
                for z in [self.min[2], self.max[2]] {
                    let point = transform * Vector4::new(x, y, z, 1.0);
                    minimum_y = minimum_y.min(point.y);
                }
            }
        }
        minimum_y
    }

    /// Returns an axis-aligned box containing this box after an arbitrary transform.
    pub fn transformed(&self, transform: Matrix4<f32>) -> Self {
        let mut result = Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        };
        for x in [self.min[0], self.max[0]] {
            for y in [self.min[1], self.max[1]] {
                for z in [self.min[2], self.max[2]] {
                    let point = transform * Vector4::new(x, y, z, 1.0);
                    for (axis, value) in [point.x, point.y, point.z].into_iter().enumerate() {
                        result.min[axis] = result.min[axis].min(value);
                        result.max[axis] = result.max[axis].max(value);
                    }
                }
            }
        }
        result
    }

    /// Expands this box to include another box.
    pub fn include(&mut self, other: Self) {
        for axis in 0..3 {
            self.min[axis] = self.min[axis].min(other.min[axis]);
            self.max[axis] = self.max[axis].max(other.max[axis]);
        }
    }
}

impl Mesh {
    /// Creates a new empty `Mesh`.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Creates a unit plane mesh.
    pub fn unit_plane() -> Self {
        Self {
            vertices: vec![
                Vertex {
                    pos: [-1.0, 0.0, 1.0],
                    normal: [0.0, 1.0, 0.0],
                    color: [1.0, 1.0, 1.0],
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, -1.0],
                },
                Vertex {
                    pos: [1.0, 0.0, 1.0],
                    normal: [0.0, 1.0, 0.0],
                    color: [0.0, 0.0, 1.0],
                    uv: [1.0, 0.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, -1.0],
                },
                Vertex {
                    pos: [1.0, 0.0, -1.0],
                    normal: [0.0, 1.0, 0.0],
                    color: [0.0, 1.0, 0.0],
                    uv: [1.0, 1.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, -1.0],
                },
                Vertex {
                    pos: [-1.0, 0.0, -1.0],
                    normal: [0.0, 1.0, 0.0],
                    color: [1.0, 0.0, 0.0],
                    uv: [0.0, 1.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, -1.0],
                },
            ],
            indices: vec![0, 1, 2, 2, 3, 0],
        }
    }

    /// Creates a unit cube mesh.
    pub fn cube() -> Self {
        Self {
            vertices: vec![
                // +Z face (blue), normal [0,0,1]
                Vertex {
                    pos: [-1.0, -1.0, 1.0],
                    normal: [0.0, 0.0, 1.0],
                    color: [1.0, 1.0, 1.0],
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                Vertex {
                    pos: [1.0, -1.0, 1.0],
                    normal: [0.0, 0.0, 1.0],
                    color: [0.0, 0.0, 1.0],
                    uv: [1.0, 0.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                Vertex {
                    pos: [1.0, 1.0, 1.0],
                    normal: [0.0, 0.0, 1.0],
                    color: [0.0, 1.0, 0.0],
                    uv: [1.0, 1.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                Vertex {
                    pos: [-1.0, 1.0, 1.0],
                    normal: [0.0, 0.0, 1.0],
                    color: [1.0, 0.0, 0.0],
                    uv: [0.0, 1.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                // -Z face (green), normal [0,0,-1]
                Vertex {
                    pos: [1.0, -1.0, -1.0],
                    normal: [0.0, 0.0, -1.0],
                    color: [1.0, 1.0, 1.0],
                    uv: [0.0, 0.0],
                    tangent: [-1.0, 0.0, 0.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                Vertex {
                    pos: [-1.0, -1.0, -1.0],
                    normal: [0.0, 0.0, -1.0],
                    color: [0.0, 0.0, 1.0],
                    uv: [1.0, 0.0],
                    tangent: [-1.0, 0.0, 0.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                Vertex {
                    pos: [-1.0, 1.0, -1.0],
                    normal: [0.0, 0.0, -1.0],
                    color: [0.0, 1.0, 0.0],
                    uv: [1.0, 1.0],
                    tangent: [-1.0, 0.0, 0.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                Vertex {
                    pos: [1.0, 1.0, -1.0],
                    normal: [0.0, 0.0, -1.0],
                    color: [1.0, 0.0, 0.0],
                    uv: [0.0, 1.0],
                    tangent: [-1.0, 0.0, 0.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                // +Y face (red), normal [0,1,0]
                Vertex {
                    pos: [-1.0, 1.0, 1.0],
                    normal: [0.0, 1.0, 0.0],
                    color: [1.0, 1.0, 1.0],
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, -1.0],
                },
                Vertex {
                    pos: [1.0, 1.0, 1.0],
                    normal: [0.0, 1.0, 0.0],
                    color: [0.0, 0.0, 1.0],
                    uv: [1.0, 0.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, -1.0],
                },
                Vertex {
                    pos: [1.0, 1.0, -1.0],
                    normal: [0.0, 1.0, 0.0],
                    color: [0.0, 1.0, 0.0],
                    uv: [1.0, 1.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, -1.0],
                },
                Vertex {
                    pos: [-1.0, 1.0, -1.0],
                    normal: [0.0, 1.0, 0.0],
                    color: [1.0, 0.0, 0.0],
                    uv: [0.0, 1.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, -1.0],
                },
                // -Y face (yellow), normal [0,-1,0]
                Vertex {
                    pos: [-1.0, -1.0, -1.0],
                    normal: [0.0, -1.0, 0.0],
                    color: [1.0, 1.0, 1.0],
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, 1.0],
                },
                Vertex {
                    pos: [1.0, -1.0, -1.0],
                    normal: [0.0, -1.0, 0.0],
                    color: [0.0, 0.0, 1.0],
                    uv: [1.0, 0.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, 1.0],
                },
                Vertex {
                    pos: [1.0, -1.0, 1.0],
                    normal: [0.0, -1.0, 0.0],
                    color: [0.0, 1.0, 0.0],
                    uv: [1.0, 1.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, 1.0],
                },
                Vertex {
                    pos: [-1.0, -1.0, 1.0],
                    normal: [0.0, -1.0, 0.0],
                    color: [1.0, 0.0, 0.0],
                    uv: [0.0, 1.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, 1.0],
                },
                // +X face (magenta), normal [1,0,0]
                Vertex {
                    pos: [1.0, -1.0, 1.0],
                    normal: [1.0, 0.0, 0.0],
                    color: [1.0, 1.0, 1.0],
                    uv: [0.0, 0.0],
                    tangent: [0.0, 0.0, -1.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                Vertex {
                    pos: [1.0, -1.0, -1.0],
                    normal: [1.0, 0.0, 0.0],
                    color: [0.0, 0.0, 1.0],
                    uv: [1.0, 0.0],
                    tangent: [0.0, 0.0, -1.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                Vertex {
                    pos: [1.0, 1.0, -1.0],
                    normal: [1.0, 0.0, 0.0],
                    color: [0.0, 1.0, 0.0],
                    uv: [1.0, 1.0],
                    tangent: [0.0, 0.0, -1.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                Vertex {
                    pos: [1.0, 1.0, 1.0],
                    normal: [1.0, 0.0, 0.0],
                    color: [1.0, 0.0, 0.0],
                    uv: [0.0, 1.0],
                    tangent: [0.0, 0.0, -1.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                // -X face (cyan), normal [-1,0,0]
                Vertex {
                    pos: [-1.0, -1.0, -1.0],
                    normal: [-1.0, 0.0, 0.0],
                    color: [1.0, 1.0, 1.0],
                    uv: [0.0, 0.0],
                    tangent: [0.0, 0.0, 1.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                Vertex {
                    pos: [-1.0, -1.0, 1.0],
                    normal: [-1.0, 0.0, 0.0],
                    color: [0.0, 0.0, 1.0],
                    uv: [1.0, 0.0],
                    tangent: [0.0, 0.0, 1.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                Vertex {
                    pos: [-1.0, 1.0, 1.0],
                    normal: [-1.0, 0.0, 0.0],
                    color: [0.0, 1.0, 0.0],
                    uv: [1.0, 1.0],
                    tangent: [0.0, 0.0, 1.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
                Vertex {
                    pos: [-1.0, 1.0, -1.0],
                    normal: [-1.0, 0.0, 0.0],
                    color: [1.0, 0.0, 0.0],
                    uv: [0.0, 1.0],
                    tangent: [0.0, 0.0, 1.0],
                    bitangent: [0.0, 1.0, 0.0],
                },
            ],
            indices: vec![
                0, 1, 2, 2, 3, 0, // +Z
                4, 5, 6, 6, 7, 4, // -Z
                8, 9, 10, 10, 11, 8, // +Y
                12, 13, 14, 14, 15, 12, // -Y
                16, 17, 18, 18, 19, 16, // +X
                20, 21, 22, 22, 23, 20, // -X
            ],
        }
    }
}

/// Holds GPU‐ready mesh buffers
pub struct LoadedMesh {
    pub name: String,
    pub v_buffer: VertexBuffer,
    pub i_buffer: IndexBuffer,
    /// CPU-calculated bounds used for placement and future visibility tests.
    pub bounds: MeshBounds,
}

impl LoadedMesh {
    /// Uploads the vertices and indices into a GPU buffer.
    /// # Arguments
    /// * `name` - The name of the mesh.
    /// * `allocator` - Global Vulkan memory allocator.
    /// * `mesh` - The CPU-side mesh data.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - Returns the initialized `LoadedMesh` on success, or an error on failure.
    pub fn load(name: String, allocator: &Allocator, mesh: &Mesh) -> Result<Self, Box<dyn Error>> {
        let verts = &mesh.vertices;
        let indices = &mesh.indices;
        let bounds = MeshBounds::from_vertices(verts).ok_or("cannot upload an empty mesh")?;

        let vb = VertexBuffer::new(allocator, verts)?;
        let ib = IndexBuffer::new(allocator, indices)?;

        Ok(LoadedMesh {
            name,
            v_buffer: vb,
            i_buffer: ib,
            bounds,
        })
    }

    /// Creates a unit plane mesh and uploads it to the GPU.
    /// # Arguments
    /// * `name` - The name of the mesh.
    /// * `allocator` - Global Vulkan memory allocator.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - Returns the initialized `LoadedMesh` on success, or an error on failure.
    pub fn unit_plane(name: String, allocator: &Allocator) -> Result<Self, Box<dyn Error>> {
        let mesh = Mesh::unit_plane();
        Self::load(name, allocator, &mesh)
    }

    /// Creates a unit cube mesh and uploads it to the GPU.
    /// # Arguments
    /// * `name` - The name of the mesh.
    /// * `allocator` - Global Vulkan memory allocator.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - Returns the initialized `LoadedMesh` on success, or an error on failure.
    pub fn cube(name: String, allocator: &Allocator) -> Result<Self, Box<dyn Error>> {
        let mesh = Mesh::cube();
        Self::load(name, allocator, &mesh)
    }

    /// Record only the indexed draw commands into the given secondary CB
    /// # Arguments
    /// * `device` - The Vulkan device.
    /// * `cmd_buf` - The command buffer to record into.
    pub fn record(&self, device: &Device, cmd_buf: vk::CommandBuffer) {
        // Bind & draw
        unsafe {
            device.cmd_bind_vertex_buffers(cmd_buf, 0, &[self.v_buffer.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd_buf, self.i_buffer.buffer, 0, vk::IndexType::UINT32);
            device.cmd_draw_indexed(cmd_buf, self.i_buffer.count, 1, 0, 0, 0);
        }
    }

    /// Frees the GPU buffer and its memory.
    /// # Arguments
    /// * `allocator` - Global Vulkan memory allocator.
    pub fn cleanup(&mut self, allocator: &Allocator) {
        self.v_buffer.cleanup(allocator);
        self.i_buffer.cleanup(allocator);
    }
}

#[cfg(test)]
mod tests {
    use super::{Mesh, MeshBounds};
    use cgmath::{Matrix4, Vector3};

    #[test]
    fn retained_bounds_find_the_transformed_bottom_of_a_mesh() {
        let cube = Mesh::cube();
        let bounds = MeshBounds::from_vertices(&cube.vertices).unwrap();
        assert_eq!(bounds.min, [-1.0, -1.0, -1.0]);
        assert_eq!(bounds.max, [1.0, 1.0, 1.0]);

        let transform =
            Matrix4::from_translation(Vector3::new(0.0, 5.0, 0.0)) * Matrix4::from_scale(0.5);
        assert_eq!(bounds.transformed_min_y(transform), 4.5);
    }
}
