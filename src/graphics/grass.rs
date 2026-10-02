//! --------------------------------------------------------------------------------------
//! Instanced Grass (grass.rs)
//!
//! A grass blade is a narrow, segmented ribbon.  Its shape is uploaded once; each instance
//! supplies only a position, size, rotation, and colour variation.  Visible terrain chunks
//! reuse that shared mesh, so a large field does not submit every blade every frame.
//!
//! --------------------------------------------------------------------------------------

use ash::vk;
use bytemuck::{Pod, Zeroable};
use std::error::Error;
use std::mem::size_of;
use vk_mem::{Alloc, Allocation, Allocator, MemoryUsage};

use crate::graphics::mesh::{IndexBuffer, VertexBuffer};
use crate::graphics::pipeline::Pipeline;
use crate::graphics::terrain::TerrainSettings;
use crate::vulkan::base::VulkanBase;

mod generation;
mod pipeline;
#[cfg(test)]
mod tests;
mod upload;
mod wind;

use generation::{
    build_blade_indices_for_segments, build_blade_vertices, build_chunked_grass_instances,
    build_reed_indices, build_reed_instances, build_reed_vertices, chunk_intersects_frustum,
    draw_command, grass_lod,
};
use pipeline::{GrassShaders, grass_push_constants, vertex_input_descriptions};
use upload::GrassUploadBatch;
use wind::{WindMap, build_wind_map_pixels, create_wind_map_resources};

/// Vegetation reaches almost to the valley crest, while leaving the final terrain
/// rollover bare enough that the end of the mesh cannot be mistaken for a grass edge.
const GRASS_FIELD_SIZE: f32 = 160.0;
// The grid retains the original 125 roots per square metre.  Outer roots are thinned
// deterministically below, keeping the centre dense without storing four times as much data.
const GRASS_GRID_SIDE: usize = 1792;
/// Two offset samples per grid cell preserve coverage after halving blade dimensions.
const GRASS_DENSITY_LAYERS: usize = 2;
/// Chunking lets us avoid submitting vegetation outside the camera's nearby view.
const GRASS_CHUNK_SIZE: f32 = 5.0;
const GRASS_CHUNKS_PER_SIDE: usize = (GRASS_FIELD_SIZE / GRASS_CHUNK_SIZE) as usize;
/// Grass is fully dense around the playable scene, then gradually thins on distant slopes.
const FULL_DENSITY_RADIUS: f32 = 40.0;
const OUTER_DENSITY: f32 = 0.30;
/// Only the closest blades need all four curved ribbon segments.
const HIGH_DETAIL_GRASS_DISTANCE: f32 = 12.0;
const NEAR_GRASS_DISTANCE: f32 = 24.0;
const MID_GRASS_DISTANCE: f32 = 100.0;
const REED_DISTANCE: f32 = 64.0;
const MAX_BLADE_WIDTH: f32 = 0.1;
// Instance positions are stored as UNORM16 within these known scene bounds. The small
// margin includes reed roots that spread just beyond the regular grass field.
const INSTANCE_X_RANGE: [f32; 2] = [-80.5, 80.5];
const INSTANCE_Y_RANGE: [f32; 2] = [0.0, 6.0];
const INSTANCE_Z_RANGE: [f32; 2] = [-82.5, 78.5];
const INSTANCE_HEIGHT_RANGE: [f32; 2] = [0.0, 1.0];
const WIND_MAP_SIZE: u32 = 128;
/// Sparse groups of reeds break up the otherwise even lawn-like silhouette.
const TALL_TUFT_COUNT: usize = 1_280;
const TALL_BLADES_PER_TUFT: usize = 9;

/// One vertex in the small, repeated grass-ribbon mesh.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct GrassVertex {
    local_position: [f32; 3],
    local_normal: [f32; 3],
    /// Zero for a green stalk/blade and one for a coloured reed flower head.
    flower_head: f32,
}

/// Data that changes once per blade rather than once per vertex.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct GrassInstance {
    /// World X/Y/Z and height packed into known scene ranges as UNORM16 values.
    position_height: [u16; 4],
    /// Rotation, width, colour variation, and wind phase packed as UNORM16 values.
    /// Vulkan expands these back to `0.0..=1.0` before the vertex shader sees them.
    rotation_width_tint_phase: [u16; 4],
}

/// A contiguous subrange of an instance buffer belonging to one terrain chunk.
#[derive(Clone, Copy, Default)]
struct InstanceRange {
    /// Offset passed to Vulkan as `first_instance`; it does not copy instance data.
    first: u32,
    count: u32,
}

/// One 5×5 m section of the grass field, plus the instance ranges for each LOD.
#[derive(Clone, Copy, Default)]
struct VegetationChunk {
    centre: [f32; 2],
    radius: f32,
    /// Actual world-space vertical bounds; terrain is no longer a flat Y=0 plane.
    min_y: f32,
    max_y: f32,
    near_grass: InstanceRange,
    mid_grass: InstanceRange,
    reeds: InstanceRange,
}

/// Temporary CPU-owned data built during startup before its vectors are uploaded to Vulkan.
///
/// `GrassUploadBatch` borrows this slice while staging it into GPU memory. The renderer then
/// keeps the GPU buffer and small chunk metadata, while Rust drops the temporary `Vec` normally.
struct ChunkedGrassInstances {
    instances: Vec<GrassInstance>,
    chunks: Vec<VegetationChunk>,
}

/// CPU result of one culling pass. The distance is retained for LOD selection and sorting.
#[derive(Clone, Copy)]
struct VisibleChunk {
    index: usize,
    distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GrassLod {
    Near,
    Medium,
    Mid,
}

/// Persistently mapped indirect commands for one swapchain image.
///
/// A separate buffer per image is important: the CPU may prepare this frame while another
/// image's commands are still being consumed by the GPU.
struct IndirectBuffer {
    buffer: vk::Buffer,
    allocation: Allocation,
    mapped: *mut vk::DrawIndexedIndirectCommand,
    capacity: usize,
}

/// Owns the shared blade geometry, per-blade data, and grass-only graphics pipeline.
pub struct GrassRenderer {
    near_blade_vertices: VertexBuffer,
    near_blade_indices: IndexBuffer,
    /// Every grass LOD shares this buffer; each chunk stores its thinned subset first.
    instances: VertexBuffer,
    medium_blade_vertices: VertexBuffer,
    medium_blade_indices: IndexBuffer,
    mid_blade_vertices: VertexBuffer,
    mid_blade_indices: IndexBuffer,
    reed_vertices: VertexBuffer,
    reed_indices: IndexBuffer,
    reed_instances: VertexBuffer,
    chunks: Vec<VegetationChunk>,
    visible_chunks: Vec<VisibleChunk>,
    near_commands: Vec<vk::DrawIndexedIndirectCommand>,
    medium_commands: Vec<vk::DrawIndexedIndirectCommand>,
    mid_commands: Vec<vk::DrawIndexedIndirectCommand>,
    reed_commands: Vec<vk::DrawIndexedIndirectCommand>,
    indirect_buffers: Vec<IndirectBuffer>,
    wind_map: WindMap,
    near_pipeline: Pipeline,
    mid_pipeline: Pipeline,
    shaders: GrassShaders,
}

impl GrassRenderer {
    /// Generates a dense, deterministic field plus sparse flower-reed clusters around the hut.
    pub fn new(vb: &VulkanBase, terrain: TerrainSettings) -> Result<Self, Box<dyn Error>> {
        let near_blade_vertices = build_blade_vertices(4);
        let near_blade_indices = build_blade_indices_for_segments(4);
        let mut chunked_grass = build_chunked_grass_instances(terrain);
        let reed_vertices = build_reed_vertices();
        let reed_indices = build_reed_indices();
        let reed_instances = build_reed_instances(terrain, &mut chunked_grass.chunks);
        let medium_blade_vertices = build_blade_vertices(2);
        let medium_blade_indices = build_blade_indices_for_segments(2);
        let mid_blade_vertices = build_blade_vertices(1);
        let mid_blade_indices = build_blade_indices_for_segments(1);
        let allocator = vb
            .allocator
            .as_ref()
            .expect("allocator exists after Vulkan setup");

        // All immutable geometry, instances, and wind pixels share one transfer submission.
        // The returned buffers live in device-preferred memory; only the small per-frame
        // indirect-command buffers remain host visible.
        let mut uploads = GrassUploadBatch::new(vb)?;
        let (buffer, allocation) = uploads.upload_buffer(
            &vb.device,
            allocator,
            &near_blade_vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let near_blade_vertex_buffer = VertexBuffer { buffer, allocation };
        let (buffer, allocation) = uploads.upload_buffer(
            &vb.device,
            allocator,
            &near_blade_indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        let near_blade_index_buffer = IndexBuffer {
            buffer,
            allocation,
            count: near_blade_indices.len() as u32,
        };
        let (buffer, allocation) = uploads.upload_buffer(
            &vb.device,
            allocator,
            &chunked_grass.instances,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let instance_buffer = VertexBuffer { buffer, allocation };
        let (buffer, allocation) = uploads.upload_buffer(
            &vb.device,
            allocator,
            &medium_blade_vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let medium_blade_vertex_buffer = VertexBuffer { buffer, allocation };
        let (buffer, allocation) = uploads.upload_buffer(
            &vb.device,
            allocator,
            &medium_blade_indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        let medium_blade_index_buffer = IndexBuffer {
            buffer,
            allocation,
            count: medium_blade_indices.len() as u32,
        };
        let (buffer, allocation) = uploads.upload_buffer(
            &vb.device,
            allocator,
            &mid_blade_vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let mid_blade_vertex_buffer = VertexBuffer { buffer, allocation };
        let (buffer, allocation) = uploads.upload_buffer(
            &vb.device,
            allocator,
            &mid_blade_indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        let mid_blade_index_buffer = IndexBuffer {
            buffer,
            allocation,
            count: mid_blade_indices.len() as u32,
        };
        let (buffer, allocation) = uploads.upload_buffer(
            &vb.device,
            allocator,
            &reed_vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let reed_vertex_buffer = VertexBuffer { buffer, allocation };
        let (buffer, allocation) = uploads.upload_buffer(
            &vb.device,
            allocator,
            &reed_indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        let reed_index_buffer = IndexBuffer {
            buffer,
            allocation,
            count: reed_indices.len() as u32,
        };
        let (buffer, allocation) = uploads.upload_buffer(
            &vb.device,
            allocator,
            &reed_instances,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let reed_instance_buffer = VertexBuffer { buffer, allocation };
        let (wind_image, wind_allocation) = uploads.upload_wind_image(&vb.device, allocator)?;
        uploads.finish(vb, allocator)?;
        let wind_map = create_wind_map_resources(&vb.device, wind_image, wind_allocation)?;

        let shaders = GrassShaders::load(&vb.device)?;
        let set_layouts = [vb.set0_global_layout, wind_map.descriptor_set_layout];
        // Grass and reed fragments are fully opaque, so both pipelines skip blending.
        let mut near_pipeline = Pipeline::new_opaque(&vb.device, &set_layouts, true)?;
        let mut mid_pipeline = Pipeline::new_opaque(&vb.device, &set_layouts, true)?;
        let (bindings, attributes) = vertex_input_descriptions();
        near_pipeline.create_graphics_pipeline_with_vertex_input(
            &vb.device,
            vb.pipeline_cache,
            vb.swapchain.extent,
            vb.swapchain.color_format,
            vb.swapchain.depth_format,
            &[&shaders.vertex, &shaders.fragment],
            &vb.engine_settings,
            &bindings,
            &attributes,
            // Each thin ribbon must remain visible from both sides.
            vk::CullModeFlags::NONE,
        )?;
        mid_pipeline.create_graphics_pipeline_with_vertex_input(
            &vb.device,
            vb.pipeline_cache,
            vb.swapchain.extent,
            vb.swapchain.color_format,
            vb.swapchain.depth_format,
            &[&shaders.vertex, &shaders.mid_fragment],
            &vb.engine_settings,
            &bindings,
            &attributes,
            vk::CullModeFlags::NONE,
        )?;
        let indirect_buffers =
            create_indirect_buffers(vb, vb.swapchain.swapchain_image_views.len())?;

        println!(
            "🌱 Uploaded {} grass blades and {} flower reeds (grass LODs: {}/{}/{} triangles, {}-byte instances)",
            chunked_grass.instances.len(),
            reed_instances.len(),
            near_blade_indices.len() / 3,
            medium_blade_indices.len() / 3,
            mid_blade_indices.len() / 3,
            size_of::<GrassInstance>(),
        );

        Ok(Self {
            near_blade_vertices: near_blade_vertex_buffer,
            near_blade_indices: near_blade_index_buffer,
            instances: instance_buffer,
            medium_blade_vertices: medium_blade_vertex_buffer,
            medium_blade_indices: medium_blade_index_buffer,
            mid_blade_vertices: mid_blade_vertex_buffer,
            mid_blade_indices: mid_blade_index_buffer,
            reed_vertices: reed_vertex_buffer,
            reed_indices: reed_index_buffer,
            reed_instances: reed_instance_buffer,
            chunks: chunked_grass.chunks,
            visible_chunks: Vec::with_capacity(GRASS_CHUNKS_PER_SIDE * GRASS_CHUNKS_PER_SIDE),
            near_commands: Vec::with_capacity(GRASS_CHUNKS_PER_SIDE * GRASS_CHUNKS_PER_SIDE),
            medium_commands: Vec::with_capacity(GRASS_CHUNKS_PER_SIDE * GRASS_CHUNKS_PER_SIDE),
            mid_commands: Vec::with_capacity(GRASS_CHUNKS_PER_SIDE * GRASS_CHUNKS_PER_SIDE),
            reed_commands: Vec::with_capacity(GRASS_CHUNKS_PER_SIDE * GRASS_CHUNKS_PER_SIDE),
            indirect_buffers,
            wind_map,
            near_pipeline,
            mid_pipeline,
            shaders,
        })
    }

    /// Rebuilds only the pipeline when swapchain, MSAA, or wireframe state changes.
    pub fn recreate_pipeline(&mut self, vb: &VulkanBase) -> Result<(), Box<dyn Error>> {
        let (bindings, attributes) = vertex_input_descriptions();
        self.near_pipeline.recreate_with_vertex_input(
            &vb.device,
            vb.pipeline_cache,
            vb.swapchain.extent,
            vb.swapchain.color_format,
            vb.swapchain.depth_format,
            &[&self.shaders.vertex, &self.shaders.fragment],
            &vb.engine_settings,
            &bindings,
            &attributes,
            vk::CullModeFlags::NONE,
        )?;
        self.mid_pipeline.recreate_with_vertex_input(
            &vb.device,
            vb.pipeline_cache,
            vb.swapchain.extent,
            vb.swapchain.color_format,
            vb.swapchain.depth_format,
            &[&self.shaders.vertex, &self.shaders.mid_fragment],
            &vb.engine_settings,
            &bindings,
            &attributes,
            vk::CullModeFlags::NONE,
        )?;
        if self.indirect_buffers.len() != vb.swapchain.swapchain_image_views.len() {
            let allocator = vb.allocator.as_ref().expect("allocator");
            cleanup_indirect_buffers(allocator, &mut self.indirect_buffers);
            self.indirect_buffers =
                create_indirect_buffers(vb, vb.swapchain.swapchain_image_views.len())?;
        }
        Ok(())
    }

    /// Records visible nearby chunks at full density and distant chunks at a cheaper LOD.
    pub fn draw(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        vb: &VulkanBase,
        image_index: usize,
        camera: &crate::graphics::camera::Camera,
        time_seconds: f32,
    ) {
        self.prepare_draw_commands(camera);
        let near_push = grass_push_constants(camera, time_seconds, 0.0);
        let medium_push = grass_push_constants(camera, time_seconds, 0.5);
        let mid_push = grass_push_constants(camera, time_seconds, 1.0);
        let near_buffers = [self.near_blade_vertices.buffer, self.instances.buffer];
        let offsets = [0, 0];
        let descriptor_sets = [
            vb.set0_descriptor_sets[image_index],
            self.wind_map.descriptor_set,
        ];

        let (near_offset, medium_offset, mid_offset, reed_offset) =
            self.write_indirect_commands(vb.allocator.as_ref().expect("allocator"), image_index);
        let indirect = &self.indirect_buffers[image_index];

        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.near_pipeline.vk_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.near_pipeline.vk_layout,
                0,
                &descriptor_sets,
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.near_pipeline.vk_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &near_push,
            );
            // The first bound buffer is shared blade geometry; the second advances once per
            // instance.  Each chunk draw selects its range through `first_instance` below.
            device.cmd_bind_vertex_buffers(cmd, 0, &near_buffers, &offsets);
            device.cmd_bind_index_buffer(
                cmd,
                self.near_blade_indices.buffer,
                0,
                vk::IndexType::UINT32,
            );
            self.submit_commands(
                device,
                cmd,
                indirect.buffer,
                near_offset,
                &self.near_commands,
                vb.supports_multi_draw_indirect,
                vb.max_draw_indirect_count,
            );

            // The next ring keeps every root but halves each blade's geometry.
            device.cmd_push_constants(
                cmd,
                self.near_pipeline.vk_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &medium_push,
            );
            let medium_buffers = [self.medium_blade_vertices.buffer, self.instances.buffer];
            device.cmd_bind_vertex_buffers(cmd, 0, &medium_buffers, &offsets);
            device.cmd_bind_index_buffer(
                cmd,
                self.medium_blade_indices.buffer,
                0,
                vk::IndexType::UINT32,
            );
            self.submit_commands(
                device,
                cmd,
                indirect.buffer,
                medium_offset,
                &self.medium_commands,
                vb.supports_multi_draw_indirect,
                vb.max_draw_indirect_count,
            );

            // Mid-distance grass uses one ribbon segment and half of the instances.
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.mid_pipeline.vk_pipeline,
            );
            device.cmd_push_constants(
                cmd,
                self.mid_pipeline.vk_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &mid_push,
            );
            let mid_buffers = [self.mid_blade_vertices.buffer, self.instances.buffer];
            device.cmd_bind_vertex_buffers(cmd, 0, &mid_buffers, &offsets);
            device.cmd_bind_index_buffer(
                cmd,
                self.mid_blade_indices.buffer,
                0,
                vk::IndexType::UINT32,
            );
            self.submit_commands(
                device,
                cmd,
                indirect.buffer,
                mid_offset,
                &self.mid_commands,
                vb.supports_multi_draw_indirect,
                vb.max_draw_indirect_count,
            );

            // Reeds share the shader and instance layout with grass, but use a different
            // repeated mesh: a narrow stalk capped by a low-poly flower capsule.
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.near_pipeline.vk_pipeline,
            );
            device.cmd_push_constants(
                cmd,
                self.near_pipeline.vk_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &near_push,
            );
            let reed_buffers = [self.reed_vertices.buffer, self.reed_instances.buffer];
            device.cmd_bind_vertex_buffers(cmd, 0, &reed_buffers, &offsets);
            device.cmd_bind_index_buffer(cmd, self.reed_indices.buffer, 0, vk::IndexType::UINT32);
            self.submit_commands(
                device,
                cmd,
                indirect.buffer,
                reed_offset,
                &self.reed_commands,
                vb.supports_multi_draw_indirect,
                vb.max_draw_indirect_count,
            );
        }
    }

    /// Culls every chunk once, sorts front-to-back for early depth rejection, then builds the
    /// LOD command lists without allocating new vectors.
    fn prepare_draw_commands(&mut self, camera: &crate::graphics::camera::Camera) {
        self.visible_chunks.clear();
        self.near_commands.clear();
        self.medium_commands.clear();
        self.mid_commands.clear();
        self.reed_commands.clear();

        let view_projection = *camera.get_projection() * *camera.get_view();
        let camera_position = camera.position();
        for (index, chunk) in self.chunks.iter().enumerate() {
            let dx = chunk.centre[0] - camera_position.x;
            let dz = chunk.centre[1] - camera_position.z;
            let distance = ((dx * dx + dz * dz).sqrt() - chunk.radius).max(0.0);
            // A conservative corner test can reject a box intersected by the near plane.
            // Always retain the chunk under the player, then frustum-test the rest.
            let camera_is_over_chunk =
                dx.abs() <= GRASS_CHUNK_SIZE * 0.5 && dz.abs() <= GRASS_CHUNK_SIZE * 0.5;
            if distance <= MID_GRASS_DISTANCE
                && (camera_is_over_chunk || chunk_intersects_frustum(view_projection, chunk))
            {
                self.visible_chunks.push(VisibleChunk { index, distance });
            }
        }
        self.visible_chunks
            .sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));

        for visible in &self.visible_chunks {
            let chunk = &self.chunks[visible.index];
            match grass_lod(visible.distance) {
                GrassLod::Near if chunk.near_grass.count > 0 => {
                    self.near_commands.push(draw_command(
                        self.near_blade_indices.count,
                        chunk.near_grass,
                    ));
                }
                GrassLod::Medium if chunk.near_grass.count > 0 => {
                    self.medium_commands.push(draw_command(
                        self.medium_blade_indices.count,
                        chunk.near_grass,
                    ));
                }
                GrassLod::Mid if chunk.mid_grass.count > 0 => {
                    self.mid_commands
                        .push(draw_command(self.mid_blade_indices.count, chunk.mid_grass));
                }
                _ => {}
            }
            if visible.distance <= REED_DISTANCE && chunk.reeds.count > 0 {
                self.reed_commands
                    .push(draw_command(self.reed_indices.count, chunk.reeds));
            }
        }
    }

    /// Copies this frame's compact command lists into the acquired image's mapped buffer.
    fn write_indirect_commands(
        &self,
        allocator: &Allocator,
        image_index: usize,
    ) -> (
        vk::DeviceSize,
        vk::DeviceSize,
        vk::DeviceSize,
        vk::DeviceSize,
    ) {
        let indirect = &self.indirect_buffers[image_index];
        let stride = size_of::<vk::DrawIndexedIndirectCommand>();
        let near_first = 0;
        let medium_first = self.near_commands.len();
        let mid_first = medium_first + self.medium_commands.len();
        let reed_first = mid_first + self.mid_commands.len();
        let total = reed_first + self.reed_commands.len();
        debug_assert!(total <= indirect.capacity);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.near_commands.as_ptr(),
                indirect.mapped.add(near_first),
                self.near_commands.len(),
            );
            std::ptr::copy_nonoverlapping(
                self.medium_commands.as_ptr(),
                indirect.mapped.add(medium_first),
                self.medium_commands.len(),
            );
            std::ptr::copy_nonoverlapping(
                self.mid_commands.as_ptr(),
                indirect.mapped.add(mid_first),
                self.mid_commands.len(),
            );
            std::ptr::copy_nonoverlapping(
                self.reed_commands.as_ptr(),
                indirect.mapped.add(reed_first),
                self.reed_commands.len(),
            );
        }
        allocator
            .flush_allocation(&indirect.allocation, 0, (total * stride) as u64)
            .expect("flush vegetation indirect commands");
        (
            (near_first * stride) as u64,
            (medium_first * stride) as u64,
            (mid_first * stride) as u64,
            (reed_first * stride) as u64,
        )
    }

    /// Uses one multi-draw call when supported, retaining a portable per-command fallback.
    // These values map directly to one Vulkan indirect-draw call; grouping them
    // would hide rather than simplify that API boundary.
    #[allow(clippy::too_many_arguments)]
    unsafe fn submit_commands(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        commands: &[vk::DrawIndexedIndirectCommand],
        supports_multi_draw: bool,
        max_draw_count: u32,
    ) {
        let stride = size_of::<vk::DrawIndexedIndirectCommand>() as u32;
        if commands.is_empty() {
            return;
        }
        if supports_multi_draw {
            let batch_size = max_draw_count.max(1) as usize;
            for first in (0..commands.len()).step_by(batch_size) {
                let count = (commands.len() - first).min(batch_size) as u32;
                unsafe {
                    device.cmd_draw_indexed_indirect(
                        cmd,
                        buffer,
                        offset + first as u64 * stride as u64,
                        count,
                        stride,
                    )
                };
            }
        } else {
            for index in 0..commands.len() {
                unsafe {
                    device.cmd_draw_indexed_indirect(
                        cmd,
                        buffer,
                        offset + (index as u64 * stride as u64),
                        1,
                        stride,
                    )
                };
            }
        }
    }

    /// Releases grass-owned Vulkan objects before the device is dropped.
    pub fn cleanup(&mut self, device: &ash::Device, allocator: &Allocator) {
        self.near_pipeline.cleanup(device);
        self.mid_pipeline.cleanup(device);
        self.near_blade_vertices.cleanup(allocator);
        self.near_blade_indices.cleanup(allocator);
        self.instances.cleanup(allocator);
        self.medium_blade_vertices.cleanup(allocator);
        self.medium_blade_indices.cleanup(allocator);
        self.mid_blade_vertices.cleanup(allocator);
        self.mid_blade_indices.cleanup(allocator);
        self.reed_vertices.cleanup(allocator);
        self.reed_indices.cleanup(allocator);
        self.reed_instances.cleanup(allocator);
        cleanup_indirect_buffers(allocator, &mut self.indirect_buffers);
        self.wind_map.cleanup(device, allocator);
    }
}

fn create_indirect_buffers(
    vb: &VulkanBase,
    count: usize,
) -> Result<Vec<IndirectBuffer>, vk::Result> {
    let allocator = vb.allocator.as_ref().expect("allocator");
    // Worst case: one command per chunk in each of three grass LODs plus reeds.
    let capacity = GRASS_CHUNKS_PER_SIDE * GRASS_CHUNKS_PER_SIDE * 4;
    let size = (capacity * size_of::<vk::DrawIndexedIndirectCommand>()) as u64;
    let mut buffers = Vec::with_capacity(count);
    for _ in 0..count {
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::INDIRECT_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };
        let (buffer, allocation) =
            unsafe { allocator.create_buffer(&buffer_info, &allocation_info)? };
        let mapped = allocator.get_allocation_info(&allocation).mapped_data
            as *mut vk::DrawIndexedIndirectCommand;
        debug_assert!(!mapped.is_null());
        buffers.push(IndirectBuffer {
            buffer,
            allocation,
            mapped,
            capacity,
        });
    }
    Ok(buffers)
}

fn cleanup_indirect_buffers(allocator: &Allocator, buffers: &mut Vec<IndirectBuffer>) {
    for mut indirect in buffers.drain(..) {
        unsafe { allocator.destroy_buffer(indirect.buffer, &mut indirect.allocation) };
    }
}
