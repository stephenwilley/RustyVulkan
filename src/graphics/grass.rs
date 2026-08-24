//! --------------------------------------------------------------------------------------
//! Instanced Grass (grass.rs)
//!
//! A grass blade is a narrow, segmented ribbon.  Its shape is uploaded once; each instance
//! supplies only a position, size, rotation, and colour variation.  Visible terrain chunks
//! reuse that shared mesh, so a large field does not submit every blade every frame.
//!
//! --------------------------------------------------------------------------------------

use ash::vk;
use bytemuck::{Pod, Zeroable, offset_of};
use cgmath::{Matrix4, SquareMatrix, Vector4};
use std::error::Error;
use std::mem::size_of;
use vk_mem::{Alloc, Allocation, Allocator, MemoryUsage};

use crate::graphics::mesh::{IndexBuffer, VertexBuffer};
use crate::graphics::pipeline::Pipeline;
use crate::graphics::shaders::{ShaderModule, ShaderStageInfo};
use crate::graphics::terrain::TerrainSettings;
use crate::vulkan::base::VulkanBase;
use crate::vulkan::main_pass::compute_push_constant_per_obj;

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
const NEAR_GRASS_DISTANCE: f32 = 24.0;
const MID_GRASS_DISTANCE: f32 = 100.0;
const REED_DISTANCE: f32 = 64.0;
const MAX_BLADE_WIDTH: f32 = 0.1;
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
    /// World X/Y/Z, followed by blade height in metres.
    position_height: [f32; 4],
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

/// Grass-owned wind image and descriptor resources.
struct WindMap {
    image: vk::Image,
    allocation: Allocation,
    view: vk::ImageView,
    sampler: vk::Sampler,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
}

/// Owns the shared blade geometry, per-blade data, and grass-only graphics pipeline.
pub struct GrassRenderer {
    near_blade_vertices: VertexBuffer,
    near_blade_indices: IndexBuffer,
    /// Near and mid LODs share this buffer; each chunk stores its mid subset first.
    instances: VertexBuffer,
    mid_blade_vertices: VertexBuffer,
    mid_blade_indices: IndexBuffer,
    reed_vertices: VertexBuffer,
    reed_indices: IndexBuffer,
    reed_instances: VertexBuffer,
    chunks: Vec<VegetationChunk>,
    visible_chunks: Vec<VisibleChunk>,
    near_commands: Vec<vk::DrawIndexedIndirectCommand>,
    mid_commands: Vec<vk::DrawIndexedIndirectCommand>,
    reed_commands: Vec<vk::DrawIndexedIndirectCommand>,
    indirect_buffers: Vec<IndirectBuffer>,
    wind_map: WindMap,
    near_pipeline: Pipeline,
    mid_pipeline: Pipeline,
    shaders: GrassShaders,
}

/// Records every immutable grass buffer and the wind image into one startup transfer.
struct GrassUploadBatch {
    command_buffer: vk::CommandBuffer,
    staging_buffers: Vec<(vk::Buffer, Allocation)>,
}

impl GrassUploadBatch {
    fn new(vb: &VulkanBase) -> Result<Self, vk::Result> {
        let allocate_info = vk::CommandBufferAllocateInfo {
            command_pool: vb.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        let command_buffer = unsafe { vb.device.allocate_command_buffers(&allocate_info)?[0] };
        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe {
            vb.device
                .begin_command_buffer(command_buffer, &begin_info)?
        };
        Ok(Self {
            command_buffer,
            staging_buffers: Vec::new(),
        })
    }

    fn upload_buffer<T: Pod>(
        &mut self,
        device: &ash::Device,
        allocator: &Allocator,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, Allocation), vk::Result> {
        let byte_size = std::mem::size_of_val(data) as vk::DeviceSize;
        let staging_info = vk::BufferCreateInfo {
            size: byte_size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let staging_alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        };
        let (staging, mut staging_allocation) =
            unsafe { allocator.create_buffer(&staging_info, &staging_alloc_info)? };
        unsafe {
            let destination = allocator.map_memory(&mut staging_allocation)? as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), destination, data.len());
            allocator.flush_allocation(&staging_allocation, 0, byte_size)?;
            allocator.unmap_memory(&mut staging_allocation);
        }

        let device_info = vk::BufferCreateInfo {
            size: byte_size,
            usage: usage | vk::BufferUsageFlags::TRANSFER_DST,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let device_alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        let (buffer, allocation) =
            unsafe { allocator.create_buffer(&device_info, &device_alloc_info)? };
        let copy = vk::BufferCopy {
            size: byte_size,
            ..Default::default()
        };
        unsafe { device.cmd_copy_buffer(self.command_buffer, staging, buffer, &[copy]) };
        self.staging_buffers.push((staging, staging_allocation));
        Ok((buffer, allocation))
    }

    fn upload_wind_image(
        &mut self,
        device: &ash::Device,
        allocator: &Allocator,
    ) -> Result<(vk::Image, Allocation), vk::Result> {
        let pixels = build_wind_map_pixels();
        let staging_info = vk::BufferCreateInfo {
            size: pixels.len() as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let staging_alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        };
        let (staging, mut staging_allocation) =
            unsafe { allocator.create_buffer(&staging_info, &staging_alloc_info)? };
        unsafe {
            let destination = allocator.map_memory(&mut staging_allocation)?;
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), destination, pixels.len());
            allocator.flush_allocation(&staging_allocation, 0, pixels.len() as u64)?;
            allocator.unmap_memory(&mut staging_allocation);
        }

        let image_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D {
                width: WIND_MAP_SIZE,
                height: WIND_MAP_SIZE,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let image_alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        let (image, allocation) =
            unsafe { allocator.create_image(&image_info, &image_alloc_info)? };
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let to_transfer = vk::ImageMemoryBarrier {
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            image,
            subresource_range: range,
            ..Default::default()
        };
        let to_shader = vk::ImageMemoryBarrier {
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            image,
            subresource_range: range,
            ..Default::default()
        };
        let copy = vk::BufferImageCopy {
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: vk::Extent3D {
                width: WIND_MAP_SIZE,
                height: WIND_MAP_SIZE,
                depth: 1,
            },
            ..Default::default()
        };
        unsafe {
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer],
            );
            device.cmd_copy_buffer_to_image(
                self.command_buffer,
                staging,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_shader],
            );
        }
        self.staging_buffers.push((staging, staging_allocation));
        Ok((image, allocation))
    }

    fn finish(mut self, vb: &VulkanBase, allocator: &Allocator) -> Result<(), vk::Result> {
        unsafe {
            // Make every transfer write visible to later vertex/index fetches. Queue order alone
            // orders execution, but this barrier supplies the required memory dependency.
            let barrier = vk::MemoryBarrier {
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::VERTEX_ATTRIBUTE_READ
                    | vk::AccessFlags::INDEX_READ,
                ..Default::default()
            };
            vb.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            vb.device.end_command_buffer(self.command_buffer)?;
            let fence = vb
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;
            let submit = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &self.command_buffer,
                ..Default::default()
            };
            vb.device
                .queue_submit(vb.graphics_queue, &[submit], fence)?;
            vb.device.wait_for_fences(&[fence], true, u64::MAX)?;
            vb.device.destroy_fence(fence, None);
            vb.device
                .free_command_buffers(vb.command_pool, &[self.command_buffer]);
            for (buffer, mut allocation) in self.staging_buffers.drain(..) {
                allocator.destroy_buffer(buffer, &mut allocation);
            }
        }
        Ok(())
    }
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
            "🌱 Uploaded {} grass blades and {} flower reeds ({} grass triangles per blade, {}-byte instances)",
            chunked_grass.instances.len(),
            reed_instances.len(),
            near_blade_indices.len() / 3,
            size_of::<GrassInstance>(),
        );

        Ok(Self {
            near_blade_vertices: near_blade_vertex_buffer,
            near_blade_indices: near_blade_index_buffer,
            instances: instance_buffer,
            mid_blade_vertices: mid_blade_vertex_buffer,
            mid_blade_indices: mid_blade_index_buffer,
            reed_vertices: reed_vertex_buffer,
            reed_indices: reed_index_buffer,
            reed_instances: reed_instance_buffer,
            chunks: chunked_grass.chunks,
            visible_chunks: Vec::with_capacity(GRASS_CHUNKS_PER_SIDE * GRASS_CHUNKS_PER_SIDE),
            near_commands: Vec::with_capacity(GRASS_CHUNKS_PER_SIDE * GRASS_CHUNKS_PER_SIDE),
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
        let mid_push = grass_push_constants(camera, time_seconds, 1.0);
        let near_buffers = [self.near_blade_vertices.buffer, self.instances.buffer];
        let offsets = [0, 0];
        let descriptor_sets = [
            vb.set0_descriptor_sets[image_index],
            self.wind_map.descriptor_set,
        ];

        let (near_offset, mid_offset, reed_offset) =
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
    /// three LOD command lists without allocating new vectors.
    fn prepare_draw_commands(&mut self, camera: &crate::graphics::camera::Camera) {
        self.visible_chunks.clear();
        self.near_commands.clear();
        self.mid_commands.clear();
        self.reed_commands.clear();

        let view_projection = *camera.get_projection() * *camera.get_view();
        let camera_position = camera.position();
        for (index, chunk) in self.chunks.iter().enumerate() {
            let dx = chunk.centre[0] - camera_position.x;
            let dz = chunk.centre[1] - camera_position.z;
            let distance = ((dx * dx + dz * dz).sqrt() - chunk.radius).max(0.0);
            if distance <= MID_GRASS_DISTANCE && chunk_intersects_frustum(view_projection, chunk) {
                self.visible_chunks.push(VisibleChunk { index, distance });
            }
        }
        self.visible_chunks
            .sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));

        for visible in &self.visible_chunks {
            let chunk = &self.chunks[visible.index];
            if visible.distance <= NEAR_GRASS_DISTANCE && chunk.near_grass.count > 0 {
                self.near_commands.push(draw_command(
                    self.near_blade_indices.count,
                    chunk.near_grass,
                ));
            } else if chunk.mid_grass.count > 0 {
                self.mid_commands
                    .push(draw_command(self.mid_blade_indices.count, chunk.mid_grass));
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
    ) -> (vk::DeviceSize, vk::DeviceSize, vk::DeviceSize) {
        let indirect = &self.indirect_buffers[image_index];
        let stride = size_of::<vk::DrawIndexedIndirectCommand>();
        let near_first = 0;
        let mid_first = self.near_commands.len();
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
            (mid_first * stride) as u64,
            (reed_first * stride) as u64,
        )
    }

    /// Uses one multi-draw call when supported, retaining a portable per-command fallback.
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
        self.mid_blade_vertices.cleanup(allocator);
        self.mid_blade_indices.cleanup(allocator);
        self.reed_vertices.cleanup(allocator);
        self.reed_indices.cleanup(allocator);
        self.reed_instances.cleanup(allocator);
        cleanup_indirect_buffers(allocator, &mut self.indirect_buffers);
        self.wind_map.cleanup(device, allocator);
    }
}

/// Shader modules are owned here so they outlive every pipeline build/rebuild.
struct GrassShaders {
    vertex: ShaderStageInfo,
    fragment: ShaderStageInfo,
    mid_fragment: ShaderStageInfo,
}

impl GrassShaders {
    fn load(device: &ash::Device) -> Result<Self, Box<dyn Error>> {
        let entry = c"main";
        Ok(Self {
            vertex: ShaderStageInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                shader_module: ShaderModule::from_spv_file(
                    device,
                    "assets/shaders/spv/grass.vert.spv",
                )?,
                entry_name: entry,
            },
            fragment: ShaderStageInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                shader_module: ShaderModule::from_spv_file(
                    device,
                    "assets/shaders/spv/grass.frag.spv",
                )?,
                entry_name: entry,
            },
            mid_fragment: ShaderStageInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                shader_module: ShaderModule::from_spv_file(
                    device,
                    "assets/shaders/spv/grass_mid.frag.spv",
                )?,
                entry_name: entry,
            },
        })
    }
}

fn vertex_input_descriptions() -> (
    [vk::VertexInputBindingDescription; 2],
    [vk::VertexInputAttributeDescription; 5],
) {
    let bindings = [
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<GrassVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        },
        vk::VertexInputBindingDescription {
            binding: 1,
            stride: std::mem::size_of::<GrassInstance>() as u32,
            input_rate: vk::VertexInputRate::INSTANCE,
        },
    ];
    let attributes = [
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(GrassVertex, local_position) as u32,
        },
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 1,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(GrassVertex, local_normal) as u32,
        },
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 2,
            format: vk::Format::R32_SFLOAT,
            offset: offset_of!(GrassVertex, flower_head) as u32,
        },
        vk::VertexInputAttributeDescription {
            binding: 1,
            location: 3,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: offset_of!(GrassInstance, position_height) as u32,
        },
        vk::VertexInputAttributeDescription {
            binding: 1,
            location: 4,
            format: vk::Format::R16G16B16A16_UNORM,
            offset: offset_of!(GrassInstance, rotation_width_tint_phase) as u32,
        },
    ];
    (bindings, attributes)
}

/// Reuses the application's standard push-constant packing and fills the grass-only LOD slot.
fn grass_push_constants(
    camera: &crate::graphics::camera::Camera,
    time_seconds: f32,
    lod: f32,
) -> [u8; 144] {
    let mut bytes =
        compute_push_constant_per_obj(camera, &Matrix4::identity(), [time_seconds, 0.16]);
    bytes[136..140].copy_from_slice(&lod.to_ne_bytes());
    bytes
}

fn pack_instance_params(rotation: f32, width: f32, tint: f32, phase: f32) -> [u16; 4] {
    let to_unorm = |value: f32| (value.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16;
    [
        to_unorm(rotation / std::f32::consts::TAU),
        to_unorm(width / MAX_BLADE_WIDTH),
        to_unorm(tint),
        to_unorm(phase / std::f32::consts::TAU),
    ]
}

/// Builds one ribbon with the requested segment count.  The near LOD uses four segments;
/// the mid LOD uses one.  Instance rotation makes many ribbons read as volumetric grass.
fn build_blade_vertices(segments: usize) -> Vec<GrassVertex> {
    let mut vertices = Vec::with_capacity((segments + 1) * 2);

    for segment in 0..=segments {
        let height_fraction = segment as f32 / segments as f32;
        // Narrow toward the tip so the ribbon reads as a blade rather than a rectangle.
        let half_width = 0.5 * (1.0 - 0.82 * height_fraction);
        for side in [-1.0, 1.0] {
            vertices.push(GrassVertex {
                local_position: [half_width * side, height_fraction, 0.0],
                local_normal: [0.0, 0.0, 1.0],
                flower_head: 0.0,
            });
        }
    }
    vertices
}

fn build_blade_indices_for_segments(segments: usize) -> Vec<u32> {
    let mut indices = Vec::with_capacity(segments * 6);

    for segment in 0..segments {
        let lower_left = (segment * 2) as u32;
        let lower_right = lower_left + 1;
        let upper_left = lower_left + 2;
        let upper_right = lower_left + 3;
        indices.extend_from_slice(&[
            lower_left,
            lower_right,
            upper_right,
            upper_right,
            upper_left,
            lower_left,
        ]);
    }
    indices
}

/// Builds a six-sided stalk with a rounded capsule at the top.  The mesh is expressed in
/// unit height and width; each instance supplies its final height and radius.
fn build_reed_vertices() -> Vec<GrassVertex> {
    const SIDES: usize = 6;
    const HEAD_RINGS: &[(f32, f32, f32)] = &[
        // (height, radius, vertical normal component)
        (0.68, 0.18, -0.66),
        (0.73, 0.55, -0.31),
        (0.88, 0.55, 0.31),
        (0.94, 0.18, 0.66),
    ];
    let mut vertices = Vec::with_capacity(SIDES * (2 + HEAD_RINGS.len()) + 2);

    // The narrow green stalk uses two rings.  Its top overlaps the flower head slightly.
    for height in [0.0, 0.99] {
        for side in 0..SIDES {
            let angle = side as f32 / SIDES as f32 * std::f32::consts::TAU;
            vertices.push(GrassVertex {
                local_position: [angle.cos() * 0.16, height, angle.sin() * 0.16],
                local_normal: [angle.cos(), 0.0, angle.sin()],
                flower_head: 0.0,
            });
        }
    }

    // These rings approximate an elongated capsule, which keeps the flower deliberately
    // simple and stylised while still lighting like a small solid object.
    for &(height, radius, normal_y) in HEAD_RINGS {
        let horizontal_normal = (1.0 - normal_y * normal_y).sqrt();
        for side in 0..SIDES {
            let angle = side as f32 / SIDES as f32 * std::f32::consts::TAU;
            vertices.push(GrassVertex {
                local_position: [angle.cos() * radius, height, angle.sin() * radius],
                local_normal: [
                    angle.cos() * horizontal_normal,
                    normal_y,
                    angle.sin() * horizontal_normal,
                ],
                flower_head: 1.0,
            });
        }
    }

    vertices.push(GrassVertex {
        local_position: [0.0, 0.66, 0.0],
        local_normal: [0.0, -1.0, 0.0],
        flower_head: 1.0,
    });
    vertices.push(GrassVertex {
        // The cap sits just below the stalk tip, leaving a small green point visible.
        local_position: [0.0, 0.96, 0.0],
        local_normal: [0.0, 1.0, 0.0],
        flower_head: 1.0,
    });
    vertices
}

fn build_reed_indices() -> Vec<u32> {
    const SIDES: usize = 6;
    const RING_COUNT: usize = 6;
    let mut indices = Vec::with_capacity(4 * SIDES * 6 + SIDES * 6);

    // The capsule overlaps the stalk, so there is deliberately no surface joining their
    // rings.  That avoids a visible seam and keeps their distinct material flags separate.
    for (lower_ring, upper_ring) in [(0, 1), (2, 3), (3, 4), (4, 5)] {
        for side in 0..SIDES {
            let next_side = (side + 1) % SIDES;
            let lower_left = (lower_ring * SIDES + side) as u32;
            let lower_right = (lower_ring * SIDES + next_side) as u32;
            let upper_left = (upper_ring * SIDES + side) as u32;
            let upper_right = (upper_ring * SIDES + next_side) as u32;
            indices.extend_from_slice(&[
                lower_left,
                upper_right,
                lower_right,
                upper_right,
                lower_left,
                upper_left,
            ]);
        }
    }

    let lower_cap = (RING_COUNT * SIDES) as u32;
    let upper_cap = lower_cap + 1;
    let flower_first_ring = 2 * SIDES;
    let flower_last_ring = (RING_COUNT - 1) * SIDES;
    for side in 0..SIDES {
        let next_side = (side + 1) % SIDES;
        indices.extend_from_slice(&[
            lower_cap,
            (flower_first_ring + side) as u32,
            (flower_first_ring + next_side) as u32,
            upper_cap,
            (flower_last_ring + next_side) as u32,
            (flower_last_ring + side) as u32,
        ]);
    }
    indices
}

/// Builds regular-but-jittered grass in contiguous 5×5 m chunks.  Keeping each chunk's
/// instances together lets `first_instance` select it without copying GPU data per frame.
fn build_chunked_grass_instances(terrain: TerrainSettings) -> ChunkedGrassInstances {
    let spacing = GRASS_FIELD_SIZE / GRASS_GRID_SIDE as f32;
    let half_size = GRASS_FIELD_SIZE * 0.5;
    // The central field keeps every candidate while the much larger outer area is
    // thinned, so reserving the complete candidate count would waste considerable RAM.
    let candidate_count = GRASS_GRID_SIDE * GRASS_GRID_SIDE * GRASS_DENSITY_LAYERS;
    let mut instances = Vec::with_capacity(candidate_count * 3 / 4);
    let mut chunks = Vec::with_capacity(GRASS_CHUNKS_PER_SIDE * GRASS_CHUNKS_PER_SIDE);
    let cells_per_chunk = GRASS_GRID_SIDE / GRASS_CHUNKS_PER_SIDE;
    let chunk_radius = GRASS_CHUNK_SIZE * std::f32::consts::FRAC_1_SQRT_2 + 0.8;

    for chunk_row in 0..GRASS_CHUNKS_PER_SIDE {
        for chunk_column in 0..GRASS_CHUNKS_PER_SIDE {
            let first_instance = instances.len() as u32;
            let first_row = chunk_row * cells_per_chunk;
            let first_column = chunk_column * cells_per_chunk;
            let mut mid_subset = Vec::with_capacity(cells_per_chunk * cells_per_chunk / 2);
            let mut remaining = Vec::with_capacity(cells_per_chunk * cells_per_chunk * 2);

            for row in first_row..first_row + cells_per_chunk {
                for column in first_column..first_column + cells_per_chunk {
                    for layer in 0..GRASS_DENSITY_LAYERS {
                        let id = ((row * GRASS_GRID_SIDE + column) * GRASS_DENSITY_LAYERS + layer)
                            as u32;
                        // Two diagonally opposed offsets turn each former placement cell into a
                        // small staggered pair instead of placing two blades at the same root.
                        let (cell_offset_x, cell_offset_z) = if layer == 0 {
                            (0.25, 0.75)
                        } else {
                            (0.75, 0.25)
                        };
                        let jitter_x = (hash01(id, 0) - 0.5) * spacing * 0.45;
                        let jitter_z = (hash01(id, 1) - 0.5) * spacing * 0.45;
                        let x = -half_size + (column as f32 + cell_offset_x) * spacing + jitter_x;
                        let z =
                            -2.0 - half_size + (row as f32 + cell_offset_z) * spacing + jitter_z;

                        // Distant blades occupy fewer candidate roots.  The decision is
                        // stable for each blade, so no grass pops in or moves between frames.
                        if hash01(id, 31) > grass_density_at(x, z) {
                            continue;
                        }

                        if is_clearing(x, z) {
                            continue;
                        }

                        let instance = GrassInstance {
                            position_height: [
                                x,
                                terrain.height_at(x, z) + 0.003,
                                z,
                                0.17 + hash01(id, 2) * 0.21,
                            ],
                            rotation_width_tint_phase: pack_instance_params(
                                hash01(id, 3) * std::f32::consts::TAU,
                                0.025 + hash01(id, 4) * 0.03,
                                hash01(id, 5),
                                hash01(id, 6) * std::f32::consts::TAU,
                            ),
                        };
                        // The mid LOD uses a stable random half, avoiding visible rows. Half
                        // density preserves the field's colour while its two-triangle ribbon
                        // remains much cheaper than the near blade's eight triangles.
                        if hash01(id, 30) < 0.50 {
                            mid_subset.push(instance);
                        } else {
                            remaining.push(instance);
                        }
                    }
                }
            }

            // Put the mid subset first. Its range is therefore also the beginning of the
            // full near range, allowing both LODs to share one GPU instance buffer.
            let mid_count = mid_subset.len() as u32;
            let near_count = (mid_subset.len() + remaining.len()) as u32;
            instances.extend(mid_subset);
            instances.extend(remaining);

            chunks.push(VegetationChunk {
                centre: [
                    -half_size + (chunk_column as f32 + 0.5) * GRASS_CHUNK_SIZE,
                    -2.0 - half_size + (chunk_row as f32 + 0.5) * GRASS_CHUNK_SIZE,
                ],
                radius: chunk_radius,
                near_grass: InstanceRange {
                    first: first_instance,
                    count: near_count,
                },
                mid_grass: InstanceRange {
                    first: first_instance,
                    count: mid_count,
                },
                reeds: InstanceRange::default(),
            });
        }
    }

    ChunkedGrassInstances { instances, chunks }
}

/// Returns the fraction of candidate grass roots retained at a world position.
fn grass_density_at(x: f32, z: f32) -> f32 {
    // The field is centred two metres behind the origin to match its existing placement.
    // A square radius makes density reach the same value along all four field boundaries.
    let field_radius = x.abs().max((z + 2.0).abs());
    let half_size = GRASS_FIELD_SIZE * 0.5;
    let amount = ((field_radius - FULL_DENSITY_RADIUS)
        / (half_size - FULL_DENSITY_RADIUS))
        .clamp(0.0, 1.0);
    let smooth_amount = amount * amount * (3.0 - 2.0 * amount);
    1.0 - (1.0 - OUTER_DENSITY) * smooth_amount
}

/// Places a few clusters of simple flower reeds.  Every stalk in a cluster shares its
/// flower colour, making it read as one small plant rather than nine random tall blades.
fn build_reed_instances(
    terrain: TerrainSettings,
    chunks: &mut [VegetationChunk],
) -> Vec<GrassInstance> {
    let half_size = GRASS_FIELD_SIZE * 0.5;
    let mut per_chunk = vec![Vec::new(); chunks.len()];

    for tuft_index in 0..TALL_TUFT_COUNT {
        let tuft_id = tuft_index as u32;
        let centre_x = -half_size + hash01(tuft_id, 20) * GRASS_FIELD_SIZE;
        let centre_z = -2.0 - half_size + hash01(tuft_id, 21) * GRASS_FIELD_SIZE;

        if is_clearing(centre_x, centre_z) {
            continue;
        }

        for blade_index in 0..TALL_BLADES_PER_TUFT {
            let blade_id = tuft_id * TALL_BLADES_PER_TUFT as u32 + blade_index as u32;
            // A square-root radius keeps the cluster evenly filled instead of concentrating
            // blades at its centre.  Rotation then makes each tuft read as a small plant.
            let angle = hash01(blade_id, 22) * std::f32::consts::TAU;
            let radius = hash01(blade_id, 23).sqrt() * 0.28;
            let x = centre_x + angle.cos() * radius;
            let z = centre_z + angle.sin() * radius;

            per_chunk[chunk_index_for(centre_x, centre_z)].push(GrassInstance {
                position_height: [
                    x,
                    terrain.height_at(x, z) + 0.004,
                    z,
                    0.62 + hash01(blade_id, 24) * 0.18,
                ],
                rotation_width_tint_phase: pack_instance_params(
                    hash01(blade_id, 25) * std::f32::consts::TAU,
                    0.070 + hash01(blade_id, 26) * 0.018,
                    // Most clusters are cream-white; a few are a muted red variant.
                    if hash01(tuft_id, 27) < 0.22 { 0.0 } else { 1.0 },
                    hash01(blade_id, 28) * std::f32::consts::TAU,
                ),
            });
        }
    }

    let mut instances = Vec::with_capacity(TALL_TUFT_COUNT * TALL_BLADES_PER_TUFT);
    for (chunk, reed_instances) in chunks.iter_mut().zip(per_chunk) {
        chunk.reeds = InstanceRange {
            first: instances.len() as u32,
            count: reed_instances.len() as u32,
        };
        instances.extend(reed_instances);
    }
    instances
}

/// Returns the chunk containing a field-space root position.  Reed clusters use their
/// centre so all nine stalks remain in the same culling/LOD decision.
fn chunk_index_for(x: f32, z: f32) -> usize {
    let half_size = GRASS_FIELD_SIZE * 0.5;
    let column = ((x + half_size) / GRASS_CHUNK_SIZE)
        .floor()
        .clamp(0.0, (GRASS_CHUNKS_PER_SIDE - 1) as f32) as usize;
    let row = ((z + 2.0 + half_size) / GRASS_CHUNK_SIZE)
        .floor()
        .clamp(0.0, (GRASS_CHUNKS_PER_SIDE - 1) as f32) as usize;
    row * GRASS_CHUNKS_PER_SIDE + column
}

/// Conservative 3D frustum test using the eight corners of a padded chunk box.
///
/// Testing homogeneous clip coordinates avoids error-prone plane extraction. A chunk is
/// rejected only when all eight corners lie outside the same clipping plane.
fn chunk_intersects_frustum(view_projection: Matrix4<f32>, chunk: &VegetationChunk) -> bool {
    let half_width = GRASS_CHUNK_SIZE * 0.5 + 0.3;
    let mut corners = [Vector4::new(0.0, 0.0, 0.0, 1.0); 8];
    let mut index = 0;
    for y in [-0.05, 1.05] {
        for x in [chunk.centre[0] - half_width, chunk.centre[0] + half_width] {
            for z in [chunk.centre[1] - half_width, chunk.centre[1] + half_width] {
                corners[index] = view_projection * Vector4::new(x, y, z, 1.0);
                index += 1;
            }
        }
    }

    !corners.iter().all(|p| p.x < -p.w)
        && !corners.iter().all(|p| p.x > p.w)
        && !corners.iter().all(|p| p.y < -p.w)
        && !corners.iter().all(|p| p.y > p.w)
        && !corners.iter().all(|p| p.z < 0.0)
        && !corners.iter().all(|p| p.z > p.w)
}

fn draw_command(index_count: u32, range: InstanceRange) -> vk::DrawIndexedIndirectCommand {
    vk::DrawIndexedIndirectCommand {
        index_count,
        instance_count: range.count,
        first_index: 0,
        vertex_offset: 0,
        first_instance: range.first,
    }
}

fn create_indirect_buffers(
    vb: &VulkanBase,
    count: usize,
) -> Result<Vec<IndirectBuffer>, vk::Result> {
    let allocator = vb.allocator.as_ref().expect("allocator");
    let capacity = GRASS_CHUNKS_PER_SIDE * GRASS_CHUNKS_PER_SIDE * 3;
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

impl WindMap {
    fn cleanup(&mut self, device: &ash::Device, allocator: &Allocator) {
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            allocator.destroy_image(self.image, &mut self.allocation);
        }
    }
}

fn create_wind_map_resources(
    device: &ash::Device,
    image: vk::Image,
    allocation: Allocation,
) -> Result<WindMap, vk::Result> {
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let view_info = vk::ImageViewCreateInfo {
        image,
        view_type: vk::ImageViewType::TYPE_2D,
        format: vk::Format::R8G8B8A8_UNORM,
        subresource_range: range,
        ..Default::default()
    };
    let view = unsafe { device.create_image_view(&view_info, None)? };
    let sampler_info = vk::SamplerCreateInfo {
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        address_mode_u: vk::SamplerAddressMode::REPEAT,
        address_mode_v: vk::SamplerAddressMode::REPEAT,
        address_mode_w: vk::SamplerAddressMode::REPEAT,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        max_lod: 0.0,
        ..Default::default()
    };
    let sampler = unsafe { device.create_sampler(&sampler_info, None)? };
    let binding = vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::VERTEX,
        ..Default::default()
    };
    let layout_info = vk::DescriptorSetLayoutCreateInfo {
        binding_count: 1,
        p_bindings: &binding,
        ..Default::default()
    };
    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };
    let pool_size = vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
    };
    let pool_info = vk::DescriptorPoolCreateInfo {
        max_sets: 1,
        pool_size_count: 1,
        p_pool_sizes: &pool_size,
        ..Default::default()
    };
    let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
    let allocate_info = vk::DescriptorSetAllocateInfo {
        descriptor_pool,
        descriptor_set_count: 1,
        p_set_layouts: &descriptor_set_layout,
        ..Default::default()
    };
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&allocate_info)?[0] };
    let image_info = vk::DescriptorImageInfo {
        sampler,
        image_view: view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };
    let write = vk::WriteDescriptorSet {
        dst_set: descriptor_set,
        dst_binding: 0,
        descriptor_count: 1,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        p_image_info: &image_info,
        ..Default::default()
    };
    unsafe { device.update_descriptor_sets(&[write], &[]) };
    Ok(WindMap {
        image,
        allocation,
        view,
        sampler,
        descriptor_pool,
        descriptor_set_layout,
        descriptor_set,
    })
}

/// Generates two seamless value-noise channels: broad gusts in red and fine motion in green.
fn build_wind_map_pixels() -> Vec<u8> {
    let mut pixels = Vec::with_capacity((WIND_MAP_SIZE * WIND_MAP_SIZE * 4) as usize);
    for y in 0..WIND_MAP_SIZE {
        for x in 0..WIND_MAP_SIZE {
            let broad = tileable_value_noise(x, y, 8, 41);
            let detail = tileable_value_noise(x, y, 23, 97);
            pixels.extend_from_slice(&[
                (broad * 255.0).round() as u8,
                (detail * 255.0).round() as u8,
                0,
                255,
            ]);
        }
    }
    pixels
}

fn tileable_value_noise(x: u32, y: u32, cells: u32, seed: u32) -> f32 {
    let grid_x = x as f32 / WIND_MAP_SIZE as f32 * cells as f32;
    let grid_y = y as f32 / WIND_MAP_SIZE as f32 * cells as f32;
    let cell_x = grid_x.floor() as u32;
    let cell_y = grid_y.floor() as u32;
    let mut fraction_x = grid_x.fract();
    let mut fraction_y = grid_y.fract();
    fraction_x = fraction_x * fraction_x * (3.0 - 2.0 * fraction_x);
    fraction_y = fraction_y * fraction_y * (3.0 - 2.0 * fraction_y);
    let sample = |offset_x: u32, offset_y: u32| {
        let wrapped_x = (cell_x + offset_x) % cells;
        let wrapped_y = (cell_y + offset_y) % cells;
        hash01(wrapped_y * cells + wrapped_x, seed)
    };
    let bottom = sample(0, 0) + (sample(1, 0) - sample(0, 0)) * fraction_x;
    let top = sample(0, 1) + (sample(1, 1) - sample(0, 1)) * fraction_x;
    bottom + (top - bottom) * fraction_y
}

fn is_clearing(x: f32, z: f32) -> bool {
    // Hut, cube, smaller cube, and sphere.  These are temporary scene-level placement
    // hints; a future placement system can provide these radii instead.
    const CLEARINGS: &[(f32, f32, f32)] = &[
        (0.0, -2.0, 3.0),
        (7.0, 0.0, 1.5),
        (6.0, -6.0, 1.3),
        (-6.0, 0.0, 1.4),
    ];
    CLEARINGS.iter().any(|&(centre_x, centre_z, radius)| {
        let dx = x - centre_x;
        let dz = z - centre_z;
        dx * dx + dz * dz < radius * radius
    })
}

/// Deterministic pseudo-random value in `0.0..=1.0`, without storing a CPU RNG.
fn hash01(id: u32, stream: u32) -> f32 {
    let mut value = id.wrapping_mul(0x9E37_79B9) ^ stream.wrapping_mul(0x85EB_CA6B);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^= value >> 16;
    value as f32 / u32::MAX as f32
}

#[cfg(test)]
mod tests {
    use super::{
        GRASS_CHUNKS_PER_SIDE, GRASS_DENSITY_LAYERS, GRASS_GRID_SIDE, TALL_BLADES_PER_TUFT,
        build_blade_indices_for_segments, build_blade_vertices, build_chunked_grass_instances,
        build_reed_indices, build_reed_instances, build_reed_vertices, build_wind_map_pixels,
        chunk_intersects_frustum, grass_density_at,
    };
    use crate::graphics::terrain::TerrainSettings;

    #[test]
    fn grass_uses_one_shared_blade_and_many_grounded_instances() {
        let vertices = build_blade_vertices(4);
        let indices = build_blade_indices_for_segments(4);
        let chunked = build_chunked_grass_instances(TerrainSettings::default());

        assert_eq!(vertices.len(), 10);
        assert_eq!(indices.len(), 24);
        let candidate_count = GRASS_GRID_SIDE * GRASS_GRID_SIDE * GRASS_DENSITY_LAYERS;
        assert!(chunked.instances.len() > candidate_count / 5);
        assert!(chunked.instances.len() < candidate_count * 4 / 5);
        assert_eq!(
            chunked.chunks.len(),
            GRASS_CHUNKS_PER_SIDE * GRASS_CHUNKS_PER_SIDE
        );
        assert!(
            chunked
                .instances
                .iter()
                .all(|instance| instance.position_height[3] > 0.0)
        );
        assert!(
            chunked
                .instances
                .iter()
                .any(|instance| instance.position_height[3] < 0.25)
        );
        assert!(
            chunked
                .instances
                .iter()
                .any(|instance| instance.position_height[3] > 0.35)
        );
        assert!(chunked
            .instances
            .iter()
            .any(|instance| instance.position_height[0].abs() > 70.0));
        assert_eq!(grass_density_at(0.0, -2.0), 1.0);
        assert_eq!(grass_density_at(80.0, -2.0), super::OUTER_DENSITY);
        let mid_count: usize = chunked
            .chunks
            .iter()
            .map(|chunk| chunk.mid_grass.count as usize)
            .sum();
        assert!(mid_count > chunked.instances.len() * 2 / 5);
        assert!(mid_count < chunked.instances.len() * 3 / 5);
    }

    #[test]
    fn reeds_use_a_separate_capsule_mesh_and_stay_clustered() {
        let vertices = build_reed_vertices();
        let indices = build_reed_indices();
        let mut chunked = build_chunked_grass_instances(TerrainSettings::default());
        let instances = build_reed_instances(TerrainSettings::default(), &mut chunked.chunks);

        assert!(vertices.iter().any(|vertex| vertex.flower_head > 0.5));
        assert!(indices.len() > 100);
        assert_eq!(indices.len() % 3, 0, "every reed index group is a triangle");
        // Every side triangle must face the same way as its stored outward normals.  This
        // catches an easy-to-miss error where only one triangle of every quad is reversed.
        for (triangle_index, triangle) in indices[..4 * 6 * 6].chunks_exact(3).enumerate() {
            let a = vertices[triangle[0] as usize];
            let b = vertices[triangle[1] as usize];
            let c = vertices[triangle[2] as usize];
            let ab = [
                b.local_position[0] - a.local_position[0],
                b.local_position[1] - a.local_position[1],
                b.local_position[2] - a.local_position[2],
            ];
            let ac = [
                c.local_position[0] - a.local_position[0],
                c.local_position[1] - a.local_position[1],
                c.local_position[2] - a.local_position[2],
            ];
            let face_normal = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let average_normal = [
                a.local_normal[0] + b.local_normal[0] + c.local_normal[0],
                a.local_normal[1] + b.local_normal[1] + c.local_normal[1],
                a.local_normal[2] + b.local_normal[2] + c.local_normal[2],
            ];
            let winding_matches_normals = face_normal[0] * average_normal[0]
                + face_normal[1] * average_normal[1]
                + face_normal[2] * average_normal[2];
            assert!(
                winding_matches_normals > 0.0,
                "side triangle {triangle_index} is inward"
            );
        }
        assert!(instances.len() >= TALL_BLADES_PER_TUFT);
        assert!(
            instances
                .iter()
                .all(|instance| instance.position_height[3] >= 0.62)
        );
        assert_eq!(
            chunked
                .chunks
                .iter()
                .map(|chunk| chunk.reeds.count as usize)
                .sum::<usize>(),
            instances.len(),
        );
    }

    #[test]
    fn grass_chunks_cull_side_and_distant_field_sections() {
        let chunked = build_chunked_grass_instances(TerrainSettings::default());
        let camera = crate::graphics::camera::Camera::new();
        let view_projection = *camera.get_projection() * *camera.get_view();
        let visible_chunks = chunked
            .chunks
            .iter()
            .filter(|chunk| chunk_intersects_frustum(view_projection, chunk))
            .count();

        assert!(visible_chunks > 0);
        assert!(visible_chunks < chunked.chunks.len());
    }

    #[test]
    fn generated_wind_map_has_two_varying_channels() {
        let pixels = build_wind_map_pixels();
        assert_eq!(
            pixels.len(),
            (super::WIND_MAP_SIZE * super::WIND_MAP_SIZE * 4) as usize
        );
        let first_red = pixels[0];
        let first_green = pixels[1];
        assert!(pixels.chunks_exact(4).any(|pixel| pixel[0] != first_red));
        assert!(pixels.chunks_exact(4).any(|pixel| pixel[1] != first_green));
    }
}
