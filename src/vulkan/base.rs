//! --------------------------------------------------------------------------------------
//! VulkanBase Module (base.rs)
//!
//! Created: July 2025  
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module defines `VulkanBase`, the one–stop container for all Vulkan state and
//! resources in our application. It handles:
//!   • Instance creation and destruction  
//!   • Physical device selection and logical device + graphics queue setup  
//!   • Surface creation via winit + ash_window  
//!   • Swapchain management delegated to `Swapchain`  
//!   • Graphics pipeline setup delegated to `Pipeline`  
//!   • Command pool and command buffer allocation & recording  
//!   • Synchronization primitives (semaphores & fence)  
//!   
//! **Manual Teardown:**  
//! Unlike idiomatic Rust RAII, we implement a single `Drop` on `VulkanBase` that
//! explicitly destroys every Vulkan object in the correct order—waiting for the GPU
//! to idle, then tearing down swapchain, pipeline, sync, command pool, surface, device,
//! and finally the instance. This guarantees we never call Vulkan destroy-functions
//! on a device or loader that’s already been dropped.
//!
//! **Usage:**  
//! Create a `VulkanBase` via `VulkanBase::new(window, event_loop)`, call
//! `draw_frame()` in your render loop, and rely on `Drop` to clean up on exit.
//!
//! --------------------------------------------------------------------------------------

/// Number of CPU frames-in-flight (slots) for synchronization.
const INFLIGHT_FRAMES: usize = 2;
/// Frame start/end plus the boundaries needed to isolate shadow, scene, and vegetation work.
const TIMESTAMPS_PER_IMAGE: u32 = 5;

use ash::khr::surface;
use ash::khr::swapchain;
use ash::vk;
use ash::{Entry, Instance};
use ash_window::create_surface;
use ash_window::enumerate_required_extensions;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::error::Error;
use std::ffi::CStr;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use super::attachments::{AttachmentHandle, AttachmentManager, AttachmentRequest};
use super::swapchain::Swapchain;
use vk_mem::{Alloc, Allocation, Allocator, MemoryUsage};

mod setup;

/// GPU durations measured for the most recently completed frame.
///
/// Timestamp queries are asynchronous: these values describe the previous completed frame,
/// which keeps the profiler from stalling the CPU waiting for the GPU.  `scene_ms` is the
/// main scene after shadows and before vegetation; `total_ms` also includes the UI pass.
#[derive(Clone, Copy, Default)]
pub struct GpuPassTimings {
    pub total_ms: Option<f32>,
    pub shadow_ms: Option<f32>,
    pub scene_ms: Option<f32>,
    pub vegetation_ms: Option<f32>,
}

#[repr(u32)]
/// Query positions within each swapchain image's five-query range.  Keep this order in sync
/// with the calculations in `begin_frame`.
enum GpuTimestamp {
    FrameStart = 0,
    ShadowEnd = 1,
    VegetationStart = 2,
    VegetationEnd = 3,
    FrameEnd = 4,
}

/// The debug callback function that prints validation layer messages.
#[cfg(debug_assertions)]
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    unsafe {
        let message = CStr::from_ptr((*p_callback_data).p_message);
        let severity = match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "📢 [VERBOSE]",
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "ℹ️ [INFO]",
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "⚠️ [WARNING]",
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "❌ [ERROR]",
            _ => "[UNKNOWN SEVERITY]",
        };
        let ty = match message_type {
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[GENERAL]",
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[VALIDATION]",
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[PERFORMANCE]",
            _ => "[UNKNOWN TYPE]",
        };
        eprintln!("{} {} {:?}", severity, ty, message);
    }
    vk::FALSE
}

/// Represents the Vulkan backend, encapsulating all Vulkan-related state and operations
/// and acts as the parent for the swapchain and rendering pipeline.
///
/// It handles initialization, resource management, and rendering logic.
/// This includes creating the Vulkan instance, physical device selection,
/// logical device creation, command pools, command buffers, synchronization objects,
/// and hands off swapchain management to the `Swapchain` struct.
pub struct VulkanBase {
    /// Owning Vulkan instance (created in `new`, destroyed in `Drop`).
    pub instance: Instance,
    /// Chosen physical device handle.
    pub physical_device: vk::PhysicalDevice,
    /// Logical device used for all Vulkan calls.
    pub device: ash::Device,
    /// Reused while creating graphics pipelines to avoid recompiling shared
    /// pipeline state during runtime rebuilds.
    pub pipeline_cache: vk::PipelineCache,
    /// Global Vulkan memory allocator (VMA).
    pub allocator: Option<vk_mem::Allocator>,
    /// Graphics queue from the selected family.
    pub graphics_queue: vk::Queue,
    // -- Global set 0 related --
    /// Descriptor set layout for the global UBO (set = 0, binding = 0).
    pub set0_global_layout: vk::DescriptorSetLayout,
    set0_descriptor_pool: vk::DescriptorPool,
    /// One descriptor set per swapchain image for set 0.
    pub set0_descriptor_sets: Vec<vk::DescriptorSet>,
    ubo_buffers: Vec<vk::Buffer>,
    /// Allocations backing each per-image UBO buffer.
    pub ubo_allocations: Vec<Allocation>,
    /// Sampler for the directional shadow map (set=0, binding=1)
    pub shadow_sampler: vk::Sampler,
    /// Last shadow view written to each per-image global descriptor set.
    shadow_descriptor_views: Vec<vk::ImageView>,
    // --- Timing (GPU timestamp queries) ---
    timestamp_query_pool: vk::QueryPool,
    timestamp_period_ns: f32,
    /// Timings collected from the last completed GPU frame.
    last_gpu_timings: GpuPassTimings,
    /// For each CPU frame-in-flight slot, which swapchain image index it last submitted
    last_image_per_slot: Vec<Option<u32>>,
    // -- Synchronization objects --
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    /// Fences tracking which frame-in-flight slot is using a GPU submission.
    in_flight_fences: Vec<vk::Fence>,
    /// For each swapchain image, tracks which fence (frame slot) currently owns it.
    image_owner_fence: Vec<vk::Fence>,
    // -- Swapchain objects --
    /// Wrapper containing the swapchain and related image resources.
    pub swapchain: Swapchain,
    swapchain_loader: swapchain::Device,
    surface: vk::SurfaceKHR,
    surface_loader: surface::Instance,
    // -- Command objects --
    /// Primary command pool for the graphics queue family.
    pub command_pool: vk::CommandPool,
    /// One primary command buffer per swapchain image.
    pub command_buffers: Vec<vk::CommandBuffer>,
    // -- Misc --
    /// Runtime toggles (wireframe, ms/frame overlay).
    pub engine_settings: EngineSettings,
    /// Handles offscreen and transient attachments per swapchain image.
    pub attachment_manager: AttachmentManager,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,
    #[cfg(debug_assertions)]
    debug_utils_loader: ash::ext::debug_utils::Instance,
    frame_slot: usize,
    current_image_index: usize,
    /// Generation counter for pipeline-dependent changes (swapchain, MSAA, wireframe, etc)
    pipeline_generation: u64,
    pub sample_count_flags_supported: vk::SampleCountFlags,
    /// Whether one indirect command can execute several indexed draws.
    pub supports_multi_draw_indirect: bool,
    /// Device limit used to split unusually large indirect command lists safely.
    pub max_draw_indirect_count: u32,
    /// Tracks a pending MSAA sample count change requested by the UI.
    pending_msaa_samples: Option<u32>,
    /// Set when acquire or presentation reports an out-of-date/suboptimal
    /// swapchain.  The render graph recreates it at the next safe boundary.
    swapchain_recreation_needed: bool,
}

/// Frame context returned by `begin_frame` and consumed by `end_frame`.
/// Holds which swapchain image we're drawing to and the command buffer to record into.
pub struct FrameCtx {
    /// Command buffer for this swapchain image.
    pub cmd_buf: vk::CommandBuffer,
    /// Index of the acquired swapchain image.
    pub image_index: u32,
    // CPU "frame-in-flight" slot that owns the fence/semaphore for this submission.
    frame_slot: usize,
}

/// Describes a transition for an image attachment between two usages.
#[derive(Clone, Copy, Debug)]
pub struct ImageTransition {
    pub image: vk::Image,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub src_access_mask: vk::AccessFlags,
    pub dst_access_mask: vk::AccessFlags,
    pub src_stage_mask: vk::PipelineStageFlags,
    pub dst_stage_mask: vk::PipelineStageFlags,
    pub aspect_mask: vk::ImageAspectFlags,
}

impl VulkanBase {
    /// Begin a frame: wait/reset fences, acquire the next image, and begin the command buffer & render pass.
    ///
    /// Returns `Ok(None)` when the swapchain is out-of-date so the caller can skip this frame.
    ///
    /// # Steps
    /// 1. Wait for the previous GPU work that used this CPU frame slot.
    /// 2. Acquire the next swapchain image (signals the per-slot image-available semaphore).
    /// 3. If that image is still owned by another slot, wait on that image's owner fence.
    /// 4. Reset this slot's fence and mark it as the owner of the acquired image.
    /// 5. Begin the command buffer and open the render pass so the caller can record draws.
    pub fn begin_frame(&mut self) -> Result<Option<FrameCtx>, Box<dyn Error>> {
        unsafe {
            let slot = self.frame_slot;

            // Wait for the previous work submitted on this CPU slot.
            self.device
                .wait_for_fences(&[self.in_flight_fences[slot]], true, u64::MAX)?;

            // After the fence is signaled, GPU work for that slot is complete.
            // If we have a previous image index for this slot, read back its timestamps.
            if let Some(prev_image_index) = self.last_image_per_slot[slot] {
                let base = self.image_query_base(prev_image_index);
                let mut data = [0_u64; TIMESTAMPS_PER_IMAGE as usize];
                let res = self.device.get_query_pool_results(
                    self.timestamp_query_pool,
                    base,
                    &mut data,
                    vk::QueryResultFlags::TYPE_64,
                );
                if res.is_ok() {
                    self.last_gpu_timings = GpuPassTimings {
                        total_ms: self.timestamp_delta_ms(data[0], data[4]),
                        shadow_ms: self.timestamp_delta_ms(data[0], data[1]),
                        scene_ms: self.timestamp_delta_ms(data[1], data[2]),
                        vegetation_ms: self.timestamp_delta_ms(data[2], data[3]),
                    };
                }
                // Clear the record; we'll set it again on submit
                self.last_image_per_slot[slot] = None;
            }

            // Acquire an image; signal when it's ready via the per-slot image-available semaphore.
            let (image_index, is_suboptimal) = match self.swapchain_loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                self.image_available_semaphores[slot],
                vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.swapchain_recreation_needed = true;
                    return Ok(None);
                }
                Err(e) => return Err(e.into()),
            };
            self.swapchain_recreation_needed |= is_suboptimal;

            let idx = image_index as usize;
            self.current_image_index = idx;

            // If this image is still tied to an older in-flight slot, wait for that slot to finish first.
            if self.image_owner_fence[idx] != vk::Fence::null() {
                self.device
                    .wait_for_fences(&[self.image_owner_fence[idx]], true, u64::MAX)?;
            }

            // Reuse this slot's fence for the new submit and associate it with this image.
            self.device.reset_fences(&[self.in_flight_fences[slot]])?;
            self.image_owner_fence[idx] = self.in_flight_fences[slot];

            // Begin recording; passes are responsible for starting rendering.
            let cmd_buf = self.command_buffers[idx];

            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device.begin_command_buffer(cmd_buf, &begin_info)?;

            Ok(Some(FrameCtx {
                cmd_buf,
                image_index,
                frame_slot: slot,
            }))
        }
    }

    /// Insert image memory barriers for a set of attachment transitions.
    ///
    /// Each [`ImageTransition`] describes how a single image's layout and
    /// access masks change. The command buffer must be in the recording state
    /// and will receive a single `vkCmdPipelineBarrier` covering all
    /// transitions.
    pub fn insert_attachment_barriers(
        &self,
        cmd: vk::CommandBuffer,
        transitions: &[ImageTransition],
    ) {
        if transitions.is_empty() {
            return;
        }
        let mut barriers: Vec<vk::ImageMemoryBarrier> = Vec::new();
        let mut src_stage = vk::PipelineStageFlags::empty();
        let mut dst_stage = vk::PipelineStageFlags::empty();
        for t in transitions {
            barriers.push(vk::ImageMemoryBarrier {
                src_access_mask: t.src_access_mask,
                dst_access_mask: t.dst_access_mask,
                old_layout: t.old_layout,
                new_layout: t.new_layout,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: t.image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: t.aspect_mask,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            });
            src_stage |= t.src_stage_mask;
            dst_stage |= t.dst_stage_mask;
        }
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }
    }

    /// End a frame: finish the render pass, submit, present, and advance the slot.
    ///
    /// # Steps
    /// 1. End the render pass and command buffer.
    /// 2. Submit the command buffer: wait on the per-slot image-available semaphore and signal the per-image render-finished semaphore.
    /// 3. Present the image, waiting on the render-finished semaphore for this image.
    /// 4. Advance to the next CPU frame-in-flight slot.
    pub fn end_frame(&mut self, frame: FrameCtx) -> Result<(), Box<dyn Error>> {
        unsafe {
            // Command buffer was already closed by the passes.
            self.device.end_command_buffer(frame.cmd_buf)?;

            // Submit: wait for image-available (slot), signal render-finished (per-image)
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let wait_sems = [self.image_available_semaphores[frame.frame_slot]];
            let signal_sems = [self.render_finished_semaphores[frame.image_index as usize]];
            let submit_info = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: wait_sems.as_ptr(),
                p_wait_dst_stage_mask: wait_stages.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: &frame.cmd_buf,
                signal_semaphore_count: 1,
                p_signal_semaphores: signal_sems.as_ptr(),
                ..Default::default()
            };
            self.device.queue_submit(
                self.graphics_queue,
                &[submit_info],
                self.in_flight_fences[frame.frame_slot],
            )?;

            // Present the image, waiting on the render-finished semaphore for this image
            let present_wait = [self.render_finished_semaphores[frame.image_index as usize]];
            let image_index = frame.image_index; // keep a local so we can take a stable pointer
            let present_info = vk::PresentInfoKHR {
                wait_semaphore_count: 1,
                p_wait_semaphores: present_wait.as_ptr(),
                swapchain_count: 1,
                p_swapchains: &self.swapchain.handle,
                p_image_indices: &image_index,
                ..Default::default()
            };
            match self
                .swapchain_loader
                .queue_present(self.graphics_queue, &present_info)
            {
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.swapchain_recreation_needed = true;
                }
                Err(e) => return Err(e.into()),
                _ => {}
            }

            // Advance to next CPU slot
            self.frame_slot = (frame.frame_slot + 1) % self.image_available_semaphores.len();

            // Record which image this slot just submitted, so on next begin_frame we can read timestamps
            if self.last_image_per_slot.len() != self.image_available_semaphores.len() {
                self.last_image_per_slot = vec![None; self.image_available_semaphores.len()];
            }
            self.last_image_per_slot[frame.frame_slot] = Some(frame.image_index);
        }
        Ok(())
    }

    /// Retrieve or create an attachment for the current swapchain image.
    pub fn get_attachment(&mut self, request: AttachmentRequest) -> AttachmentHandle {
        let allocator = self.allocator.as_ref().expect("allocator");
        self.attachment_manager.get_attachment(
            &self.device,
            allocator,
            self.current_image_index,
            request,
        )
    }

    /// Starts the frame's timestamp range and clears its per-pass boundaries.
    ///
    /// Queries belong to the acquired swapchain image, so an image is never reset while the
    /// fence protecting that image still has GPU work in flight.
    pub fn begin_gpu_timing(&self, cmd: vk::CommandBuffer, image_index: u32) {
        unsafe {
            let base = self.image_query_base(image_index);
            self.device.cmd_reset_query_pool(
                cmd,
                self.timestamp_query_pool,
                base,
                TIMESTAMPS_PER_IMAGE,
            );
            self.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.timestamp_query_pool,
                base + GpuTimestamp::FrameStart as u32,
            );
        }
    }

    /// Marks the end of the depth-only shadow pass.
    pub fn mark_shadow_timing_end(&self, cmd: vk::CommandBuffer, image_index: u32) {
        self.write_gpu_timestamp(cmd, image_index, GpuTimestamp::ShadowEnd);
    }

    /// Marks the portion of the main pass used by instanced grass and reeds.
    pub fn mark_vegetation_timing_start(&self, cmd: vk::CommandBuffer, image_index: u32) {
        self.write_gpu_timestamp(cmd, image_index, GpuTimestamp::VegetationStart);
    }

    /// Marks the end of instanced vegetation rendering.
    pub fn mark_vegetation_timing_end(&self, cmd: vk::CommandBuffer, image_index: u32) {
        self.write_gpu_timestamp(cmd, image_index, GpuTimestamp::VegetationEnd);
    }

    /// Marks the end of all recorded rendering work, including the UI pass but not present.
    pub fn end_gpu_timing(&self, cmd: vk::CommandBuffer, image_index: u32) {
        self.write_gpu_timestamp(cmd, image_index, GpuTimestamp::FrameEnd);
    }

    fn write_gpu_timestamp(&self, cmd: vk::CommandBuffer, image_index: u32, point: GpuTimestamp) {
        unsafe {
            self.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.timestamp_query_pool,
                self.image_query_base(image_index) + point as u32,
            );
        }
    }

    /// Per-pass timings from the previous completed GPU submission.
    pub fn latest_gpu_pass_timings(&self) -> GpuPassTimings {
        self.last_gpu_timings
    }

    #[inline]
    fn image_query_base(&self, image_index: u32) -> u32 {
        image_index * TIMESTAMPS_PER_IMAGE
    }

    /// Converts a monotonic timestamp pair into milliseconds.  An unavailable or invalid pair
    /// remains `None` instead of producing a misleading profiler value.
    fn timestamp_delta_ms(&self, start: u64, end: u64) -> Option<f32> {
        (end > start).then(|| {
            let delta_ticks = end - start;
            let ns = (delta_ticks as f64) * (self.timestamp_period_ns as f64);
            (ns / 1_000_000.0) as f32
        })
    }

    /// Update the global UBO with lighting data
    pub fn update_global_ubo(&mut self, image_index: usize, ubo: &GlobalUbo) {
        let allocator = self.allocator.as_ref().expect("allocator");
        let allocation = &self.ubo_allocations[image_index];
        unsafe {
            // These allocations were created with MAPPED, so VMA keeps the pointer valid
            // for their lifetime and the per-frame path does not need map/unmap calls.
            let ptr = allocator.get_allocation_info(allocation).mapped_data as *mut u8;
            debug_assert!(!ptr.is_null());
            std::ptr::copy_nonoverlapping(
                ubo as *const GlobalUbo as *const u8,
                ptr,
                std::mem::size_of::<GlobalUbo>(),
            );
            allocator
                .flush_allocation(allocation, 0, std::mem::size_of::<GlobalUbo>() as u64)
                .expect("flush UBO");
        }
    }

    /// Update set=0 binding=1 to point at the current frame's shadow image view
    pub fn update_shadow_descriptor(&mut self, image_index: usize, image_view: vk::ImageView) {
        // The view normally stays the same for an image, so avoid a redundant descriptor write.
        if self.shadow_descriptor_views[image_index] == image_view {
            return;
        }
        let set = self.set0_descriptor_sets[image_index];
        let image_info = vk::DescriptorImageInfo {
            sampler: self.shadow_sampler,
            image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let write = vk::WriteDescriptorSet {
            dst_set: set,
            dst_binding: 1,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info: &image_info,
            ..Default::default()
        };
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        self.shadow_descriptor_views[image_index] = image_view;
    }

    /// Apply any queued surface-dependent changes (like MSAA) at a frame boundary.
    pub fn apply_pending_surface_changes(
        &mut self,
        window: &winit::window::Window,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let mut recreated = false;
        if let Some(new_samples) = self.pending_msaa_samples.take()
            && new_samples != self.engine_settings.msaa_samples
        {
            self.engine_settings.msaa_samples = new_samples;
            // Sample count is baked into graphics pipelines; extent is not.
            self.pipeline_generation = self.pipeline_generation.saturating_add(1);
            self.recreate_swapchain(window)?;
            recreated = true;
        }
        Ok(recreated)
    }

    /// # Arguments
    /// * `entry` - The Ash Entry point.
    /// * `event_loop` - The winit event loop.
    /// * `layers` - The instance layers to enable.
    /// # Returns
    /// * `Result<Instance, Box<dyn Error>>` - The created Vulkan instance on success, or an error on failure.
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or an error if the swapchain could not be recreated.
    pub fn recreate_swapchain(&mut self, window: &Window) -> Result<(), Box<dyn Error>> {
        let old_color_format = self.swapchain.color_format;
        let old_depth_format = self.swapchain.depth_format;
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle before recreating swapchain");
        }
        self.swapchain.recreate(
            &self.instance,
            &self.device,
            self.physical_device,
            &self.surface,
            &self.surface_loader,
            window,
            self.allocator.as_ref().unwrap(),
            self.engine_settings.msaa_samples,
        )?;
        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);
        }
        self.command_buffers = Self::allocate_command_buffers(
            &self.device,
            self.command_pool,
            vk::CommandBufferLevel::PRIMARY,
            self.swapchain.swapchain_image_views.len(),
        )?;
        // Destroy old per-image present semaphores
        for &sem in &self.render_finished_semaphores {
            unsafe {
                self.device.destroy_semaphore(sem, None);
            }
        }
        // Recreate to match new image count
        let new_image_count = self.swapchain.swapchain_image_views.len();
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        self.render_finished_semaphores = Vec::with_capacity(new_image_count);
        for _ in 0..new_image_count {
            unsafe {
                self.render_finished_semaphores
                    .push(self.device.create_semaphore(&semaphore_info, None)?);
            }
        }
        // Reset per-image fence tracking
        self.image_owner_fence = vec![vk::Fence::null(); new_image_count];
        // Tear down old UBO buffers and descriptor pool
        for (buf, alloc) in self.ubo_buffers.iter().zip(self.ubo_allocations.iter_mut()) {
            unsafe {
                self.allocator.as_ref().unwrap().destroy_buffer(*buf, alloc);
            }
        }
        unsafe {
            self.device
                .destroy_descriptor_pool(self.set0_descriptor_pool, None);
        }

        // Recreate UBO buffers and set0 descriptor sets for the new image count
        let (new_ubo_buffers, new_ubo_allocations) =
            Self::create_uniform_buffers(self.allocator.as_ref().unwrap(), new_image_count);
        let (new_pool, new_sets) = Self::create_set0_descriptor_pool_and_sets(
            &self.device,
            self.set0_global_layout,
            &new_ubo_buffers,
        );

        self.ubo_buffers = new_ubo_buffers;
        self.ubo_allocations = new_ubo_allocations;
        self.set0_descriptor_pool = new_pool;
        self.set0_descriptor_sets = new_sets;
        self.shadow_descriptor_views = vec![vk::ImageView::null(); new_image_count];

        self.attachment_manager
            .cleanup(&self.device, self.allocator.as_ref().unwrap());
        self.attachment_manager = AttachmentManager::new(new_image_count);
        // Recreate the timestamp query pool to match new image count and per-pass markers.
        unsafe {
            self.device
                .destroy_query_pool(self.timestamp_query_pool, None);
        }
        let qp_info = vk::QueryPoolCreateInfo {
            query_type: vk::QueryType::TIMESTAMP,
            query_count: (new_image_count as u32) * TIMESTAMPS_PER_IMAGE,
            ..Default::default()
        };
        self.timestamp_query_pool = unsafe { self.device.create_query_pool(&qp_info, None)? };
        // Invalidate last GPU time reading since the pool was recreated
        self.last_gpu_timings = GpuPassTimings::default();
        // Avoid reading uninitialized queries on the next frame after recreation
        self.last_image_per_slot = vec![None; self.image_available_semaphores.len()];
        // Dynamic viewport/scissor removed extent from pipeline compatibility. A rare
        // surface-format change still requires every graphics pipeline to be rebuilt.
        if self.swapchain.color_format != old_color_format
            || self.swapchain.depth_format != old_depth_format
        {
            self.pipeline_generation = self.pipeline_generation.saturating_add(1);
        }
        self.swapchain_recreation_needed = false;
        Ok(())
    }

    const MSAA_CHOICES: [u32; 7] = [1, 2, 4, 8, 16, 32, 64];

    fn flag_for(samples: u32) -> vk::SampleCountFlags {
        vk::SampleCountFlags::from_raw(samples)
    }

    fn clamp_msaa_samples(&self, desired: u32) -> u32 {
        let mut opts = self.supported_msaa_samples();
        if opts.is_empty() {
            return 1;
        }
        opts.sort_unstable();
        // highest supported <= desired, else the smallest supported
        opts.iter()
            .copied()
            .rev()
            .find(|&n| n <= desired)
            .unwrap_or(opts[0])
    }

    /// Returns the list of supported MSAA sample counts for the current device.
    pub fn supported_msaa_samples(&self) -> Vec<u32> {
        let supported = self.sample_count_flags_supported;
        Self::MSAA_CHOICES
            .into_iter()
            .filter(|&n| supported.contains(Self::flag_for(n)))
            .collect()
    }

    /// Requests a new MSAA sample count. The swapchain/pipelines will be
    /// recreated on the next safe point to apply the change.
    pub fn request_msaa_samples(&mut self, desired: u32) {
        self.pending_msaa_samples = Some(self.clamp_msaa_samples(desired));
    }

    /// Current MSAA sample count in effect.
    pub fn get_msaa_samples(&self) -> u32 {
        self.engine_settings.msaa_samples
    }

    /// Toggles the wireframe mode in the debug settings.
    pub fn toggle_wireframe(&mut self) {
        let old = self.engine_settings.wireframe;
        self.engine_settings.wireframe = !self.engine_settings.wireframe;
        if self.engine_settings.wireframe != old {
            // Wireframe affects pipelines; ensure rebuild next frame
            self.pipeline_generation = self.pipeline_generation.saturating_add(1);
        }
    }
    /// Returns the current pipeline generation counter.
    pub fn pipeline_generation(&self) -> u64 {
        self.pipeline_generation
    }

    /// Returns and clears the pending swapchain-recreation request.
    pub fn take_swapchain_recreation_request(&mut self) -> bool {
        std::mem::take(&mut self.swapchain_recreation_needed)
    }

    /// Queue a resize for the next frame boundary. Several window events collapse into one.
    pub fn request_swapchain_recreation(&mut self) {
        self.swapchain_recreation_needed = true;
    }
}

impl Drop for VulkanBase {
    /// Cleans up Vulkan resources when the `VulkanBase` is dropped.
    /// This includes waiting for the device to be idle,
    /// cleaning up the swapchain, pipeline, command pool,
    /// and other Vulkan objects.
    fn drop(&mut self) {
        println!("💧 Dropping VulkanBase");
        unsafe {
            // `Drop` gives us `&mut self`, but Vulkan does not know Rust's
            // ownership graph.  Destroy resources in reverse dependency order.
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");

            self.attachment_manager
                .cleanup(&self.device, self.allocator.as_ref().unwrap());
            self.swapchain.cleanup(
                &self.instance,
                &self.device,
                self.allocator.as_ref().unwrap(),
            );
            self.device
                .destroy_descriptor_set_layout(self.set0_global_layout, None);

            // Then destroy the rest of the resources
            for &sem in &self.image_available_semaphores {
                self.device.destroy_semaphore(sem, None);
            }
            for &sem in &self.render_finished_semaphores {
                self.device.destroy_semaphore(sem, None);
            }
            for &fence in &self.in_flight_fences {
                self.device.destroy_fence(fence, None);
            }
            self.device
                .destroy_query_pool(self.timestamp_query_pool, None);
            for &buffer in &self.command_buffers {
                self.device
                    .free_command_buffers(self.command_pool, &[buffer]);
            }
            // UBO and descriptor resources
            for (buf, alloc) in self.ubo_buffers.iter().zip(self.ubo_allocations.iter_mut()) {
                self.allocator.as_ref().unwrap().destroy_buffer(*buf, alloc);
            }
            self.device
                .destroy_descriptor_pool(self.set0_descriptor_pool, None);
            self.device.destroy_sampler(self.shadow_sampler, None);
            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);
            self.device.destroy_command_pool(self.command_pool, None);
            // `Option::take` moves the allocator out while we only have `&mut self`.
            // Dropping it here frees VMA memory before its Vulkan device disappears.
            if let Some(alloc) = self.allocator.take() {
                // (optional) quick summary before it goes away
                if let Ok(stats) = alloc.calculate_statistics() {
                    let s = stats.total.statistics;
                    println!(
                        "🗑️ VMA total before drop: allocs={} blocks={} allocBytes={} blockBytes={}",
                        s.allocationCount, s.blockCount, s.allocationBytes, s.blockBytes
                    );
                }
                drop(alloc);
            }
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            #[cfg(debug_assertions)]
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Debug settings for the Vulkan engine
pub struct EngineSettings {
    pub wireframe: bool,
    pub show_ui: bool,
    pub msaa_samples: u32,
    pub shadow_map_resolution: u32,
    pub shadow_distance: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GpuLight {
    pub position: [f32; 3],
    pub intensity: f32, // packs with position to 16 bytes
    pub color: [f32; 3],
    pub _pad: f32, // pad to 16 bytes
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GpuDirLight {
    pub direction: [f32; 3],
    pub intensity: f32, // keep std140 friendly packing
    pub color: [f32; 3],
    pub _pad: f32,
}

/// Global (per-frame/per-image) uniform buffer object shared across all pipelines via **descriptor set 0, binding 0**.
/// There is **one buffer per swapchain image** so the CPU can update the UBO while another image is still in-flight.
/// Keep fields 16-byte aligned for std140-like layouts. The order here must match the GLSL:
///     Light lights[MAX_LIGHTS]; uint light_count; uvec3 _pad0;
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct GlobalUbo {
    // Directional light + shadow matrix first for clarity
    pub dir_light: GpuDirLight,  // single directional light (sun)
    pub light_vp: [[f32; 4]; 4], // light view-projection (column-major)
    // Then the array of point lights (std140 array of structs)
    pub lights: [GpuLight; crate::app::app::MAX_LIGHTS], // array of point lights
    pub light_count: u32,                                // number of active point lights
    pub _pad0: [u32; 3],                                 // pad to 16B multiple (std140)
}
