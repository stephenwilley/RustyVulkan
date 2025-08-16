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

use ash::{Entry, Instance};
use ash::vk;
use ash_window::enumerate_required_extensions;
use ash::khr::surface;
use ash_window::create_surface;
use ash::khr::swapchain;
use winit::event_loop::{ActiveEventLoop};
use winit::window::Window;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::ffi::CStr;
use std::error::Error;

use super::swapchain::Swapchain;

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
    /// Graphics queue from the selected family.
    pub graphics_queue: vk::Queue,
    // -- Global set 0 related --
    /// Descriptor set layout for the global UBO (set = 0, binding = 0).
    pub set0_global_layout: vk::DescriptorSetLayout,
    set0_descriptor_pool: vk::DescriptorPool,
    /// One descriptor set per swapchain image for set 0.
    pub set0_descriptor_sets: Vec<vk::DescriptorSet>,
    ubo_buffers: Vec<vk::Buffer>,
    /// Device memory backing each per-image UBO buffer.
    pub ubo_memory: Vec<vk::DeviceMemory>,
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
    pub debug_settings: EngineDebugSettings,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,
    #[cfg(debug_assertions)]
    debug_utils_loader: ash::ext::debug_utils::Instance,
    frame_slot: usize,
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
            self.device.wait_for_fences(&[self.in_flight_fences[slot]], true, u64::MAX)?;

            // Acquire an image; signal when it's ready via the per-slot image-available semaphore.
            let (image_index, _is_suboptimal) = match self.swapchain_loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                self.image_available_semaphores[slot],
                vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(None),
                Err(e) => return Err(e.into()),
            };

            let idx = image_index as usize;

            // If this image is still tied to an older in-flight slot, wait for that slot to finish first.
            if self.image_owner_fence[idx] != vk::Fence::null() {
                self.device.wait_for_fences(&[self.image_owner_fence[idx]], true, u64::MAX)?;
            }

            // Reuse this slot's fence for the new submit and associate it with this image.
            self.device.reset_fences(&[self.in_flight_fences[slot]])?;
            self.image_owner_fence[idx] = self.in_flight_fences[slot];

            // Begin recording and open dynamic rendering so the caller can just bind/draw.
            let cmd_buf    = self.command_buffers[idx];
            let extent      = self.swapchain.extent;

            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device.begin_command_buffer(cmd_buf, &begin_info)?;

            // Transition swapchain image to COLOR_ATTACHMENT_OPTIMAL and depth to DEPTH_ATTACHMENT_OPTIMAL
            let color_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                old_layout: self.swapchain.image_layouts[idx],
                new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: self.swapchain.images[idx],
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };
            let depth_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                old_layout: self.swapchain.depth_layouts[idx],
                new_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: self.swapchain.depth_images[idx],
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[color_barrier, depth_barrier],
            );
            self.swapchain.image_layouts[idx] = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            self.swapchain.depth_layouts[idx] = vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL;

            let clear_color = vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } };
            let clear_depth = vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } };
            let color_attachment = vk::RenderingAttachmentInfo {
                image_view: self.swapchain.swapchain_image_views[idx],
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value: clear_color,
                ..Default::default()
            };
            let depth_attachment = vk::RenderingAttachmentInfo {
                image_view: self.swapchain.depth_image_views[idx],
                image_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::DONT_CARE,
                clear_value: clear_depth,
                ..Default::default()
            };
            let rendering_info = vk::RenderingInfo {
                render_area: vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent },
                layer_count: 1,
                color_attachment_count: 1,
                p_color_attachments: &color_attachment,
                p_depth_attachment: &depth_attachment,
                ..Default::default()
            };
            self.device.cmd_begin_rendering(cmd_buf, &rendering_info);

            Ok(Some(FrameCtx { cmd_buf, image_index, frame_slot: slot }))
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
            // Close dynamic rendering and command buffer
            self.device.cmd_end_rendering(frame.cmd_buf);

            // Transition image back for presentation
            let idx = frame.image_index as usize;
            let present_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::empty(),
                old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: self.swapchain.images[idx],
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier(
                frame.cmd_buf,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[present_barrier],
            );
            self.swapchain.image_layouts[idx] = vk::ImageLayout::PRESENT_SRC_KHR;

            self.device.end_command_buffer(frame.cmd_buf)?;

            // Submit: wait for image-available (slot), signal render-finished (per-image)
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let wait_sems   = [self.image_available_semaphores[frame.frame_slot]];
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
            match self.swapchain_loader.queue_present(self.graphics_queue, &present_info) {
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => { /* caller will recreate if needed */ }
                Err(e) => return Err(e.into()),
                _ => {}
            }

            // Advance to next CPU slot
            self.frame_slot = (frame.frame_slot + 1) % self.image_available_semaphores.len();
        }
        Ok(())
    }
    /// Helper: find a suitable memory type on the physical device.
    fn find_memory_type(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        for i in 0..mem_props.memory_type_count {
            let mt = mem_props.memory_types[i as usize];
            if (type_filter & (1 << i)) != 0 && mt.property_flags.contains(properties) {
                return i;
            }
        }
        panic!("Failed to find suitable memory type");
    }

    /// Creates a Vulkan instance.
    /// # Arguments
    /// * `entry` - The Ash Entry point.
    /// * `event_loop` - The winit event loop.
    /// * `layers` - The instance layers to enable.
    /// # Returns
    /// * `Result<Instance, Box<dyn Error>>` - The created Vulkan instance on success, or an error on failure.
    fn create_instance(
        entry: &Entry, event_loop: &ActiveEventLoop, layers: &[*const i8]
    ) -> Result<Instance, Box<dyn Error>> {
        let app_name = std::ffi::CString::new("Ash Vulkan Tutorial")?;

        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: app_name.as_ptr(),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::API_VERSION_1_3,
            ..Default::default()
        };

        let ext_names = enumerate_required_extensions(event_loop.display_handle().unwrap().as_raw())?;
        let mut extension_ptrs: Vec<*const i8> = ext_names.to_vec();

        #[cfg(debug_assertions)]
        extension_ptrs.push(ash::ext::debug_utils::NAME.as_ptr());

        // Add portability enumeration so macOS MoltenVK gets picked up:
        extension_ptrs.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
        extension_ptrs.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr());

        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
            p_application_info: &app_info,
            enabled_layer_count: layers.len() as u32,
            pp_enabled_layer_names: layers.as_ptr(),
            enabled_extension_count: extension_ptrs.len() as u32,
            pp_enabled_extension_names: extension_ptrs.as_ptr(),
            ..Default::default()
        };

        Ok(unsafe { entry.create_instance(&create_info, None)? })
    }

    #[cfg(debug_assertions)]
    fn print_physical_devices(instance: &Instance, physical_devices: &[vk::PhysicalDevice]) {
        println!("Detected {} physical device(s):", physical_devices.len());

        for device in physical_devices.iter() {
            let props = unsafe {
                instance.get_physical_device_properties(*device)
            };
            let name_cstr = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
            let name = name_cstr.to_str().unwrap_or("<invalid utf-8>");
            println!(
                " • Device: '{}' (type: {:?}) — API version: {}.{}.{}",
                name,
                props.device_type,
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version),
                vk::api_version_patch(props.api_version),
            );
        }
    }

    /// Chooses a suitable physical device.
    /// Prioritizes discrete GPUs, otherwise selects the first available device.
    /// # Arguments
    /// * `instance` - The Vulkan instance.
    /// * `physical_devices` - A slice of available physical devices.
    /// # Returns
    /// * `vk::PhysicalDevice` - The chosen physical device.
    fn choose_device(instance: &Instance, physical_devices: &[vk::PhysicalDevice]) -> vk::PhysicalDevice {
        physical_devices
            .iter()
            .find(|&d| {
                let props = unsafe { instance.get_physical_device_properties(*d) };
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .copied()
            .unwrap_or(physical_devices[0])
    }

    /// Finds the index of a queue family that supports graphics operations.
    /// # Arguments
    /// * `instance` - The Vulkan instance.
    /// * `physical_device` - The physical device to query.
    /// # Returns
    /// * `Result<u32, String>` - The queue family index on success, or an error string if not found.
    fn find_graphics_queue_family_index(instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<u32, String> {
        let queue_family_properties = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };

        let index = queue_family_properties
            .iter()
            .enumerate()
            .find(|(_, info)| info.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(index, _)| index as u32);

        match index {
            Some(i) => {
                println!("🎯 Found graphics queue family at index {}", i);
                Ok(i)
            }
            None => Err("Could not find a graphics queue family".to_string()),
        }
    }

    /// Creates a logical device and retrieves the graphics queue.
    /// # Arguments
    /// * `instance` - The Vulkan instance.
    /// * `physical_device` - The physical device to create the logical device from.
    /// * `queue_family_index` - The index of the graphics queue family.
    /// # Returns
    /// * `Result<(ash::Device, vk::Queue), vk::Result>` - A tuple containing the logical device and graphics queue on success, or a Vulkan error on failure.
    fn create_logical_device_and_queue(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> Result<(ash::Device, vk::Queue), vk::Result> {
        let queue_priority = [1.0_f32];

        // Query and enable device features, including sampler anisotropy
        let mut device_features = unsafe {
            instance.get_physical_device_features(physical_device)
        };
        device_features.sampler_anisotropy = vk::TRUE;

        let queue_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: queue_priority.as_ptr(),
            ..Default::default()
        };

        let device_extensions = [
            vk::KHR_SWAPCHAIN_NAME.as_ptr(),
            vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr(),
            vk::KHR_DYNAMIC_RENDERING_NAME.as_ptr(),
        ];

        let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default();
        dynamic_rendering_features.dynamic_rendering = vk::TRUE;

        let device_create_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            p_next: &mut dynamic_rendering_features as *mut _ as *const std::ffi::c_void,
            p_queue_create_infos: &queue_info,
            queue_create_info_count: 1,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            p_enabled_features: &device_features,
            ..Default::default()
        };

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        println!("✅ Logical device and graphics queue ready");
        Ok((device, queue))
    }

    /// Creates a Vulkan surface for rendering.
    /// # Arguments
    /// * `entry` - The Ash Entry point.
    /// * `instance` - The Vulkan instance.
    /// * `window` - The winit window.
    /// * `event_loop` - The winit event loop.
    /// # Returns
    /// * `Result<vk::SurfaceKHR, vk::Result>` - The created surface on success, or a Vulkan error on failure.
    fn create_surface(
        entry: &Entry,
        instance: &Instance,
        window: &Window,
        event_loop: &ActiveEventLoop,
    ) -> Result<vk::SurfaceKHR, vk::Result> {
        let surface = unsafe {
            create_surface(
                entry,
                instance,
                event_loop.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )?
        };
        println!("🌊 Surface created");
        Ok(surface)
    }

    /// Creates a Vulkan command pool.
    /// # Arguments
    /// * `device` - The Vulkan logical device.
    /// * `queue_family_index` - The index of the queue family to associate with the command pool.
    /// # Returns
    /// * `Result<vk::CommandPool, vk::Result>` - The created command pool on success, or a Vulkan error on failure.
    fn create_command_pool(device: &ash::Device, queue_family_index: u32) -> Result<vk::CommandPool, vk::Result> {
        let info = vk::CommandPoolCreateInfo {
            queue_family_index,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };
        let command_pool = unsafe { device.create_command_pool(&info, None)? };
        println!("📝 Command pool created");
        Ok(command_pool)
    }

    /// Allocates Vulkan command buffers.
    /// # Arguments
    /// * `device` - The Vulkan logical device.
    /// * `command_pool` - The command pool to allocate from.
    /// * `buffer_level` - The level of the command buffers.
    /// * `count` - The number of command buffers to allocate.
    /// # Returns
    /// * `Result<Vec<vk::CommandBuffer>, vk::Result>` - A vector of allocated command buffers on success, or a Vulkan error on failure.
    fn allocate_command_buffers(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        buffer_level: vk::CommandBufferLevel,
        count: usize,
    ) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool,
            level: buffer_level,
            command_buffer_count: count as u32,
            ..Default::default()
        };
        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };
        println!("📝 Allocated {} command buffers", count);
        Ok(command_buffers)
    }

    /// Create N host-visible, coherent uniform buffers sized for `GlobalUbo`.
    fn create_uniform_buffers(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        count: usize,
    ) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
        let mut buffers = Vec::with_capacity(count);
        let mut memories = Vec::with_capacity(count);

        let buffer_size = std::mem::size_of::<GlobalUbo>() as vk::DeviceSize;

        for _ in 0..count {
            // Buffer
            let buffer_info = vk::BufferCreateInfo {
                size: buffer_size,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let buffer = unsafe { device.create_buffer(&buffer_info, None).expect("create uniform buffer") };

            // Memory
            let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
            let mem_type = Self::find_memory_type(
                instance,
                physical_device,
                reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            let alloc_info = vk::MemoryAllocateInfo {
                allocation_size: reqs.size,
                memory_type_index: mem_type,
                ..Default::default()
            };
            let memory = unsafe { device.allocate_memory(&alloc_info, None).expect("alloc uniform memory") };

            unsafe { device.bind_buffer_memory(buffer, memory, 0).expect("bind uniform memory"); }

            buffers.push(buffer);
            memories.push(memory);
        }

        (buffers, memories)
    }

    /// Build a descriptor pool and one set=0 descriptor set per swapchain image, then write binding 0 to each UBO.
    fn create_set0_descriptor_pool_and_sets(
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
        ubo_buffers: &[vk::Buffer],
    ) -> (vk::DescriptorPool, Vec<vk::DescriptorSet>) {
        let count = ubo_buffers.len() as u32;

        // Pool
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: count,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo {
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            max_sets: count,
            ..Default::default()
        };
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None).expect("create set0 pool") };

        // Allocate
        let layouts = vec![layout; count as usize];
        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: pool,
            descriptor_set_count: count,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };
        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info).expect("alloc set0 sets") };

        // Write binding 0
        let range = std::mem::size_of::<GlobalUbo>() as vk::DeviceSize;
        let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(count as usize);
        for &b in ubo_buffers {
            buf_infos.push(vk::DescriptorBufferInfo { buffer: b, offset: 0, range });
        }

        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(count as usize);
        for i in 0..(count as usize) {
            writes.push(vk::WriteDescriptorSet {
                dst_set: sets[i],
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_buffer_info: &buf_infos[i],
                ..Default::default()
            });
        }
        unsafe { device.update_descriptor_sets(&writes, &[]); }

        (pool, sets)
    }

    /// Recreates the swapchain and associated resources when the window is resized.
    /// # Arguments
    /// * `window` - The winit `Window` to associate with the new swapchain.
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or an error if the swapchain could not be recreated.
    pub fn recreate_swapchain(&mut self, window: &Window) -> Result<(), Box<dyn Error>> {
        unsafe {
            self.device.device_wait_idle().expect("Failed to wait device idle before recreating swapchain");
        }
        self.swapchain.recreate(
            &self.instance,
            &self.device,
            self.physical_device,
            &self.surface,
            &self.surface_loader,
            window
        )?;
        unsafe {
            self.device.free_command_buffers(self.command_pool, &self.command_buffers);
        }
        self.command_buffers = Self::allocate_command_buffers(
            &self.device,
            self.command_pool,
            vk::CommandBufferLevel::PRIMARY,
            self.swapchain.swapchain_image_views.len(),
        )?;
        // Destroy old per-image present semaphores
        for &sem in &self.render_finished_semaphores {
            unsafe { self.device.destroy_semaphore(sem, None); }
        }
        // Recreate to match new image count
        let new_image_count = self.swapchain.swapchain_image_views.len();
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        self.render_finished_semaphores = Vec::with_capacity(new_image_count);
        for _ in 0..new_image_count {
            unsafe { self.render_finished_semaphores.push(self.device.create_semaphore(&semaphore_info, None)?); }
        }
        // Reset per-image fence tracking
        self.image_owner_fence = vec![vk::Fence::null(); new_image_count];
        // Tear down old UBO buffers and descriptor pool
        for &buf in &self.ubo_buffers {
            unsafe { self.device.destroy_buffer(buf, None); }
        }
        for &mem in &self.ubo_memory {
            unsafe { self.device.free_memory(mem, None); }
        }
        unsafe { self.device.destroy_descriptor_pool(self.set0_descriptor_pool, None); }

        // Recreate UBO buffers and set0 descriptor sets for the new image count
        let (new_ubo_buffers, new_ubo_memory) =
            Self::create_uniform_buffers(&self.instance, &self.device, self.physical_device, new_image_count);
        let (new_pool, new_sets) =
            Self::create_set0_descriptor_pool_and_sets(&self.device, self.set0_global_layout, &new_ubo_buffers);

        self.ubo_buffers = new_ubo_buffers;
        self.ubo_memory = new_ubo_memory;
        self.set0_descriptor_pool = new_pool;
        self.set0_descriptor_sets = new_sets;

        Ok(())
    }

    /// Toggles the wireframe mode in the debug settings.
    pub fn toggle_wireframe(&mut self) {
        self.debug_settings.wireframe = !self.debug_settings.wireframe;
    }
    /// Toggles the FPS display
    pub fn toggle_ms_per_frame(&mut self) {
        self.debug_settings.show_ms_per_frame = !self.debug_settings.show_ms_per_frame;
    }

    /// Creates a new `VulkanBase` instance, initializing Vulkan resources and setting up the swapchain.
    /// # Arguments
    /// * `window` - The winit `Window` to associate with the Vulkan instance.
    /// * `event_loop` - The winit `ActiveEventLoop` to use for event handling.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - Returns the initialized `VulkanBase` on success, or an error on failure.
    pub fn new(window: &Window, event_loop: &ActiveEventLoop) -> Result<Self, Box<dyn Error>> {
        let debug_settings = EngineDebugSettings {
            wireframe: false,
            show_ms_per_frame: false,
        };

        let entry = Entry::linked();

        #[cfg(debug_assertions)]
        let layer_names = [c"VK_LAYER_KHRONOS_validation"];
        #[cfg(debug_assertions)]
        let layer_name_ptrs: Vec<*const i8> = layer_names.iter().map(|s| s.as_ptr()).collect();

        #[cfg(not(debug_assertions))]
        let layer_name_ptrs: Vec<*const i8> = Vec::new();

        #[cfg(debug_assertions)]
        {
            unsafe {
                let available_layers = entry.enumerate_instance_layer_properties()?;
                let validation_layer_name = c"VK_LAYER_KHRONOS_validation";
                let is_layer_available = available_layers.iter().any(|layer| {
                    let name = CStr::from_ptr(layer.layer_name.as_ptr());
                    name == validation_layer_name
                });
                if !is_layer_available {
                    return Err("Validation layers requested, but not available.".into());
                }
            }
            println!("✅ Validation layers available and requested.");
        }

        let instance = Self::create_instance(&entry, event_loop, &layer_name_ptrs)?;
        println!("🛡️ Vulkan Instance created");

        #[cfg(debug_assertions)]
        let (debug_utils_loader, debug_messenger) = Self::setup_debug_messenger(&entry, &instance)?;

        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        #[cfg(debug_assertions)]
        Self::print_physical_devices(&instance, &physical_devices);

        let physical_device = Self::choose_device(&instance, &physical_devices);
        let chosen_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let chosen_name = unsafe {
            CStr::from_ptr(chosen_props.device_name.as_ptr()).to_str().unwrap_or("<invalid utf-8>")
        };
        println!("👉 Selected device for next steps: '{}'", chosen_name);

        let graphics_queue_family_index = Self::find_graphics_queue_family_index(&instance, physical_device)?;

        let (device, graphics_queue) =
            Self::create_logical_device_and_queue(&instance, physical_device, graphics_queue_family_index)?;

        let surface = Self::create_surface(&entry, &instance, window, event_loop)?;
        let surface_loader = surface::Instance::new(&entry, &instance);

        let command_pool = Self::create_command_pool(&device, graphics_queue_family_index)?;

        // --- Global set-0 layout: reserve binding 0 for a per-frame/per-image UBO ---
        let ubo_binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        };
        let set0_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: 1,
            p_bindings: &ubo_binding,
            ..Default::default()
        };
        let set0_global_layout = unsafe { device.create_descriptor_set_layout(&set0_info, None)? };
        println!("🔧 Created global set=0 layout (binding 0 = UBO)");

        let swapchain = Swapchain::new(
            &instance,
            &device,
            physical_device,
            &surface,
            &surface_loader,
            window)?;
        let swapchain_loader= swapchain::Device::new(&instance, &device);

        // Create sync objects: N frames-in-flight worth of semaphores/fences
        let image_count = swapchain.swapchain_image_views.len();
        let (ubo_buffers, ubo_memory) =
            Self::create_uniform_buffers(&instance, &device, physical_device, image_count);
        let (set0_descriptor_pool, set0_descriptor_sets) =
            Self::create_set0_descriptor_pool_and_sets(&device, set0_global_layout, &ubo_buffers);
        let mut image_available_semaphores = Vec::with_capacity(INFLIGHT_FRAMES);
        let mut in_flight_fences = Vec::with_capacity(INFLIGHT_FRAMES);
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };
        for _ in 0..INFLIGHT_FRAMES {
            unsafe {
                image_available_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
                in_flight_fences.push(device.create_fence(&fence_info, None)?);
            }
        }

        let mut render_finished_semaphores = Vec::with_capacity(image_count);
        for _ in 0..image_count {
            unsafe { render_finished_semaphores.push(device.create_semaphore(&semaphore_info, None)?); }
        }

        // Per-swapchain-image tracker: which fence currently owns each image (or null)
        let image_owner_fence = vec![vk::Fence::null(); image_count];

        let command_buffers = Self::allocate_command_buffers(
            &device,
            command_pool,
            vk::CommandBufferLevel::PRIMARY,
            swapchain.swapchain_image_views.len())?;

        let vulkan_base = Self {
            instance,
            physical_device,
            device,
            graphics_queue,
            // set 0
            set0_global_layout,
            set0_descriptor_pool,
            set0_descriptor_sets,
            ubo_buffers,
            ubo_memory,
            // sync
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            image_owner_fence,
            // swapchain
            swapchain,
            swapchain_loader,
            surface,
            surface_loader,
            // commands
            command_pool,
            command_buffers,
            // misc
            debug_settings,
            #[cfg(debug_assertions)]
            debug_messenger,
            #[cfg(debug_assertions)]
            debug_utils_loader,
            frame_slot: 0,
        };

        println!("✅ VulkanBase initialized successfully");
        Ok(vulkan_base)
    }

    #[cfg(debug_assertions)]
    fn setup_debug_messenger(
        entry: &Entry,
        instance: &Instance,
    ) -> Result<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT), Box<dyn Error>> {
        let loader = ash::ext::debug_utils::Instance::new(entry, instance);
        let create_info = vk::DebugUtilsMessengerCreateInfoEXT {
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            pfn_user_callback: Some(vulkan_debug_callback),
            ..Default::default()
        };
        let messenger = unsafe {
            loader
                .create_debug_utils_messenger(&create_info, None)?
        };
        println!("🔍 Debug messenger created");
        Ok((loader, messenger))
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
            self.device.device_wait_idle().expect("Failed to wait device idle");

            self.swapchain.cleanup(&self.instance, &self.device);
            self.device.destroy_descriptor_set_layout(self.set0_global_layout, None);

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
            for &buffer in &self.command_buffers {
                self.device.free_command_buffers(self.command_pool, &[buffer]);
            }
            // UBO and descriptor resources
            for &buf in &self.ubo_buffers {
                self.device.destroy_buffer(buf, None);
            }
            for &mem in &self.ubo_memory {
                self.device.free_memory(mem, None);
            }
            self.device.destroy_descriptor_pool(self.set0_descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            #[cfg(debug_assertions)]
            self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Debug settings for the Vulkan engine
pub struct EngineDebugSettings {
    pub wireframe: bool,
    pub show_ms_per_frame: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GpuLight {
    pub position: [f32; 3],
    pub intensity: f32,   // packs with position to 16 bytes
    pub color: [f32; 3],
    pub _pad: f32,        // pad to 16 bytes
}

/// Global (per-frame/per-image) uniform buffer object shared across all pipelines via **descriptor set 0, binding 0**.
/// There is **one buffer per swapchain image** so the CPU can update the UBO while another image is still in-flight.
/// Keep fields 16-byte aligned for std140-like layouts. The order here must match the GLSL:
///     Light lights[MAX_LIGHTS]; uint light_count; uvec3 _pad0;
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct GlobalUbo {
    pub lights: [GpuLight; crate::app::app::MAX_LIGHTS],      // array of structs (std140: 16-byte aligned/strided)
    pub light_count: u32,                                     // scalar after array
    pub _pad0: [u32; 3],                                      // pad to 16B multiple (std140)
}