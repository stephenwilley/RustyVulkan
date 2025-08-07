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
    pub instance: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    surface: vk::SurfaceKHR,
    surface_loader: surface::Instance,
    pub command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    pub swapchain: Swapchain,
    swapchain_loader: swapchain::Device,
    secondary_command_pool: vk::CommandPool,
    secondary_command_buffers: Vec<vk::CommandBuffer>,
    pub debug_settings: EngineDebugSettings,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,
    #[cfg(debug_assertions)]
    debug_utils_loader: ash::ext::debug_utils::Instance,
    current_frame: usize,
}

impl VulkanBase {
    fn create_instance(
        entry: &Entry, event_loop: &ActiveEventLoop, layers: &[*const i8]
    ) -> Result<Instance, Box<dyn Error>> {
        let app_name = std::ffi::CString::new("Ash Vulkan Tutorial")?;

        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: app_name.as_ptr(),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::API_VERSION_1_0,
            ..Default::default()
        };

        let ext_names = enumerate_required_extensions(event_loop.display_handle().unwrap().as_raw())?;
        let mut extension_ptrs: Vec<*const i8> = ext_names.iter().copied().collect();

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
        ];

        let device_create_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            p_next: std::ptr::null(),
            p_queue_create_infos: &queue_info,
            queue_create_info_count: 1,
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            p_enabled_features: &device_features,
            ..Default::default()
        };

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        println!("✅ Logical device and graphics queue ready");
        Ok((device, queue))
    }

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

    /// Records the commands for a single primary command buffer.
    /// This involves beginning the render pass and executing the corresponding secondary command buffer.
    fn record_primary_command_buffer(
        &self,
        image_index: usize,
    ) -> Result<(), vk::Result> {
        let command_buffer = self.command_buffers[image_index];
        let framebuffer = self.swapchain.framebuffers[image_index];
        let render_pass = self.swapchain.render_pass;
        let extent = self.swapchain.extent;

        let begin_info = vk::CommandBufferBeginInfo::default();
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            },
        ];
        let render_pass_info = vk::RenderPassBeginInfo {
            render_pass,
            framebuffer,
            render_area: vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent },
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.device.begin_command_buffer(command_buffer, &begin_info)?;
            self.device.cmd_begin_render_pass(command_buffer, &render_pass_info, vk::SubpassContents::SECONDARY_COMMAND_BUFFERS);
            self.device.cmd_execute_commands(command_buffer, &[self.secondary_command_buffers[image_index]]);
            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer)?;
        }
        Ok(())
    }

    pub fn record_command_buffers(&self) -> Result<(), vk::Result> {
        for (index, _) in self.command_buffers.iter().enumerate() {
            self.record_primary_command_buffer(index)?;
        }
        Ok(())
    }

    fn record_secondary_command_buffer<F>(
        &self,
        image_index: usize,
        mut record_fn: F,
    ) -> Result<(), vk::Result>
    where F: FnMut(&VulkanBase, vk::CommandBuffer),
    {
        let inh = vk::CommandBufferInheritanceInfo {
            render_pass:   self.swapchain.render_pass,
            subpass:       0,
            framebuffer:   self.swapchain.framebuffers[image_index],
            ..Default::default()
        };
        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
                | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
            p_inheritance_info: &inh,
            ..Default::default()
        };

        unsafe {
            let cmd = self.secondary_command_buffers[image_index];
            self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(cmd, &begin_info)?;
            record_fn(self, cmd);
            self.device.end_command_buffer(cmd)?;
        }

        Ok(())
    }

    /// Draws a frame by acquiring an image from the swapchain, submitting a command buffer,
    /// and presenting the image.
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or an error if the frame could not be drawn.
    pub fn draw_frame<F>(
        &mut self,
        mut record_secondary: F,
    ) -> Result<(), Box<dyn Error>>
    where F: FnMut(&VulkanBase, vk::CommandBuffer),
    {
        unsafe {
            let frame = self.current_frame;
            // Wait for the fence of the frame we want to use. This ensures that the command
            // buffer and semaphores for this frame index are no longer in use by the GPU.
            self.device.wait_for_fences(&[self.in_flight_fences[frame]], true, u64::MAX)?;

            // Acquire next image
            let (image_index, _is_suboptimal) = match self.swapchain_loader
                .acquire_next_image(
                    self.swapchain.handle,
                    u64::MAX,
                    self.image_available_semaphores[frame],
                    vk::Fence::null(),
                ) {
                    Ok(result) => result,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        // The swapchain is out of date (e.g., window was resized) and
                        // must be recreated. The event loop will handle this.
                        return Ok(());
                    }
                    Err(e) => return Err(e.into()),
                };
            
            // Now that we've waited, we can safely reset the fence for this frame.
            self.device.reset_fences(&[self.in_flight_fences[frame]])?;

            let idx = image_index as usize;
            self.record_secondary_command_buffer(idx, &mut record_secondary)?;
            // Re-record the primary command buffer right before submission to ensure it's valid.
            self.record_primary_command_buffer(idx)?;

            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let submit_info = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: &self.image_available_semaphores[frame],
                p_wait_dst_stage_mask: wait_stages.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: &self.command_buffers[idx],
                signal_semaphore_count: 1,
                p_signal_semaphores: &self.render_finished_semaphores[frame],
                ..Default::default()
            };

            // Submit the command buffer to the graphics queue, signaling the in_flight_fence
            // when it's done.
            self.device.queue_submit(
                self.graphics_queue,
                &[submit_info],
                self.in_flight_fences[frame],
            )?;

            let present_info = vk::PresentInfoKHR {
                wait_semaphore_count: 1,
                p_wait_semaphores: &self.render_finished_semaphores[frame],
                swapchain_count: 1,
                p_swapchains: &self.swapchain.handle,
                p_image_indices: &image_index,
                ..Default::default()
            };

            match self.swapchain_loader.queue_present(self.graphics_queue, &present_info) {
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    // Swapchain is suboptimal or out of date. The event loop will handle recreation.
                }
                Err(e) => return Err(e.into()),
                _ => {}
            }
            
            self.current_frame = (frame + 1) % self.image_available_semaphores.len();
        }
        Ok(())
    }

    /// Recreates the swapchain and associated resources when the window is resized.
    /// # Arguments
    /// * `window` - The winit `Window` to associate with the new swapchain.
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or an error if the swapchain could not be recreated.
    pub fn recreate_swapchain(&mut self, window: &Window) -> Result<(), Box<dyn Error>> {
        // Ensure GPU is idle before destroying and reallocating command buffers
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
        // Free old primary command buffers before reallocating
        unsafe {
            self.device.free_command_buffers(self.command_pool, &self.command_buffers);
        }
        self.command_buffers = Self::allocate_command_buffers(
            &self.device,
            self.command_pool,
            vk::CommandBufferLevel::PRIMARY,
            self.swapchain.swapchain_image_views.len(),
        )?;
        // Free old secondary command buffers before reallocating
        unsafe {
            self.device.free_command_buffers(self.secondary_command_pool, &self.secondary_command_buffers);
        }
        self.secondary_command_buffers = Self::allocate_command_buffers(
            &self.device,
            self.secondary_command_pool,
            vk::CommandBufferLevel::SECONDARY,
            self.swapchain.swapchain_image_views.len(),
        )?;
        // Pre-record each secondary command buffer (no-op) so they're executable
        for idx in 0..self.secondary_command_buffers.len() {
            self.record_secondary_command_buffer(idx, |_, _| {})
                .expect("Failed to pre-record secondary command buffer");
        }
        self.record_command_buffers()?;
        // The secondary command buffer is recorded in the draw_frame method
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
        let layer_names = [CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap()];
        #[cfg(debug_assertions)]
        let layer_name_ptrs: Vec<*const i8> = layer_names.iter().map(|s| s.as_ptr()).collect();

        #[cfg(not(debug_assertions))]
        let layer_name_ptrs: Vec<*const i8> = Vec::new();

        #[cfg(debug_assertions)]
        {
            unsafe {
                let available_layers = entry.enumerate_instance_layer_properties()?;
                let validation_layer_name = CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap();
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

        let swapchain = Swapchain::new(
            &instance,
            &device,
            physical_device,
            &surface,
            &surface_loader,
            window)?;
        let swapchain_loader= swapchain::Device::new(&instance, &device);

        // Create per-swapchain-image semaphores and fences:
        let image_count = swapchain.swapchain_image_views.len();
        let mut image_available_semaphores = Vec::with_capacity(image_count);
        let mut render_finished_semaphores = Vec::with_capacity(image_count);
        let mut in_flight_fences = Vec::with_capacity(image_count);
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };
        for _ in 0..image_count {
            unsafe {
                image_available_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
                render_finished_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
                in_flight_fences.push(device.create_fence(&fence_info, None)?);
            }
        }

        let command_buffers = Self::allocate_command_buffers(
            &device,
            command_pool,
            vk::CommandBufferLevel::PRIMARY,
            swapchain.swapchain_image_views.len())?;
        let secondary_command_pool = Self::create_command_pool(&device, graphics_queue_family_index)?;
        let secondary_command_buffers = Self::allocate_command_buffers(
            &device,
            secondary_command_pool,
            vk::CommandBufferLevel::SECONDARY,
            swapchain.swapchain_image_views.len()
        )?;

        let vulkan_base = Self {
            instance,
            physical_device,
            device,
            graphics_queue,
            surface,
            surface_loader,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            swapchain,
            swapchain_loader,
            secondary_command_pool,
            secondary_command_buffers,
            debug_settings,
            #[cfg(debug_assertions)]
            debug_messenger,
            #[cfg(debug_assertions)]
            debug_utils_loader,
            current_frame: 0,
        };

        // Pre-record each secondary command buffer (no-op) so they're executable
        for idx in 0..vulkan_base.secondary_command_buffers.len() {
            vulkan_base
                .record_secondary_command_buffer(idx, |_, _| {})
                .expect("Failed to pre-record secondary command buffer");
        }
        // The proper secondary command buffer is recorded in the draw_frame method

        vulkan_base.record_command_buffers()?;

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
            for &buffer in &self.secondary_command_buffers {
                self.device.free_command_buffers(self.secondary_command_pool, &[buffer]);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_command_pool(self.secondary_command_pool, None);
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