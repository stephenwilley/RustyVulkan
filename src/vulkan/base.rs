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
use crate::graphics::pipeline::Pipeline;
use crate::graphics::shaders::load_default_stages;

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
    graphics_queue: vk::Queue,
    surface: vk::SurfaceKHR,
    surface_loader: surface::Instance,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
    pub swapchain: Swapchain,
    swapchain_loader: swapchain::Device,
    pub pipeline: Pipeline,
    secondary_command_pool: vk::CommandPool,
    secondary_command_buffer: vk::CommandBuffer,
    debug_settings: EngineDebugSettings,
}

impl VulkanBase {
    // Creates a new `VulkanBase` instance, initializing Vulkan resources and setting up the swapchain.
    fn create_instance(entry: &Entry, event_loop: &ActiveEventLoop) -> Result<Instance, Box<dyn Error>> {
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

        // Add portability enumeration so macOS MoltenVK gets picked up:
        extension_ptrs.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());

        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
            p_application_info: &app_info,
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
            p_queue_create_infos: &queue_info,
            queue_create_info_count: 1,
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: device_extensions.as_ptr(),
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

    fn record_command_buffers(
        &self,
        render_pass: vk::RenderPass,
        framebuffers: &[vk::Framebuffer],
        extent: vk::Extent2D,
        graphics_pipeline: vk::Pipeline,
    ) -> Result<(), vk::Result> {
        for (index, &command_buffer) in self.command_buffers.iter().enumerate() {
            let framebuffer = framebuffers[index];
            let begin_info = vk::CommandBufferBeginInfo::default();
            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
            },
            vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            }];
            let render_pass_info = vk::RenderPassBeginInfo {
                render_pass,
                framebuffer,
                render_area: vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                },
                clear_value_count: clear_values.len() as u32,
                p_clear_values: clear_values.as_ptr(),
                ..Default::default()
            };
            unsafe {
                self.device.begin_command_buffer(command_buffer, &begin_info)?;
                self.device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_info,
                    vk::SubpassContents::INLINE,
                );
                self.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    graphics_pipeline,
                );
                self.device.cmd_execute_commands(command_buffer, &[self.secondary_command_buffer]);
                self.device.cmd_end_render_pass(command_buffer);
                self.device.end_command_buffer(command_buffer)?;
            }
            println!("📜 Recorded command buffer {}", index);
        }
        Ok(())
    }

    fn create_sync_objects(device: &ash::Device) -> Result<(vk::Semaphore, vk::Semaphore, vk::Fence), vk::Result> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let image_available_semaphore = unsafe { device.create_semaphore(&semaphore_info, None)? };
        let render_finished_semaphore = unsafe { device.create_semaphore(&semaphore_info, None)? };
        let in_flight_fence = unsafe { device.create_fence(&fence_info, None)? };

        println!("⛓️ Sync objects created");
        Ok((image_available_semaphore, render_finished_semaphore, in_flight_fence))
    }

    fn record_secondary_command_buffer<F>(
        &self,
        image_index: usize,
        mut record_fn: F) -> Result<(), vk::Result> 
    where F: FnMut(vk::CommandBuffer, &ash::Device), {
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
            self.device.reset_command_buffer(self.secondary_command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(self.secondary_command_buffer, &begin_info)?;

            record_fn(self.secondary_command_buffer, &self.device);

            self.device.end_command_buffer(self.secondary_command_buffer)?;
        }

        Ok(())
    }

    /// Draws a frame by acquiring an image from the swapchain, submitting a command buffer,
    /// and presenting the image.
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or an error if the frame could not be drawn.
    pub fn draw_frame<F>(
        &mut self,
        mut record_secondary: F) -> Result<(), Box<dyn Error>>
    where F: FnMut(vk::CommandBuffer, &ash::Device), {
        unsafe {
            self.device.wait_for_fences(&[self.in_flight_fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.in_flight_fence])?;

            let (image_index, _is_suboptimal) = match self.swapchain_loader
                .acquire_next_image(
                    self.swapchain.handle,
                    u64::MAX,
                    self.image_available_semaphore,
                    vk::Fence::null(),
                ) {
                    Ok(result) => result,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        // Handling of out-of-date swapchain should be done in the event loop
                        return Ok(());
                    }
                    Err(e) => return Err(e.into()),
                };

            self.record_secondary_command_buffer(image_index as usize, |cmd_buf, device| {
                record_secondary(cmd_buf, device);
            })?;

            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let submit_info = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: &self.image_available_semaphore,
                p_wait_dst_stage_mask: wait_stages.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: &self.command_buffers[image_index as usize],
                signal_semaphore_count: 1,
                p_signal_semaphores: &self.render_finished_semaphore,
                ..Default::default()
            };

            self.device.queue_submit(
                self.graphics_queue,
                &[submit_info],
                self.in_flight_fence,
            )?;

            let present_info = vk::PresentInfoKHR {
                wait_semaphore_count: 1,
                p_wait_semaphores: &self.render_finished_semaphore,
                swapchain_count: 1,
                p_swapchains: &self.swapchain.handle,
                p_image_indices: &image_index,
                ..Default::default()
            };

            match self.swapchain_loader.queue_present(self.graphics_queue, &present_info) {
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    // Handling of out-of-date or suboptimal swapchain should be done in the event loop
                }
                Err(e) => return Err(e.into()),
                _ => {}
            }
        }
        Ok(())
    }

    /// Recreates the swapchain and associated resources when the window is resized.
    /// # Arguments
    /// * `window` - The winit `Window` to associate with the new swapchain.
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or an error if the swapchain could not be recreated.
    pub fn recreate_swapchain(&mut self, window: &Window, shader_stages: &[&crate::graphics::shaders::ShaderStageInfo],) -> Result<(), Box<dyn Error>> {
        self.swapchain.recreate(
            &self.instance,
            &self.device,
            self.physical_device,
            &self.surface,
            &self.surface_loader,
            &self.pipeline,
            window,
            shader_stages,
            self.debug_settings.wireframe,
        )?;
        self.command_buffers = Self::allocate_command_buffers(
            &self.device,
            self.command_pool,
            vk::CommandBufferLevel::PRIMARY,
            self.swapchain.swapchain_image_views.len(),
        )?;
        self.record_command_buffers(
            self.swapchain.render_pass,
            &self.swapchain.framebuffers,
            self.swapchain.extent,
            self.swapchain.graphics_pipeline,
        )?;
        // The secondary command buffer is recorded in the draw_frame method
        Ok(())
    }

    /// Swap out the old graphics pipeline *and* re-record all primary command buffers
    /// so they bind that new pipeline.
    pub fn recreate_pipeline_and_record(
        &mut self,
        shader_stages: &[&crate::graphics::shaders::ShaderStageInfo],
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe { self.device.device_wait_idle()?; }

        self.swapchain.recreate_pipeline(
            &self.device,
            &self.pipeline,
            shader_stages,
            self.debug_settings.wireframe,
        )?;

        self.record_command_buffers(
            self.swapchain.render_pass,
            &self.swapchain.framebuffers,
            self.swapchain.extent,
            self.swapchain.graphics_pipeline,
        )?;

        println!("🛠️ Pipeline re-created and command buffers re-recorded");

        Ok(())
    }

    /// Toggles the wireframe mode in the debug settings.
    pub fn toggle_wireframe(&mut self) {
        self.debug_settings.wireframe = !self.debug_settings.wireframe;
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
        };

        let entry = Entry::linked();
        let instance = Self::create_instance(&entry, event_loop)?;
        println!("🛡️ Vulkan Instance created");

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
        
        let pipeline = Pipeline::new(&device)?;

        let (image_available_semaphore, render_finished_semaphore, in_flight_fence) =
            Self::create_sync_objects(&device)?;

        let initial_shader_stages = load_default_stages(&device)?;
        let swapchain = Swapchain::new(
            &instance,
            &device,
            physical_device,
            &surface,
            &surface_loader,
            &pipeline, window,
            &[&initial_shader_stages[0], &initial_shader_stages[1]],
            false)?;
        let swapchain_loader= swapchain::Device::new(&instance, &device);

        let command_buffers = Self::allocate_command_buffers(&device, command_pool, vk::CommandBufferLevel::PRIMARY, swapchain.swapchain_image_views.len())?;
        let secondary_command_pool = Self::create_command_pool(&device, graphics_queue_family_index)?;
        let secondary_command_buffer = Self::allocate_command_buffers(&device, secondary_command_pool, vk::CommandBufferLevel::SECONDARY, 1)?[0];

        let vulkan_base = Self {
            instance,
            physical_device,
            device,
            graphics_queue,
            surface,
            surface_loader,
            command_pool,
            command_buffers,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
            swapchain,
            swapchain_loader,
            pipeline,
            secondary_command_pool,
            secondary_command_buffer,
            debug_settings,
        };

        vulkan_base.record_command_buffers(
            vulkan_base.swapchain.render_pass,
            &vulkan_base.swapchain.framebuffers,
            vulkan_base.swapchain.extent,
            vulkan_base.swapchain.graphics_pipeline,
        )?;
        // The secondary command buffer is recorded in the draw_frame method

        println!("✅ VulkanBase initialized successfully");
        Ok(vulkan_base)
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
            self.pipeline.cleanup(&self.device);

            // Then destroy the rest of the resources
            self.device.destroy_command_pool(self.secondary_command_pool, None);
            self.device.destroy_semaphore(self.image_available_semaphore, None);
            self.device.destroy_semaphore(self.render_finished_semaphore, None);
            self.device.destroy_fence(self.in_flight_fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Debug settings for the Vulkan engine
struct EngineDebugSettings {
    pub wireframe: bool,
}