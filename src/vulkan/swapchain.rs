//! --------------------------------------------------------------------------------------
//! Swapchain Module (swapchain.rs)
//!
//! Created: July 2025  
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module defines the `Swapchain` type, which encapsulates all Vulkan swapchain
//! operations and their dependent resources. It handles:
//!   • Querying physical device swapchain support (formats, present modes, capabilities)  
//!   • Choosing the best surface format, present mode, and swap extent for the window  
//!   • Creating the Vulkan swapchain, image views, render pass, graphics pipeline, and framebuffers  
//!   • Recreating all those resources cleanly when the window is resized  
//!   • Performing explicit manual cleanup of swapchain, image views, framebuffers, render pass, and pipeline  
//!
//! Usage:
//!   1. Call `Swapchain::new(…)` during initialization.  
//!   2. On window resize, call `swapchain.recreate(…)`.  
//!   3. Before shutdown (or prior to recreation), call `swapchain.cleanup(…)`.  
//! --------------------------------------------------------------------------------------

use ash::Instance;
use ash::vk;
use ash::khr::surface;
use ash::khr::swapchain;
use winit::window::Window;
use std::error::Error;

/// Represents the Vulkan swapchain and associated resources
/// including image views, framebuffers, and render pass.
/// It handles swapchain creation, recreation, and cleanup.
pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub depth_image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub render_pass: vk::RenderPass,
    pub extent: vk::Extent2D,
    pub depth_images: Vec<vk::Image>,
    pub depth_memories: Vec<vk::DeviceMemory>,
}

impl Swapchain {
    /// Creates a new `Swapchain` instance, initializing the swapchain, image views, render pass,
    /// graphics pipeline, and framebuffers.
    /// # Arguments
    /// * `instance` - The Vulkan `Instance` to use for creating the swapchain.
    /// * `device` - The Vulkan logical device to use for creating resources.
    /// * `physical_device` - The physical device to query capabilities and formats.
    /// * `surface` - The Vulkan surface to associate with the swapchain.
    /// * `surface_loader` - The surface loader to manage the surface.
    /// * `pipeline` - The `Pipeline` containing shader modules and layout for the graphics pipeline.
    /// * `window` - The winit `Window` to determine swapchain extent.
    /// # Returns
    /// * `Result<Self, Box<dyn Error>>` - Returns the initialized `Swapchain` on success, or an error on failure.
    pub fn new(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
        surface_loader: &surface::Instance,
        window: &Window,
    ) -> Result<Self, Box<dyn Error>> {
        let swapchain_support = SwapchainSupportDetails::query(physical_device, *surface, surface_loader)?;

        let surface_format = Self::choose_swap_surface_format(&swapchain_support.formats);
        let present_mode = Self::choose_swap_present_mode(&swapchain_support.present_modes);
        let extent = Self::choose_swap_extent(&swapchain_support.capabilities, window);

        let mut image_count = 3;
        if image_count < swapchain_support.capabilities.min_image_count {
            image_count = swapchain_support.capabilities.min_image_count;
        }
        if swapchain_support.capabilities.max_image_count > 0 && image_count > swapchain_support.capabilities.max_image_count {
            image_count = swapchain_support.capabilities.max_image_count;
        }

        let create_info = vk::SwapchainCreateInfoKHR {
            surface: *surface,
            min_image_count: image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            ..Default::default()
        };

        let swapchain_loader = swapchain::Device::new(instance, device);
        let handle = unsafe { swapchain_loader.create_swapchain(&create_info, None)? };

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(handle)? };
        println!("🖼️ Swapchain created with {} images", swapchain_images.len());

        let depth_images = Self::create_depth_images(device, &swapchain_images, extent)?;
        let depth_memories = Self::create_depth_memories(device, instance, physical_device, &depth_images)?;
        let depth_image_views = Self::create_depth_image_views(device, &depth_images)?;

        let swapchain_image_views = Self::create_image_views(device, &swapchain_images, surface_format.format)?;
        let render_pass = Self::create_render_pass(device, surface_format.format)?;

        let framebuffers = Self::create_framebuffers(device, render_pass, &swapchain_image_views, &depth_image_views, extent)?;

        Ok(Self {
            handle,
            swapchain_image_views,
            depth_image_views,
            framebuffers,
            render_pass,
            extent,
            depth_images,
            depth_memories,
        })
    }

    /// Recreates the swapchain and associated resources when the window is resized.
    /// This method waits for the device to be idle, cleans up existing resources,
    /// and creates a new swapchain with the updated parameters.
    /// # Arguments
    /// * `instance` - The Vulkan `Instance` to use for creating the swapchain
    /// * `device` - The Vulkan logical device to use for creating resources
    /// * `physical_device` - The physical device to query capabilities and formats
    /// * `surface` - The Vulkan surface to associate with the swapchain
    /// * `surface_loader` - The surface loader to manage the surface
    /// * `pipeline` - The `Pipeline` containing shader modules and layout for the graphics
    /// * `window` - The winit `Window` to determine swapchain extent
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok on success, or
    ///   an error if the swapchain could not be recreated.
    pub fn recreate(
        &mut self,
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
        surface_loader: &surface::Instance,
        window: &Window) -> Result<(), Box<dyn Error>> {
        unsafe {
            device.device_wait_idle()?;
        }
        self.cleanup(instance, device);

        let new_swapchain = Swapchain::new(instance, device, physical_device, surface, surface_loader, window)?;
        *self = new_swapchain;

        println!("🔄 Swapchain recreated successfully");
        Ok(())
    }

    /// Cleans up the swapchain and associated resources.
    /// This method destroys the swapchain, image views, framebuffers, and render pass.
    /// It should be called when the swapchain is no longer needed,
    /// such as when the application is shutting down or when the swapchain is being recreated.
    /// # Arguments
    /// * `instance` - The Vulkan `Instance` to use for destroying the swapchain
    /// * `device` - The Vulkan logical device to use for destroying resources
    pub fn cleanup(&mut self, instance: &Instance, device: &ash::Device) {
        unsafe {
            device.destroy_render_pass(self.render_pass, None);
            for &framebuffer in &self.framebuffers {
                device.destroy_framebuffer(framebuffer, None);
            }
            for &view in &self.swapchain_image_views {
                device.destroy_image_view(view, None);
            }
            for &dv in &self.depth_image_views {
                device.destroy_image_view(dv, None);
            }
            for &image in &self.depth_images {
                device.destroy_image(image, None);
            }
            for &memory in &self.depth_memories {
                device.free_memory(memory, None);
            }
            let swapchain_loader = swapchain::Device::new(instance, device);
            swapchain_loader.destroy_swapchain(self.handle, None);
        }
    }

    fn choose_swap_surface_format(
        available_formats: &[vk::SurfaceFormatKHR]
    ) -> vk::SurfaceFormatKHR {
        available_formats
            .iter()
            .cloned()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .unwrap_or_else(|| available_formats[0])
    }

    fn choose_swap_present_mode(
        available_present_modes: &[vk::PresentModeKHR]
    ) -> vk::PresentModeKHR {
        if available_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        }
    }

    fn choose_swap_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        window: &Window
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let (width, height): (u32, u32) = window.inner_size().into();
            vk::Extent2D {
                width: width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }

    fn create_depth_images(
        device: &ash::Device,
        swapchain_images: &[vk::Image],
        extent: vk::Extent2D,
    ) -> Result<Vec<vk::Image>, vk::Result> {
        let depth_image_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::D32_SFLOAT,
            extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let mut depth_images = Vec::new();
        for _ in swapchain_images {
            let depth_image = unsafe { device.create_image(&depth_image_info, None)? };
            depth_images.push(depth_image);
        }

        println!("🖼️ Depth images created for swapchain");
        Ok(depth_images)
    }

    fn create_depth_memories(
        device: &ash::Device,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        depth_images: &[vk::Image],
    ) -> Result<Vec<vk::DeviceMemory>, vk::Result> {
        let mut depth_memories = Vec::new();
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        for &image in depth_images {
            let memory_requirements = unsafe { device.get_image_memory_requirements(image) };
            let mem_type_index = (0 .. mem_props.memory_type_count)
                .find(|&i| {
                    (memory_requirements.memory_type_bits & (1 << i)) != 0 &&
                    mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .expect("No suitable memory type!");

            let allocate_info = vk::MemoryAllocateInfo {
                allocation_size: memory_requirements.size,
                memory_type_index: mem_type_index,
                ..Default::default()
            };

            let memory = unsafe { device.allocate_memory(&allocate_info, None)? };
            unsafe { device.bind_image_memory(image, memory, 0)? };
            depth_memories.push(memory);
        }

        println!("🖼️ Depth memories allocated for swapchain images");
        Ok(depth_memories)
    }

    fn create_depth_image_views(
        device: &ash::Device,
        depth_images: &[vk::Image],
    ) -> Result<Vec<vk::ImageView>, vk::Result> {
        let mut depth_image_views = Vec::new();
        for &image in depth_images {
            let view_info = vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: vk::Format::D32_SFLOAT,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };
            let view = unsafe { device.create_image_view(&view_info, None)? };
            depth_image_views.push(view);
        }
        println!("🖼️ Depth image views created for swapchain images");
        Ok(depth_image_views)
    }

    fn create_image_views(
        device: &ash::Device,
        swapchain_images: &[vk::Image],
        swapchain_format: vk::Format,
    ) -> Result<Vec<vk::ImageView>, vk::Result> {
        swapchain_images.iter().map(|&image| {
            let create_info = vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: swapchain_format,
                components: vk::ComponentMapping::default(),
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            unsafe { device.create_image_view(&create_info, None) }
        }).collect()
    }

    fn create_render_pass(
        device: &ash::Device,
        swapchain_format: vk::Format,
    ) -> Result<vk::RenderPass, vk::Result> {
        let color_attachment = vk::AttachmentDescription {
            format: swapchain_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_attachment = vk::AttachmentDescription {
            format: vk::Format::D32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };

        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let attachments = [color_attachment, depth_attachment];

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_depth_stencil_attachment: &depth_attachment_ref,
            ..Default::default()
        };

        let dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let render_pass_info = vk::RenderPassCreateInfo {
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 1,
            p_dependencies: &dependency,
            ..Default::default()
        };

        let render_pass = unsafe { device.create_render_pass(&render_pass_info, None)? };
        println!("🖌️ Render pass created");
        Ok(render_pass)
    }

    fn create_framebuffers(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        image_views: &[vk::ImageView],
        depth_image_views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Result<Vec<vk::Framebuffer>, vk::Result> {
        let framebuffers = image_views.iter().zip(depth_image_views.iter())
            .map(|(&view, &depth_view)| {
            let attachments = [view, depth_view];
            let info = vk::FramebufferCreateInfo {
                render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments:    attachments.as_ptr(),
                width:            extent.width,
                height:           extent.height,
                layers:           1,
                ..Default::default()
            };
            unsafe { device.create_framebuffer(&info, None) }
        }).collect::<Result<Vec<_>, _>>()?;
        println!("📦 Framebuffers created for each swapchain image view");
        Ok(framebuffers)
    }
}

/// Represents the details of swapchain support for a physical device.
/// This struct contains the surface capabilities, available formats, and present modes.
/// It provides a method to query these details from the Vulkan API.
struct SwapchainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    /// Queries the swapchain support details for a given physical device and surface.
    /// This method retrieves the surface capabilities, available formats, and present modes
    /// from the Vulkan API and returns a `SwapchainSupportDetails` instance.
    /// # Arguments
    /// * `physical_device` - The physical device to query for swapchain support.
    /// * `surface` - The surface to query for swapchain support.
    /// * `surface_loader` - The surface loader to manage the surface.
    /// # Returns
    /// * `Result<Self, vk::Result>` - Returns a `SwapchainSupportDetails` instance on success,
    ///   or an error if the query fails.
    pub fn query(
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        surface_loader: &surface::Instance,
    ) -> Result<Self, vk::Result> {
        let capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)?
        };

        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)?
        };

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)?
        };

        println!(
            "📦 Swapchain support: {} formats, {} present modes",
            formats.len(),
            present_modes.len()
        );

        Ok(Self {
            capabilities,
            formats,
            present_modes,
        })
    }
}