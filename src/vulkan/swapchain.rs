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
//!   • Creating the Vulkan swapchain and image/depth views
//!   • Recreating all those resources cleanly when the window is resized
//!   • Performing explicit manual cleanup of swapchain and associated views
//!
//! Usage:
//!   1. Call `Swapchain::new(…)` during initialization.  
//!   2. On window resize, call `swapchain.recreate(…)`.  
//!   3. Before shutdown (or prior to recreation), call `swapchain.cleanup(…)`.  
//! --------------------------------------------------------------------------------------

use ash::Instance;
use ash::khr::surface;
use ash::khr::swapchain;
use ash::vk;
use std::error::Error;
use vk_mem::{Alloc, Allocation, Allocator, MemoryUsage};
use winit::window::Window;

/// Represents the Vulkan swapchain and associated resources
/// including image views and depth images.
/// It handles swapchain creation, recreation, and cleanup.
pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub depth_image_views: Vec<vk::ImageView>,
    pub extent: vk::Extent2D,
    pub color_format: vk::Format,
    pub depth_format: vk::Format,
    pub depth_images: Vec<vk::Image>,
    pub depth_allocations: Vec<Allocation>,
    pub image_layouts: Vec<vk::ImageLayout>,
    pub depth_layouts: Vec<vk::ImageLayout>,
}

impl Swapchain {
    /// Creates a new `Swapchain` instance, initializing the swapchain and image/depth views.
    /// # Arguments
    /// * `instance` - The Vulkan `Instance` to use for creating the swapchain.
    /// * `device` - The Vulkan logical device to use for creating resources.
    /// * `physical_device` - The physical device to query capabilities and formats.
    /// * `surface` - The Vulkan surface to associate with the swapchain.
    /// * `surface_loader` - The surface loader to manage the surface.
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
        allocator: &Allocator,
    ) -> Result<Self, Box<dyn Error>> {
        let swapchain_support =
            SwapchainSupportDetails::query(physical_device, *surface, surface_loader)?;

        let surface_format = Self::choose_swap_surface_format(&swapchain_support.formats);
        let present_mode = Self::choose_swap_present_mode(&swapchain_support.present_modes);
        let extent = Self::choose_swap_extent(&swapchain_support.capabilities, window);

        let mut image_count = 3;
        if image_count < swapchain_support.capabilities.min_image_count {
            image_count = swapchain_support.capabilities.min_image_count;
        }
        if swapchain_support.capabilities.max_image_count > 0
            && image_count > swapchain_support.capabilities.max_image_count
        {
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
        let image_count = swapchain_images.len();
        println!("🖼️ Swapchain created with {} images", image_count);

        let (depth_images, depth_allocations) =
            Self::create_depth_images(allocator, &swapchain_images, extent)?;
        let depth_image_views = Self::create_depth_image_views(device, &depth_images)?;

        let swapchain_image_views =
            Self::create_image_views(device, &swapchain_images, surface_format.format)?;

        Ok(Self {
            handle,
            images: swapchain_images,
            swapchain_image_views,
            depth_image_views,
            extent,
            color_format: surface_format.format,
            depth_format: vk::Format::D32_SFLOAT,
            depth_images,
            depth_allocations,
            image_layouts: vec![vk::ImageLayout::UNDEFINED; image_count],
            depth_layouts: vec![vk::ImageLayout::UNDEFINED; image_count],
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
        window: &Window,
        allocator: &Allocator,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            device.device_wait_idle()?;
        }
        self.cleanup(instance, device, allocator);

        let new_swapchain = Swapchain::new(
            instance,
            device,
            physical_device,
            surface,
            surface_loader,
            window,
            allocator,
        )?;
        *self = new_swapchain;

        println!("🔄 Swapchain recreated successfully");
        Ok(())
    }

    /// Cleans up the swapchain and associated resources.
    /// This method destroys the swapchain, image views, and depth resources.
    /// It should be called when the swapchain is no longer needed,
    /// such as when the application is shutting down or when the swapchain is being recreated.
    /// # Arguments
    /// * `instance` - The Vulkan `Instance` to use for destroying the swapchain
    /// * `device` - The Vulkan logical device to use for destroying resources
    pub fn cleanup(&mut self, instance: &Instance, device: &ash::Device, allocator: &Allocator) {
        unsafe {
            for &view in &self.swapchain_image_views {
                device.destroy_image_view(view, None);
            }
            for &dv in &self.depth_image_views {
                device.destroy_image_view(dv, None);
            }
            for (image, allocation) in self
                .depth_images
                .iter()
                .zip(self.depth_allocations.iter_mut())
            {
                allocator.destroy_image(*image, allocation);
            }
            let swapchain_loader = swapchain::Device::new(instance, device);
            swapchain_loader.destroy_swapchain(self.handle, None);
        }
    }

    /// Chooses the best swap surface format from the available formats.
    /// # Arguments
    /// * `available_formats` - A slice of available surface formats.
    /// # Returns
    /// * `vk::SurfaceFormatKHR` - The chosen surface format.
    fn choose_swap_surface_format(
        available_formats: &[vk::SurfaceFormatKHR],
    ) -> vk::SurfaceFormatKHR {
        available_formats
            .iter()
            .cloned()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or_else(|| available_formats[0])
    }

    /// Chooses the best swap present mode from the available present modes.
    /// Prioritizes `MAILBOX` for low-latency, falls back to `FIFO` (V-Sync).
    /// # Arguments
    /// * `available_present_modes` - A slice of available present modes.
    /// # Returns
    /// * `vk::PresentModeKHR` - The chosen present mode.
    fn choose_swap_present_mode(
        available_present_modes: &[vk::PresentModeKHR],
    ) -> vk::PresentModeKHR {
        if available_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        }
    }

    /// Chooses the swap extent (resolution) for the swapchain.
    /// Uses the current extent if available, otherwise clamps to window size.
    /// # Arguments
    /// * `capabilities` - The surface capabilities.
    /// * `window` - The winit `Window`.
    /// # Returns
    /// * `vk::Extent2D` - The chosen swap extent.
    fn choose_swap_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        window: &Window,
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

    /// Creates depth images for the swapchain.
    /// # Arguments
    /// * `device` - The Vulkan logical device.
    /// * `swapchain_images` - The swapchain images.
    /// * `extent` - The extent of the swapchain.
    /// # Returns
    /// * `Result<Vec<vk::Image>, vk::Result>` - A vector of created depth images on success, or a Vulkan error on failure.
    fn create_depth_images(
        allocator: &Allocator,
        swapchain_images: &[vk::Image],
        extent: vk::Extent2D,
    ) -> Result<(Vec<vk::Image>, Vec<Allocation>), Box<dyn Error>> {
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
        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let mut depth_images = Vec::new();
        let mut depth_allocations = Vec::new();
        for _ in swapchain_images {
            let (image, allocation) =
                unsafe { allocator.create_image(&depth_image_info, &alloc_info)? };
            depth_images.push(image);
            depth_allocations.push(allocation);
        }

        println!("🖼️ Depth images created for swapchain");
        Ok((depth_images, depth_allocations))
    }

    /// Creates image views for depth images.
    /// # Arguments
    /// * `device` - The Vulkan logical device.
    /// * `depth_images` - The depth images to create views for.
    /// # Returns
    /// * `Result<Vec<vk::ImageView>, vk::Result>` - A vector of created depth image views on success, or a Vulkan error on failure.
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

    /// Creates image views for swapchain images.
    /// # Arguments
    /// * `device` - The Vulkan logical device.
    /// * `swapchain_images` - The swapchain images to create views for.
    /// * `swapchain_format` - The format of the swapchain images.
    /// # Returns
    /// * `Result<Vec<vk::ImageView>, vk::Result>` - A vector of created image views on success, or a Vulkan error on failure.
    fn create_image_views(
        device: &ash::Device,
        swapchain_images: &[vk::Image],
        swapchain_format: vk::Format,
    ) -> Result<Vec<vk::ImageView>, vk::Result> {
        swapchain_images
            .iter()
            .map(|&image| {
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
            })
            .collect()
    }

    // Render passes are no longer needed with dynamic rendering.

    // Framebuffers are no longer needed with dynamic rendering.
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
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };

        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
        };

        let present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
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
