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
    pub extent: vk::Extent2D,
    pub color_format: vk::Format,
    pub depth_format: vk::Format,
    pub color_msaa_image: vk::Image,
    pub color_msaa_image_view: vk::ImageView,
    color_msaa_allocation: Option<Allocation>,
    pub depth_msaa_image: vk::Image,
    pub depth_msaa_image_view: vk::ImageView,
    depth_msaa_allocation: Option<Allocation>,
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
    // Swapchain creation necessarily joins handles owned by the Vulkan instance,
    // device, surface, window, and allocator.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
        surface_loader: &surface::Instance,
        window: &Window,
        allocator: &Allocator,
        msaa_samples: u32,
    ) -> Result<Self, Box<dyn Error>> {
        Self::new_with_old_swapchain(
            instance,
            device,
            physical_device,
            surface,
            surface_loader,
            window,
            allocator,
            msaa_samples,
            vk::SwapchainKHR::null(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_old_swapchain(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
        surface_loader: &surface::Instance,
        window: &Window,
        allocator: &Allocator,
        msaa_samples: u32,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<Self, Box<dyn Error>> {
        let swapchain_support =
            SwapchainSupportDetails::query(physical_device, *surface, surface_loader)?;

        let surface_format = Self::choose_swap_surface_format(&swapchain_support.formats)?;
        let present_mode = Self::choose_swap_present_mode(&swapchain_support.present_modes);
        let extent = Self::choose_swap_extent(&swapchain_support.capabilities, window);
        let depth_format = vk::Format::D32_SFLOAT;

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
            composite_alpha: Self::choose_composite_alpha(
                swapchain_support.capabilities.supported_composite_alpha,
            )?,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain,
            ..Default::default()
        };

        let swapchain_loader = swapchain::Device::new(instance, device);
        let handle = unsafe { swapchain_loader.create_swapchain(&create_info, None)? };

        // Keep ownership recorded as each fallible step succeeds, so any later
        // failure can release exactly the resources that were created.
        let mut swapchain = Self {
            handle,
            images: Vec::new(),
            swapchain_image_views: Vec::new(),
            extent,
            color_format: surface_format.format,
            depth_format,
            color_msaa_image: vk::Image::null(),
            color_msaa_image_view: vk::ImageView::null(),
            color_msaa_allocation: None,
            depth_msaa_image: vk::Image::null(),
            depth_msaa_image_view: vk::ImageView::null(),
            depth_msaa_allocation: None,
        };
        let result = (|| -> Result<(), Box<dyn Error>> {
            swapchain.images = unsafe { swapchain_loader.get_swapchain_images(handle)? };
            let (image, allocation) = Self::create_msaa_image(
                allocator,
                extent,
                surface_format.format,
                msaa_samples,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            )?;
            swapchain.color_msaa_image = image;
            swapchain.color_msaa_allocation = Some(allocation);
            swapchain.color_msaa_image_view = Self::create_msaa_image_view(
                device,
                &image,
                surface_format.format,
                vk::ImageAspectFlags::COLOR,
            )?;
            let (image, allocation) = Self::create_msaa_image(
                allocator,
                extent,
                depth_format,
                msaa_samples,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            )?;
            swapchain.depth_msaa_image = image;
            swapchain.depth_msaa_allocation = Some(allocation);
            swapchain.depth_msaa_image_view = Self::create_msaa_image_view(
                device,
                &image,
                depth_format,
                vk::ImageAspectFlags::DEPTH,
            )?;
            swapchain.swapchain_image_views =
                Self::create_image_views(device, &swapchain.images, surface_format.format)?;
            Ok(())
        })();
        if let Err(error) = result {
            swapchain.cleanup(instance, device, allocator);
            return Err(error);
        }
        println!(
            "🖼️ Swapchain created with {} images",
            swapchain.images.len()
        );
        Ok(swapchain)
    }

    /// Recreates the swapchain and associated resources when the window is resized.
    /// This method waits for the device to be idle, creates the replacement,
    /// then cleans up existing resources. Vulkan retires the old swapchain when
    /// replacement creation is attempted; on failure it remains owned for
    /// cleanup, but the caller must stop rendering.
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
    #[allow(clippy::too_many_arguments)]
    pub fn recreate(
        &mut self,
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
        surface_loader: &surface::Instance,
        window: &Window,
        allocator: &Allocator,
        msaa_samples: u32,
    ) -> Result<(), Box<dyn Error>> {
        // Synchronize before destroying images or views referenced by the GPU.
        unsafe { device.device_wait_idle()? };
        let new_swapchain = Self::new_with_old_swapchain(
            instance,
            device,
            physical_device,
            surface,
            surface_loader,
            window,
            allocator,
            msaa_samples,
            self.handle,
        )?;
        self.cleanup(instance, device, allocator);
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
        // Taking allocations and clearing handles also makes cleanup safe to
        // repeat after an aborted recreation. The caller must ensure GPU work
        // that references these resources has completed.
        unsafe {
            for view in self.swapchain_image_views.drain(..) {
                device.destroy_image_view(view, None);
            }
            device.destroy_image_view(
                std::mem::replace(&mut self.color_msaa_image_view, vk::ImageView::null()),
                None,
            );
            if let Some(mut allocation) = self.color_msaa_allocation.take() {
                allocator.destroy_image(self.color_msaa_image, &mut allocation);
            }
            self.color_msaa_image = vk::Image::null();
            device.destroy_image_view(
                std::mem::replace(&mut self.depth_msaa_image_view, vk::ImageView::null()),
                None,
            );
            if let Some(mut allocation) = self.depth_msaa_allocation.take() {
                allocator.destroy_image(self.depth_msaa_image, &mut allocation);
            }
            self.depth_msaa_image = vk::Image::null();
            let swapchain_loader = swapchain::Device::new(instance, device);
            swapchain_loader.destroy_swapchain(
                std::mem::replace(&mut self.handle, vk::SwapchainKHR::null()),
                None,
            );
        }
        self.images.clear();
    }

    /// Chooses the best swap surface format from the available formats.
    /// # Arguments
    /// * `available_formats` - A slice of available surface formats.
    /// # Returns
    /// * `vk::SurfaceFormatKHR` - The chosen surface format.
    fn choose_swap_surface_format(
        available_formats: &[vk::SurfaceFormatKHR],
    ) -> Result<vk::SurfaceFormatKHR, vk::Result> {
        // Older implementations may report UNDEFINED to permit any format.
        if let [format] = available_formats
            && format.format == vk::Format::UNDEFINED
        {
            return Ok(vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: format.color_space,
            });
        }
        available_formats
            .iter()
            .copied()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| available_formats.first().copied())
            .ok_or(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)
    }

    fn choose_composite_alpha(
        supported: vk::CompositeAlphaFlagsKHR,
    ) -> Result<vk::CompositeAlphaFlagsKHR, vk::Result> {
        [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|mode| supported.contains(*mode))
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)
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

    /// Creates an MSAA image (color or depth).
    fn create_msaa_image(
        allocator: &Allocator,
        extent: vk::Extent2D,
        format: vk::Format,
        msaa_samples: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<(vk::Image, Allocation), Box<dyn Error>> {
        let image_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::from_raw(msaa_samples),
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (image, allocation) = unsafe { allocator.create_image(&image_info, &alloc_info)? };

        println!("🖼️ MSAA image created");
        Ok((image, allocation))
    }

    fn create_msaa_image_view(
        device: &ash::Device,
        image: &vk::Image,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
    ) -> Result<vk::ImageView, vk::Result> {
        let create_info = vk::ImageViewCreateInfo {
            image: *image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            components: vk::ComponentMapping::default(),
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };
        unsafe { device.create_image_view(&create_info, None) }
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
        let mut views = Vec::with_capacity(swapchain_images.len());
        for &image in swapchain_images {
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

            match unsafe { device.create_image_view(&create_info, None) } {
                Ok(view) => views.push(view),
                Err(error) => {
                    // collect::<Result<_, _>>() would drop only the Rust handles.
                    for view in views {
                        unsafe { device.destroy_image_view(view, None) };
                    }
                    return Err(error);
                }
            }
        }
        Ok(views)
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

#[cfg(test)]
mod tests {
    use super::Swapchain;
    use ash::vk;

    #[test]
    fn empty_surface_formats_return_an_error() {
        assert_eq!(
            Swapchain::choose_swap_surface_format(&[]),
            Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)
        );
    }

    #[test]
    fn surface_format_prefers_supported_bgra_and_preserves_fallback() {
        let fallback = vk::SurfaceFormatKHR {
            format: vk::Format::R8G8B8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let preferred = vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_UNORM,
            ..fallback
        };
        let chosen = Swapchain::choose_swap_surface_format(&[fallback, preferred]).unwrap();
        assert_eq!(chosen.format, preferred.format);
        let chosen = Swapchain::choose_swap_surface_format(&[fallback]).unwrap();
        assert_eq!(chosen.format, fallback.format);
        assert_eq!(chosen.color_space, fallback.color_space);
    }

    #[test]
    fn undefined_surface_format_allows_an_explicit_format() {
        let chosen = Swapchain::choose_swap_surface_format(&[vk::SurfaceFormatKHR {
            format: vk::Format::UNDEFINED,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        }])
        .unwrap();
        assert_eq!(chosen.format, vk::Format::B8G8R8A8_UNORM);
        assert_eq!(chosen.color_space, vk::ColorSpaceKHR::SRGB_NONLINEAR);
    }

    #[test]
    fn composite_alpha_uses_only_a_supported_mode() {
        let transparent =
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED | vk::CompositeAlphaFlagsKHR::INHERIT;
        assert_eq!(
            Swapchain::choose_composite_alpha(transparent),
            Ok(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
        );
        assert_eq!(
            Swapchain::choose_composite_alpha(transparent | vk::CompositeAlphaFlagsKHR::OPAQUE),
            Ok(vk::CompositeAlphaFlagsKHR::OPAQUE)
        );
        assert!(Swapchain::choose_composite_alpha(vk::CompositeAlphaFlagsKHR::empty()).is_err());
    }

    #[test]
    fn present_mode_prefers_mailbox_and_falls_back_to_fifo() {
        assert_eq!(
            Swapchain::choose_swap_present_mode(&[vk::PresentModeKHR::FIFO]),
            vk::PresentModeKHR::FIFO
        );
        assert_eq!(
            Swapchain::choose_swap_present_mode(&[
                vk::PresentModeKHR::FIFO,
                vk::PresentModeKHR::MAILBOX,
            ]),
            vk::PresentModeKHR::MAILBOX
        );
    }
}
