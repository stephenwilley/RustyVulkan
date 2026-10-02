use ash::vk;
use std::collections::HashMap;
use vk_mem::{Alloc, Allocation, AllocationCreateInfo, Allocator, MemoryUsage};

use crate::graphics::shadow_math::SHADOW_CASCADE_COUNT;

/// Distinct attachment usages the graph can create and manage.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum AttachmentKind {
    // Off-screen targets managed by AttachmentManager
    Color,
    Depth,
    Shadow,
    // Special targets managed by the RenderGraph
    SwapchainColor,
    MsaaColor,
    MsaaDepth,
}

/// Request describing an attachment to retrieve or create.
#[derive(Clone, Copy, Debug)]
pub struct AttachmentRequest {
    pub kind: AttachmentKind,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub samples: vk::SampleCountFlags,
}

impl AttachmentRequest {
    /// Creates a new request with unspecified format/extent and 1x sampling.
    pub fn new(kind: AttachmentKind) -> Self {
        Self {
            kind,
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D {
                width: 0,
                height: 0,
            },
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }
}

/// Lightweight handle for an attachment's VkImage and VkImageView.
#[derive(Clone, Copy, Debug)]
pub struct AttachmentHandle {
    pub image: vk::Image,
    /// Whole image view (a 2D array for cascaded shadows).
    pub view: vk::ImageView,
    /// Individual views used while rendering each shadow-map layer.
    pub layer_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
}

struct AttachmentInternal {
    handle: AttachmentHandle,
    allocation: Allocation,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct AttachmentKey {
    kind: AttachmentKind,
    format: vk::Format,
    width: u32,
    height: u32,
    samples: vk::SampleCountFlags,
}

impl From<AttachmentRequest> for AttachmentKey {
    fn from(req: AttachmentRequest) -> Self {
        Self {
            kind: req.kind,
            format: req.format,
            width: req.extent.width,
            height: req.extent.height,
            samples: req.samples,
        }
    }
}

/// Manages per-swapchain-image transient attachments (color/depth/shadow).
pub struct AttachmentManager {
    image_count: usize,
    attachments: HashMap<AttachmentKey, Vec<AttachmentInternal>>,
}

impl AttachmentManager {
    /// Creates a new manager sized for `image_count` swapchain images.
    pub fn new(image_count: usize) -> Self {
        Self {
            image_count,
            attachments: HashMap::new(),
        }
    }

    /// Gets or lazily creates an attachment matching the request for a given image index.
    pub fn get_attachment(
        &mut self,
        device: &ash::Device,
        allocator: &Allocator,
        image_index: usize,
        request: AttachmentRequest,
    ) -> AttachmentHandle {
        let key = AttachmentKey::from(request);
        if !self.attachments.contains_key(&key) {
            self.create_attachments(device, allocator, &key);
        }
        self.attachments[&key][image_index].handle
    }

    fn create_attachments(
        &mut self,
        device: &ash::Device,
        allocator: &Allocator,
        key: &AttachmentKey,
    ) {
        let mut per_image = Vec::with_capacity(self.image_count);
        for _ in 0..self.image_count {
            let (handle, allocation) = Self::create_single(device, allocator, key);
            per_image.push(AttachmentInternal { handle, allocation });
        }
        self.attachments.insert(*key, per_image);
    }

    fn create_single(
        device: &ash::Device,
        allocator: &Allocator,
        key: &AttachmentKey,
    ) -> (AttachmentHandle, Allocation) {
        let usage = match key.kind {
            AttachmentKind::Color => {
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
            }
            AttachmentKind::Depth => {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
            }
            AttachmentKind::Shadow => {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED
            }
            _ => panic!("Unsupported attachment kind for AttachmentManager"),
        };
        let aspect = match key.kind {
            AttachmentKind::Color => vk::ImageAspectFlags::COLOR,
            AttachmentKind::Depth | AttachmentKind::Shadow => vk::ImageAspectFlags::DEPTH,
            _ => panic!("Unsupported attachment kind for AttachmentManager"),
        };
        let layer_count = if key.kind == AttachmentKind::Shadow {
            SHADOW_CASCADE_COUNT as u32
        } else {
            1
        };
        let image_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: key.format,
            extent: vk::Extent3D {
                width: key.width,
                height: key.height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: layer_count,
            samples: key.samples,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let alloc_info = AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        let (image, allocation) = unsafe {
            allocator
                .create_image(&image_info, &alloc_info)
                .expect("create attachment")
        };
        let view_info = vk::ImageViewCreateInfo {
            image,
            view_type: if layer_count > 1 {
                vk::ImageViewType::TYPE_2D_ARRAY
            } else {
                vk::ImageViewType::TYPE_2D
            },
            format: key.format,
            components: vk::ComponentMapping::default(),
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count,
            },
            ..Default::default()
        };
        let view = unsafe {
            device
                .create_image_view(&view_info, None)
                .expect("create view")
        };
        let mut layer_views = [vk::ImageView::null(); SHADOW_CASCADE_COUNT];
        if key.kind == AttachmentKind::Shadow {
            for (layer, layer_view) in layer_views.iter_mut().enumerate() {
                let layer_info = vk::ImageViewCreateInfo {
                    view_type: vk::ImageViewType::TYPE_2D,
                    subresource_range: vk::ImageSubresourceRange {
                        base_array_layer: layer as u32,
                        layer_count: 1,
                        ..view_info.subresource_range
                    },
                    ..view_info
                };
                *layer_view = unsafe {
                    device
                        .create_image_view(&layer_info, None)
                        .expect("create shadow cascade view")
                };
            }
        }
        (
            AttachmentHandle {
                image,
                view,
                layer_views,
            },
            allocation,
        )
    }

    /// Destroys all created attachments and clears internal storage.
    pub fn cleanup(&mut self, device: &ash::Device, allocator: &Allocator) {
        for (_key, mut vec) in self.attachments.drain() {
            for mut att in vec.drain(..) {
                unsafe {
                    for view in att.handle.layer_views {
                        if view != vk::ImageView::null() {
                            device.destroy_image_view(view, None);
                        }
                    }
                    device.destroy_image_view(att.handle.view, None);
                    allocator.destroy_image(att.handle.image, &mut att.allocation);
                }
            }
        }
    }
}
