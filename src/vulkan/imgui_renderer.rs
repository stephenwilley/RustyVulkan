//! --------------------------------------------------------------------------------------
//! ImGui Renderer Module (imgui_renderer.rs)
//!
//! Created: July 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module defines `ImGuiRenderer`, which encapsulates the Vulkan setup and rendering
//! logic for Dear ImGui. It handles:
//!   • Uploading the font atlas and creating associated Vulkan resources (image, view, sampler)
//!   • Creating the push-descriptor set layout for UI textures
//!   • Building the ImGui-specific graphics pipeline and pipeline layout
//!   • Recording draw commands for ImGui draw data into command buffers
//!   • Cleaning up all ImGui-related Vulkan resources on teardown
//!
//! Usage:
//!   1. `ImGuiRenderer::new(base: &mut VulkanBase, imgui: &mut imgui::Context) -> Result<Self, Box<dyn Error>>`
//!   2. `renderer.render(device, allocator, cmd_buf, draw_data, image_index)`
//!   3. `renderer.cleanup(allocator)`
//!
//! --------------------------------------------------------------------------------------

use crate::graphics::shaders::ShaderStageInfo;
use crate::graphics::shadow_math::SHADOW_CASCADE_COUNT;
use crate::vulkan::base::VulkanBase;
use ash::khr::push_descriptor;
use ash::vk;
use imgui::DrawData;
use imgui::DrawIdx;
use imgui::DrawVert;
use std::error::Error;
use vk_mem::{Alloc, Allocation, Allocator, MemoryUsage};

mod pipeline;
mod resources;

/// Host-visible ImGui geometry for one swapchain image.
#[derive(Default)]
struct UiFrameBuffers {
    vertex_buffer: vk::Buffer,
    vertex_allocation: Option<Allocation>,
    vertex_buffer_size: vk::DeviceSize,
    index_buffer: vk::Buffer,
    index_allocation: Option<Allocation>,
    index_buffer_size: vk::DeviceSize,
}

/// Renders ImGui UI elements using Vulkan.
pub struct ImGuiRenderer {
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pub vk_pipeline: vk::Pipeline,
    pub font_sampler: Option<vk::Sampler>,
    pub font_image: Option<vk::Image>,
    pub font_image_allocation: Option<Allocation>,
    pub font_image_view: Option<vk::ImageView>,
    /// One buffer pair per swapchain image: rewriting a single pair each frame would race
    /// with the previous frame, which may still be drawing from it.
    frame_buffers: Vec<UiFrameBuffers>,
    device: ash::Device,
    push_descriptor: push_descriptor::Device,
    /// Images that draw commands can show, indexed by `imgui::TextureId`; 0 is the font atlas.
    textures: Vec<vk::DescriptorImageInfo>,
    shadow_tex_ids: [Option<imgui::TextureId>; SHADOW_CASCADE_COUNT],
}

impl ImGuiRenderer {
    /// Ensures the vertex and index buffers are large enough and uploads ImGui draw data into them.
    /// # Arguments
    /// * `allocator` - The VMA allocator used to (re)create the buffers.
    /// * `draw_data` - The ImGui draw data.
    /// * `image_index` - Swapchain image being recorded; selects that image's buffer pair.
    pub fn update_buffers(
        &mut self,
        allocator: &Allocator,
        draw_data: &DrawData,
        image_index: usize,
    ) -> Result<(), vk::Result> {
        // Total vertex and index data sizes
        let vertex_size = (draw_data.total_vtx_count as usize * std::mem::size_of::<DrawVert>())
            as vk::DeviceSize;
        let index_size =
            (draw_data.total_idx_count as usize * std::mem::size_of::<DrawIdx>()) as vk::DeviceSize;
        if self.frame_buffers.len() <= image_index {
            self.frame_buffers
                .resize_with(image_index + 1, UiFrameBuffers::default);
        }
        // `begin_frame` has already waited on this image's previous submit, so its buffers
        // are idle and may be replaced or rewritten. Capacity grows geometrically to avoid
        // reallocating for small UI changes.
        let buffers = &mut self.frame_buffers[image_index];
        let grow_vertex_buffer = vertex_size > buffers.vertex_buffer_size;
        let grow_index_buffer = index_size > buffers.index_buffer_size;

        if grow_vertex_buffer {
            let capacity = vertex_size
                .max(buffers.vertex_buffer_size.saturating_mul(2))
                .max(1024);
            let buffer_info = vk::BufferCreateInfo {
                size: capacity,
                usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let alloc_info = vk_mem::AllocationCreateInfo {
                usage: MemoryUsage::AutoPreferHost,
                flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                    | vk_mem::AllocationCreateFlags::MAPPED,
                ..Default::default()
            };
            let (buf, alloc) = unsafe { allocator.create_buffer(&buffer_info, &alloc_info)? };
            if buffers.vertex_buffer != vk::Buffer::null()
                && let Some(allocation) = &mut buffers.vertex_allocation
            {
                unsafe {
                    allocator.destroy_buffer(buffers.vertex_buffer, allocation);
                }
            }
            buffers.vertex_buffer = buf;
            buffers.vertex_allocation = Some(alloc);
            buffers.vertex_buffer_size = capacity;
        }

        if grow_index_buffer {
            let capacity = index_size
                .max(buffers.index_buffer_size.saturating_mul(2))
                .max(1024);
            let buffer_info = vk::BufferCreateInfo {
                size: capacity,
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let alloc_info = vk_mem::AllocationCreateInfo {
                usage: MemoryUsage::AutoPreferHost,
                flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                    | vk_mem::AllocationCreateFlags::MAPPED,
                ..Default::default()
            };
            let (buf, alloc) = unsafe { allocator.create_buffer(&buffer_info, &alloc_info)? };
            if buffers.index_buffer != vk::Buffer::null()
                && let Some(allocation) = &mut buffers.index_allocation
            {
                unsafe {
                    allocator.destroy_buffer(buffers.index_buffer, allocation);
                }
            }
            buffers.index_buffer = buf;
            buffers.index_allocation = Some(alloc);
            buffers.index_buffer_size = capacity;
        }

        // The host-visible UI allocations stay mapped, avoiding map/unmap bookkeeping.
        unsafe {
            if let Some(allocation) = &buffers.vertex_allocation {
                let vtx_ptr = allocator.get_allocation_info(allocation).mapped_data as *mut u8;
                debug_assert!(!vtx_ptr.is_null());
                let mut offset = 0;
                for draw_list in draw_data.draw_lists() {
                    let src = draw_list.vtx_buffer();
                    let byte_len = std::mem::size_of_val(src);
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr() as *const u8,
                        vtx_ptr.add(offset),
                        byte_len,
                    );
                    offset += byte_len;
                }
                allocator.flush_allocation(allocation, 0, vertex_size)?;
            }

            if let Some(allocation) = &buffers.index_allocation {
                let idx_ptr = allocator.get_allocation_info(allocation).mapped_data as *mut u8;
                debug_assert!(!idx_ptr.is_null());
                let mut idx_offset = 0;
                for draw_list in draw_data.draw_lists() {
                    let src = draw_list.idx_buffer();
                    let byte_len = std::mem::size_of_val(src);
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr() as *const u8,
                        idx_ptr.add(idx_offset),
                        byte_len,
                    );
                    idx_offset += byte_len;
                }
                allocator.flush_allocation(allocation, 0, index_size)?;
            }
        }
        Ok(())
    }

    /// Cleans up ImGui Vulkan resources created by this renderer.
    /// This destroys all Vulkan objects owned by the renderer.
    pub fn cleanup(&mut self, allocator: &Allocator) {
        unsafe {
            // Destroy dynamic buffers
            for mut buffers in self.frame_buffers.drain(..) {
                if let Some(allocation) = &mut buffers.vertex_allocation {
                    allocator.destroy_buffer(buffers.vertex_buffer, allocation);
                }
                if let Some(allocation) = &mut buffers.index_allocation {
                    allocator.destroy_buffer(buffers.index_buffer, allocation);
                }
            }

            // Destroy font resources
            if let Some(view) = self.font_image_view.take() {
                self.device.destroy_image_view(view, None);
            }
            if let Some(sampler) = self.font_sampler.take() {
                self.device.destroy_sampler(sampler, None);
            }
            if let Some(image) = self.font_image.take()
                && let Some(mut allocation) = self.font_image_allocation.take()
            {
                allocator.destroy_image(image, &mut allocation);
            }

            // Descriptor layout and pipeline
            self.device.destroy_descriptor_set_layout(
                std::mem::replace(
                    &mut self.descriptor_set_layout,
                    vk::DescriptorSetLayout::null(),
                ),
                None,
            );
            if self.vk_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(
                    std::mem::replace(&mut self.vk_pipeline, vk::Pipeline::null()),
                    None,
                );
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device.destroy_pipeline_layout(
                    std::mem::replace(&mut self.pipeline_layout, vk::PipelineLayout::null()),
                    None,
                );
            }
        }
    }

    /// Records ImGui draw commands: bind pipeline, push textures and constants, and draw.
    /// # Arguments
    /// * `device` - The Vulkan device to use for command recording.
    /// * `cmd_buf` - The command buffer to record the ImGui draw commands into.
    /// * `draw_data` - The ImGui draw data containing vertex and index information.
    /// * `image_index` - Swapchain image being recorded (see [`UiFrameBuffers`]).
    ///   This function performs the following steps:
    ///   1. Updates the vertex and index buffers with the latest ImGui draw data.
    ///   2. Binds the vertex and index buffers to the command buffer.
    ///   3. Binds the ImGui graphics pipeline.
    ///   4. Sets the dynamic viewport based on the ImGui display size.
    ///   5. Computes the orthographic projection matrix for ImGui.
    ///   6. Iterates through the ImGui draw lists, pushing each command's texture and drawing it.
    ///
    ///   It handles scissor rectangles and indexed drawing based on ImGui's clip rects.
    ///   If there is no ImGui draw data (total vertex or index count is zero),
    ///   it simply returns without rendering anything.
    pub fn render(
        &mut self,
        device: &ash::Device,
        allocator: &Allocator,
        cmd_buf: vk::CommandBuffer,
        draw_data: &DrawData,
        image_index: usize,
    ) -> Result<(), Box<dyn Error>> {
        // If there is nothing to draw, skip UI rendering
        if draw_data.total_vtx_count == 0 || draw_data.total_idx_count == 0 {
            return Ok(());
        }
        // Account for HiDPI: logical→physical scale
        let fb_scale = draw_data.framebuffer_scale;
        // 1) Ensure buffers are up-to-date with ImGui draw data
        self.update_buffers(allocator, draw_data, image_index)?;

        // 2) Bind vertex and index buffers
        let buffers = &self.frame_buffers[image_index];
        unsafe {
            device.cmd_bind_vertex_buffers(cmd_buf, 0, &[buffers.vertex_buffer], &[0]);
            device.cmd_bind_index_buffer(cmd_buf, buffers.index_buffer, 0, vk::IndexType::UINT16);
        }
        // Bind ImGui pipeline
        unsafe {
            device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, self.vk_pipeline);
        }
        // Set dynamic viewport for UI
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: draw_data.display_size[0] * fb_scale[0],
            height: draw_data.display_size[1] * fb_scale[1],
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe {
            device.cmd_set_viewport(cmd_buf, 0, &[viewport]);
        }
        // Compute orthographic projection matrix for ImGui (matching Vulkan NDC and winit coords)
        let (l, t) = (draw_data.display_pos[0], draw_data.display_pos[1]);
        let (w, h) = (draw_data.display_size[0], draw_data.display_size[1]);
        let r = l + w;
        let b = t + h;
        // clang-format off
        let proj = [
            [2.0 / (r - l), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (b - t), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [(r + l) / (l - r), (t + b) / (t - b), 0.0, 1.0],
        ];
        // clang-format on
        unsafe {
            device.cmd_push_constants(
                cmd_buf,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::cast_slice(&proj),
            );
        }
        // 3) Iterate draw lists and issue draw calls
        unsafe {
            let mut vertex_offset: i32 = 0;
            let mut index_offset: u32 = 0;
            let mut bound_view = vk::ImageView::null();
            for draw_list in draw_data.draw_lists() {
                for cmd in draw_list.commands() {
                    if let imgui::DrawCmd::Elements { count, cmd_params } = cmd {
                        let image_info = self
                            .textures
                            .get(cmd_params.texture_id.id())
                            .unwrap_or(&self.textures[0]);
                        if image_info.image_view != bound_view {
                            let write = vk::WriteDescriptorSet::default()
                                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .image_info(std::slice::from_ref(image_info));
                            self.push_descriptor.cmd_push_descriptor_set(
                                cmd_buf,
                                vk::PipelineBindPoint::GRAPHICS,
                                self.pipeline_layout,
                                0,
                                &[write],
                            );
                            bound_view = image_info.image_view;
                        }
                        // Set scissor rectangle from ImGui clip rect
                        let clip = cmd_params.clip_rect;
                        // Convert logical clip rect to physical pixels
                        let scissor = vk::Rect2D {
                            offset: vk::Offset2D {
                                x: (clip[0] * fb_scale[0]).max(0.0) as i32,
                                y: (clip[1] * fb_scale[1]).max(0.0) as i32,
                            },
                            extent: vk::Extent2D {
                                width: ((clip[2] - clip[0]) * fb_scale[0]).max(0.0) as u32,
                                height: ((clip[3] - clip[1]) * fb_scale[1]).max(0.0) as u32,
                            },
                        };
                        device.cmd_set_scissor(cmd_buf, 0, &[scissor]);
                        // Draw indexed
                        device.cmd_draw_indexed(
                            cmd_buf,
                            count as u32,
                            1,
                            index_offset + cmd_params.idx_offset as u32,
                            vertex_offset + cmd_params.vtx_offset as i32,
                            0,
                        );
                    }
                }
                index_offset += draw_list.idx_buffer().len() as u32;
                vertex_offset += draw_list.vtx_buffer().len() as i32;
            }
        }
        Ok(())
    }
}
