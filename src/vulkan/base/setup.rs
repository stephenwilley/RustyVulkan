//! One-time Vulkan instance, device, surface, and per-image resource setup.
//!
//! These functions construct the ownership tree stored by VulkanBase. Runtime
//! frame recording stays in the parent module, where those resources are used.

use super::*;

const REQUIRED_DEVICE_EXTENSIONS: [&CStr; 3] = [
    vk::KHR_SWAPCHAIN_NAME,
    vk::KHR_PUSH_DESCRIPTOR_NAME,
    vk::KHR_MAINTENANCE5_NAME,
];

impl VulkanBase {
    fn create_instance(
        entry: &Entry,
        event_loop: &ActiveEventLoop,
        layers: &[*const i8],
    ) -> Result<Instance, Box<dyn Error>> {
        let app_name = std::ffi::CString::new("RustyVulkan")?;

        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: app_name.as_ptr(),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::API_VERSION_1_3,
            ..Default::default()
        };

        let ext_names =
            enumerate_required_extensions(event_loop.display_handle().unwrap().as_raw())?;
        let mut extension_ptrs: Vec<*const i8> = ext_names.to_vec();

        #[cfg(debug_assertions)]
        extension_ptrs.push(ash::ext::debug_utils::NAME.as_ptr());

        // Query supported instance extensions
        let supported_instance_exts =
            unsafe { entry.enumerate_instance_extension_properties(None)? };
        let supports_portability_enum = supported_instance_exts.iter().any(|e| {
            let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
            name == vk::KHR_PORTABILITY_ENUMERATION_NAME
        });

        // Enable portability enumeration only when the active loader exposes it.
        // KosmicKrisp does not need it, while MoltenVK and some other drivers do.
        let mut instance_flags = vk::InstanceCreateFlags::empty();
        if supports_portability_enum {
            extension_ptrs.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
            instance_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        // Do NOT push KHR_get_physical_device_properties2 (core since Vulkan 1.1+)
        let create_info = vk::InstanceCreateInfo {
            flags: instance_flags,
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
            let props = unsafe { instance.get_physical_device_properties(*device) };
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

    /// Selects a device and queue family that can both render and present to
    /// this window.  A discrete GPU is preferred, but only when it meets all
    /// requirements.
    fn choose_device_and_queue_family(
        instance: &Instance,
        physical_devices: &[vk::PhysicalDevice],
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(vk::PhysicalDevice, u32), Box<dyn Error>> {
        for prefer_discrete in [true, false] {
            for &physical_device in physical_devices {
                let props = unsafe { instance.get_physical_device_properties(physical_device) };
                if (props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU) != prefer_discrete {
                    continue;
                }

                // Scene and grass shaders share a 144-byte push-constant ABI.
                // Vulkan 1.3 only guarantees 128 bytes, so check before creating resources.
                if props.limits.max_push_constants_size
                    < crate::graphics::pipeline::PUSH_CONSTANT_BYTES
                {
                    eprintln!(
                        "Skipping GPU: maxPushConstantsSize={} bytes; renderer requires {}",
                        props.limits.max_push_constants_size,
                        crate::graphics::pipeline::PUSH_CONSTANT_BYTES,
                    );
                    continue;
                }

                let extensions =
                    unsafe { instance.enumerate_device_extension_properties(physical_device)? };
                let has_extension = |required: &CStr| {
                    extensions.iter().any(|extension| {
                        let name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
                        name == required
                    })
                };
                if !REQUIRED_DEVICE_EXTENSIONS
                    .iter()
                    .all(|&name| has_extension(name))
                {
                    continue;
                }

                if props.api_version < vk::API_VERSION_1_3 {
                    continue;
                }
                let supports_required_features = unsafe {
                    let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default();
                    let mut maintenance5 = vk::PhysicalDeviceMaintenance5FeaturesKHR::default();
                    let mut features2 = vk::PhysicalDeviceFeatures2::default()
                        .push_next(&mut vulkan13)
                        .push_next(&mut maintenance5);
                    instance.get_physical_device_features2(physical_device, &mut features2);
                    vulkan13.dynamic_rendering == vk::TRUE
                        && vulkan13.synchronization2 == vk::TRUE
                        && maintenance5.maintenance5 == vk::TRUE
                };
                if !supports_required_features {
                    continue;
                }

                let queue_families = unsafe {
                    instance.get_physical_device_queue_family_properties(physical_device)
                };
                for (index, family) in queue_families.iter().enumerate() {
                    if !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        continue;
                    }
                    if unsafe {
                        surface_loader.get_physical_device_surface_support(
                            physical_device,
                            index as u32,
                            surface,
                        )?
                    } {
                        println!("🎯 Found graphics/present queue family at index {}", index);
                        return Ok((physical_device, index as u32));
                    }
                }
            }
        }

        Err("No Vulkan 1.3 device supports graphics, presentation, dynamic rendering, synchronization2, push descriptors, maintenance5, and at least 144 bytes of push constants".into())
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
        enable_multi_draw_indirect: bool,
        enable_draw_indirect_first_instance: bool,
        enable_wireframe: bool,
    ) -> Result<(ash::Device, vk::Queue), vk::Result> {
        let queue_priority = [1.0_f32];

        // Enable only the features the renderer uses.
        let device_features = vk::PhysicalDeviceFeatures {
            // LINE polygon mode requires this feature to be enabled, not just supported.
            fill_mode_non_solid: if enable_wireframe {
                vk::TRUE
            } else {
                vk::FALSE
            },
            multi_draw_indirect: if enable_multi_draw_indirect {
                vk::TRUE
            } else {
                vk::FALSE
            },
            draw_indirect_first_instance: if enable_draw_indirect_first_instance {
                vk::TRUE
            } else {
                vk::FALSE
            },
            ..Default::default()
        };

        let queue_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: queue_priority.as_ptr(),
            ..Default::default()
        };

        // Query device extensions on this adapter
        let supported_dev_exts =
            unsafe { instance.enumerate_device_extension_properties(physical_device)? };

        let has_portability_subset = supported_dev_exts.iter().any(|e| {
            let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
            name == vk::KHR_PORTABILITY_SUBSET_NAME
        });

        let mut device_extensions: Vec<*const i8> = REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .map(|name| name.as_ptr())
            .collect();
        if has_portability_subset {
            // Required by portability drivers such as MoltenVK; absent on KosmicKrisp.
            device_extensions.push(vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr());
        }

        let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features {
            dynamic_rendering: vk::TRUE,
            synchronization2: vk::TRUE,
            ..Default::default()
        };
        let mut maintenance5_features = vk::PhysicalDeviceMaintenance5FeaturesKHR {
            maintenance5: vk::TRUE,
            ..Default::default()
        };

        let device_create_info = vk::DeviceCreateInfo {
            p_queue_create_infos: &queue_info,
            queue_create_info_count: 1,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            p_enabled_features: &device_features,
            ..Default::default()
        }
        .push_next(&mut vulkan13_features)
        .push_next(&mut maintenance5_features);

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
    fn create_command_pool(
        device: &ash::Device,
        queue_family_index: u32,
    ) -> Result<vk::CommandPool, vk::Result> {
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
    pub(super) fn allocate_command_buffers(
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

    /// Create N mapped host-visible uniform buffers sized for `GlobalUbo`.
    pub(super) fn create_uniform_buffers(
        allocator: &Allocator,
        count: usize,
    ) -> Result<(Vec<vk::Buffer>, Vec<Allocation>), vk::Result> {
        let mut buffers = Vec::with_capacity(count);
        let mut allocations = Vec::with_capacity(count);

        let buffer_size = std::mem::size_of::<GlobalUbo>() as vk::DeviceSize;

        for _ in 0..count {
            let buffer_info = vk::BufferCreateInfo {
                size: buffer_size,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let alloc_info = vk_mem::AllocationCreateInfo {
                usage: MemoryUsage::AutoPreferHost,
                flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                    | vk_mem::AllocationCreateFlags::MAPPED,
                ..Default::default()
            };
            let (buffer, allocation) =
                match unsafe { allocator.create_buffer(&buffer_info, &alloc_info) } {
                    Ok(resource) => resource,
                    Err(error) => {
                        for (buffer, mut allocation) in buffers.into_iter().zip(allocations) {
                            unsafe { allocator.destroy_buffer(buffer, &mut allocation) };
                        }
                        return Err(error);
                    }
                };
            buffers.push(buffer);
            allocations.push(allocation);
        }

        Ok((buffers, allocations))
    }

    /// Build a descriptor pool and one set=0 descriptor set per swapchain image, then write binding 0 to each UBO.
    pub(super) fn create_set0_descriptor_pool_and_sets(
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
        ubo_buffers: &[vk::Buffer],
    ) -> Result<(vk::DescriptorPool, Vec<vk::DescriptorSet>), vk::Result> {
        let count = ubo_buffers.len() as u32;

        // Pool: one UBO and one sampler per set
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: count,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: count,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo {
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            max_sets: count,
            ..Default::default()
        };
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        // Allocate
        let layouts = vec![layout; count as usize];
        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: pool,
            descriptor_set_count: count,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };
        let sets = match unsafe { device.allocate_descriptor_sets(&alloc_info) } {
            Ok(sets) => sets,
            Err(error) => {
                unsafe { device.destroy_descriptor_pool(pool, None) };
                return Err(error);
            }
        };

        // Write binding 0
        let range = std::mem::size_of::<GlobalUbo>() as vk::DeviceSize;
        let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(count as usize);
        for &b in ubo_buffers {
            buf_infos.push(vk::DescriptorBufferInfo {
                buffer: b,
                offset: 0,
                range,
            });
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
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }

        Ok((pool, sets))
    }

    /// Creates the Vulkan ownership tree stored by `VulkanBase`.
    pub fn new(window: &Window, event_loop: &ActiveEventLoop) -> Result<Self, Box<dyn Error>> {
        let engine_settings = EngineSettings {
            wireframe: false,
            show_ui: true, // UI enabled by default
            msaa_samples: 4,
            // Four 2048² layers use the same texel count as the old 4096² map.
            shadow_map_resolution: 2048,
            shadow_distance: 75.0,
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

        // Surface capability (including presentation support) is a property
        // of a physical device, so it must exist before selecting that device.
        let surface = Self::create_surface(&entry, &instance, window, event_loop)?;
        let surface_loader = surface::Instance::new(&entry, &instance);

        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        #[cfg(debug_assertions)]
        Self::print_physical_devices(&instance, &physical_devices);

        let (physical_device, graphics_queue_family_index) = Self::choose_device_and_queue_family(
            &instance,
            &physical_devices,
            &surface_loader,
            surface,
        )?;
        let chosen_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let chosen_name = unsafe {
            CStr::from_ptr(chosen_props.device_name.as_ptr())
                .to_str()
                .unwrap_or("<invalid utf-8>")
        };
        let sample_count_flags_supported = chosen_props.limits.framebuffer_color_sample_counts
            & chosen_props.limits.framebuffer_depth_sample_counts;
        let supported_features = unsafe { instance.get_physical_device_features(physical_device) };
        let supports_wireframe = supported_features.fill_mode_non_solid == vk::TRUE;
        let supports_multi_draw_indirect = supported_features.multi_draw_indirect == vk::TRUE;
        let supports_draw_indirect_first_instance =
            supported_features.draw_indirect_first_instance == vk::TRUE;
        println!("👉 Selected device for next steps: '{}'", chosen_name);

        let (device, graphics_queue) = Self::create_logical_device_and_queue(
            &instance,
            physical_device,
            graphics_queue_family_index,
            supports_multi_draw_indirect,
            supports_draw_indirect_first_instance,
            supports_wireframe,
        )?;
        // This cache lives for the device lifetime and is shared by every graphics pipeline.
        let pipeline_cache =
            unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)? };

        let command_pool = Self::create_command_pool(&device, graphics_queue_family_index)?;

        // Create a Vulkan Memory Allocator (VMA) instance.
        let mut allocator_info =
            vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
        allocator_info.flags |= vk_mem::AllocatorCreateFlags::EXT_MEMORY_BUDGET;
        let allocator = unsafe { Allocator::new(allocator_info)? };

        // Binding 1 contains one sampled 2D-array view of all shadow cascades.
        let ubo_binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        };
        let shadow_binding = vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        };
        let bindings = [ubo_binding, shadow_binding];
        let set0_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        let set0_global_layout = unsafe { device.create_descriptor_set_layout(&set0_info, None)? };
        println!(
            "🔧 Created global set=0 layout (binding 0 = UBO, binding 1 = cascaded shadow map)"
        );

        let swapchain = Swapchain::new(
            &instance,
            &device,
            physical_device,
            &surface,
            &surface_loader,
            window,
            &allocator,
            engine_settings.msaa_samples,
        )?;
        let swapchain_loader = swapchain::Device::new(&instance, &device);
        let push_descriptor = push_descriptor::Device::new(&instance, &device);

        // Create sync objects: N frames-in-flight worth of semaphores/fences
        let image_count = swapchain.swapchain_image_views.len();
        let (ubo_buffers, ubo_allocations) = Self::create_uniform_buffers(&allocator, image_count)?;
        let (set0_descriptor_pool, set0_descriptor_sets) =
            Self::create_set0_descriptor_pool_and_sets(&device, set0_global_layout, &ubo_buffers)?;

        // Shadow sampler (no compare; better compatibility with portability subset on macOS)
        let sampler_info = vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
            unnormalized_coordinates: vk::FALSE,
            compare_enable: vk::FALSE,
            ..Default::default()
        };
        let shadow_sampler = unsafe { device.create_sampler(&sampler_info, None)? };
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
            unsafe {
                render_finished_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
            }
        }

        // Per-swapchain-image tracker: which fence currently owns each image (or null)
        let image_owner_fence = vec![vk::Fence::null(); image_count];

        let command_buffers = Self::allocate_command_buffers(
            &device,
            command_pool,
            vk::CommandBufferLevel::PRIMARY,
            swapchain.swapchain_image_views.len(),
        )?;

        let attachment_manager = AttachmentManager::new(image_count);

        // One query range per swapchain image: frame plus shadow/scene/vegetation boundaries.
        let query_pool_info = vk::QueryPoolCreateInfo {
            query_type: vk::QueryType::TIMESTAMP,
            query_count: (image_count as u32) * TIMESTAMPS_PER_IMAGE,
            ..Default::default()
        };
        let timestamp_query_pool = unsafe { device.create_query_pool(&query_pool_info, None)? };

        let vulkan_base = Self {
            instance,
            physical_device,
            device,
            push_descriptor,
            pipeline_cache,
            allocator: Some(allocator),
            graphics_queue,
            // set 0
            set0_global_layout,
            set0_descriptor_pool,
            set0_descriptor_sets,
            ubo_buffers,
            ubo_allocations,
            shadow_sampler,
            shadow_descriptor_views: vec![vk::ImageView::null(); image_count],
            timestamp_query_pool,
            timestamp_period_ns: chosen_props.limits.timestamp_period,
            last_gpu_timings: GpuPassTimings::default(),
            last_image_per_slot: vec![None; INFLIGHT_FRAMES],
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
            engine_settings,
            attachment_manager,
            #[cfg(debug_assertions)]
            debug_messenger,
            #[cfg(debug_assertions)]
            debug_utils_loader,
            frame_slot: 0,
            current_image_index: 0,
            // RenderGraph also starts at zero; only a later settings or format change
            // should rebuild the pipelines created during scene setup.
            pipeline_generation: 0,
            sample_count_flags_supported,
            supports_multi_draw_indirect,
            supports_draw_indirect_first_instance,
            supports_wireframe,
            max_draw_indirect_count: chosen_props.limits.max_draw_indirect_count,
            pending_msaa_samples: None,
            swapchain_recreation_needed: false,
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
        let messenger = unsafe { loader.create_debug_utils_messenger(&create_info, None)? };
        println!("🔍 Debug messenger created");
        Ok((loader, messenger))
    }
}
