# Working on RustyVulkan

## Code map

- `src/main.rs` configures asset paths and the macOS driver before starting threads.
- `src/app/` owns application state, scene construction, and input handling.
- `src/graphics/` contains CPU geometry generation and GPU mesh, material,
  texture, shader, and pipeline helpers.
- `src/vulkan/base/setup.rs` initializes the backend. `base.rs` handles frames,
  synchronization, swapchain changes, and backend teardown.
- `src/vulkan/render_graph.rs` orders the shadow, main, and UI passes and records
  attachment transitions. It is a linear pass scheduler, not a general graph compiler.
- `assets/shaders/` contains GLSL sources. Add new sources to `SHADERS` in
  `build.rs`; Cargo tracks that list and the script regenerates SPIR-V when rerun.

## Ownership and failure handling

Ash handles are copyable identifiers. Copying or dropping a handle does not
create or destroy a Vulkan resource. Keep one explicit owner for each resource
and treat copies in descriptors and scene records as borrowed handles.

`App::drop` waits for GPU work and destroys dependent renderers and managers
before dropping `VulkanBase`. The backend destroys VMA allocations before the
allocator, then the logical device, surface, and instance. `ShaderModule` uses
Rust `Drop`, but its cloned Ash device wrapper does not extend the actual Vulkan
device's lifetime.

The texture cache owns shared images and its sampler. Materials hold copies of
those handles and must be destroyed before cache teardown. Upload staging memory must remain alive until
the GPU has finished reading it. Flush non-coherent host writes before submission.

For every fallible creation step, record ownership before attempting the next
step and release completed resources if a later step fails. The swapchain
constructor demonstrates this pattern. Clear destroyed handles and take owned
allocations so later shutdown cannot destroy them twice. Vulkan retires an old
swapchain when replacement creation is attempted; keeping its Rust value on a
failure preserves cleanup ownership, not permission to render with it.

Window and render errors propagate through `App::run` to a nonzero process exit
status. A render failure is fatal because an interrupted frame can leave fences
or acquisition semaphores unsuitable for another frame. Do not resume rendering
after such a failure without a deliberate recovery protocol. Skip zero-sized
windows and perform swapchain changes at frame boundaries.

## Checks

Run from the repository root with the native dependencies described in the README:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked
git diff --check
```

The tests cover CPU geometry, camera and shadow math, data layouts, and swapchain
capability selection. They link native libraries but do not create a window or
exercise Vulkan synchronization. Debug runs require `VK_LAYER_KHRONOS_validation`.

After changes to rendering or resource ownership, also run `cargo run` with
validation enabled, resize the window, minimize and restore it, toggle fullscreen
and MSAA, interact with the UI, and close the application. Check validation output
during both rendering and teardown. Automated tests alone cannot establish that
GPU resource lifetimes and image barriers are correct.

## Areas still needing work

This is an educational renderer with explicit manual resource management. Some
startup paths, texture and grass uploads, and ImGui resource creation still need
ownership guards for every partial failure. Several helpers still panic on Vulkan
errors. Avoid copying those paths as a complete production error-handling model.

Further improvements include moving generated shaders to Cargo's `OUT_DIR`
(and adapting loading and packaging), tightening resource APIs so callers cannot
destroy borrowed handles, and validating on additional drivers and platforms.
The desktop resume guard prevents repeated initialization; mobile surface-loss
and suspension handling have not been implemented.
