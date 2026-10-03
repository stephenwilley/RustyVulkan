# RustyVulkan code guide: design

## Goal

A static web guide, living in the repo, that takes a reader from no Vulkan
knowledge to understanding the whole RustyVulkan codebase. It uses the code
to explain how things work; it is not a build-along tutorial. Written against
commit `1380484`.

## Readers

Both are solid programmers with a non-programming grounding in 3D graphics
(meshes, coordinates, textures, lighting). Neither is assumed to know Rust.

1. **The ex-OpenGL C programmer** who wants to learn the new thing in detail.
2. **The skimmer** who wants a high-level understanding.

Every page must serve both.

## Page template

Each chapter page reads in 10 minutes or less and has:

1. **Header:** chapter number, title, estimated reading time, and a
   one-sentence summary of what the chapter covers.
2. **"The big picture":** a plain-English explanation of what this part of the
   code does and why, with an inline SVG diagram where it helps. This part must
   stand alone: a skimmer can stop here and still come away with something useful.
3. **A visible divider**, then **"Deeper dive":** annotated code excerpts
   copied from the source, each labelled with `path/file.rs:line`. The excerpts
   are trimmed to the relevant lines, using `// ...` for elisions.
4. **Callouts**, used where relevant rather than by quota:
   - *Coming from OpenGL:* maps a Vulkan concept to its GL ancestor, or explains
     why GL had no equivalent.
   - *Rust note:* explains Rust syntax or idiom as it meets the code, especially
     where Rust handles a C API differently (`unsafe`, `Copy` handles,
     `Drop`, `Result`/`?`, builders, `bytemuck`, `#[repr(C)]`, `cfg`, lifetimes).
     Each Rust concept is explained in full the first time it appears and
     linked back to after that.
5. **A recap** of three to five bullets.
6. **Previous/next navigation** and a link back to the contents page.

## Cover page (`index.html`)

- A hero section with the project's screenshot, the project name, and a
  one-paragraph pitch.
- A "who this is for / how to read it" paragraph that explains the two-layer pages.
- A table of contents: numbered chapters, each with a one-line description and
  reading time, grouped into parts:
  - Foundations (1–2)
  - Talking to the GPU (3–5)
  - Drawing things (6–8)
  - Making frames (9–10)
  - The scene (11–15)
  - Bringing it together (16)

## Chapters

| # | File | Title | Source covered |
|---|---|---|---|
| 1 | `01-why-vulkan.html` | Why Vulkan, and a map of this repo | `main.rs`, module layout, how Vulkan differs from OpenGL, one frame end to end, KosmicKrisp on macOS |
| 2 | `02-rust-meets-c.html` | Rust meets a C API | `ash` and its builders, `unsafe`, handles vs. ownership, `Result`, `Drop`, teardown order (as framed in DEVELOPMENT.md) |
| 3 | `03-instance-devices.html` | Instance, devices and queues | `vulkan/base/setup.rs`, validation layers, extensions, choosing a physical device |
| 4 | `04-swapchain.html` | Surfaces and the swapchain | `vulkan/swapchain.rs`, `raw-window-handle`/`ash-window` |
| 5 | `05-gpu-memory.html` | GPU memory | VMA (`vk-mem`), buffers, images, staging uploads, `graphics/mesh.rs`, `graphics/texture.rs` |
| 6 | `06-shaders.html` | Shaders | `build.rs` (GLSL to SPIR-V via shaderc), `graphics/shaders.rs`, a tour of `main.vert`/`main.frag` |
| 7 | `07-pipelines.html` | Pipelines and dynamic rendering | `graphics/pipeline.rs`, vertex layouts, fixed-function state |
| 8 | `08-descriptors-materials.html` | Descriptors and materials | `graphics/material.rs`, `materialmanager.rs`, texture cache and samplers, push constants |
| 9 | `09-frame-loop.html` | The frame loop | `vulkan/base.rs`: frames in flight, fences, semaphores, command buffers, acquire/submit/present, swapchain recreation |
| 10 | `10-render-graph.html` | The render graph | `vulkan/render_graph.rs`, `attachments.rs`, image layout transitions and barriers, MSAA |
| 11 | `11-shadows.html` | Shadows | `vulkan/shadow_pass.rs`, `graphics/shadow_math.rs`, `shadow_depth.vert` |
| 12 | `12-main-pass-scene.html` | The main pass and the scene | `vulkan/main_pass.rs`, `graphics/camera.rs`, `meshmanager.rs`, `assimp_loader.rs`, `import.rs`, `app/scene.rs` |
| 13 | `13-procedural-world.html` | A procedural world | `graphics/terrain.rs`, `sky.rs`, `rocks.rs` and their shaders |
| 14 | `14-grass.html` | Grass | `graphics/grass.rs` and `grass/*` (generation, upload, pipeline, wind, tests) |
| 15 | `15-imgui.html` | The ImGui overlay | `vulkan/imgui_renderer*`, `vulkan/ui_pass.rs`, `imgui.vert/frag` |
| 16 | `16-app-shell.html` | The app shell | `app/app.rs`, `app/input.rs`, winit event loop, resize/minimise, teardown, `scripts/package_macos.sh`, `PACKAGING.md` |

Shaders are covered in the chapters that use them. Any source file not named
above is folded into the closest chapter, so every file gets coverage.

## Visual style

- Complementary to vulkan.org: a dark charcoal header and footer, Vulkan's
  crimson as the accent colour, a clean sans-serif for body text, and a
  monospace font for code. The exact colour values come from vulkan.org at
  build time.
- Light theme by default. Dark theme follows `prefers-color-scheme`.
- Callouts get distinct, colour-coded boxes: OpenGL in a muted blue, Rust in
  Rust orange, plus a neutral style for warnings and asides.
- Code blocks get lightweight syntax highlighting. If done in JS, use
  highlight.js from a CDN and degrade gracefully to plain monospace when offline.
- Diagrams are hand-written inline SVG that uses the theme's CSS variables, so
  they work in both themes.
- Comfortable reading width (~72ch), and no horizontal page scroll on a phone.

## Files

```
docs/guide/
  index.html
  01-why-vulkan.html … 16-app-shell.html
  style.css
  images/screenshot.jpg      (downscaled from the provided 3420×2032 capture)
```

No build step, generator or dependencies. The pages open directly from disk
and are ready for GitHub Pages. README.md gets a one-line link to the guide.

## Accuracy

- Every claim about the code is checked against the source at `1380484`.
  Excerpts are copied verbatim, apart from marked elisions.
- The footer of every page states the commit the guide was written against, so
  drift is visible.
- Where the code has known gaps (DEVELOPMENT.md's "Areas still needing work"),
  the guide says so rather than presenting them as best practice.

## Out of scope

- Tooling to keep excerpts in sync automatically.
- Search, analytics, comments.
- Changes to the renderer itself.
