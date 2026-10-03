# RustyVulkan Code Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 16-chapter static HTML guide in `docs/guide/` that explains the RustyVulkan codebase, from no Vulkan knowledge through to full coverage, with a cover page and table of contents.

**Architecture:** Hand-written HTML pages share one stylesheet and one small JS file (theme-aware syntax highlighting via highlight.js from cdnjs, which degrades to plain monospace offline). Every chapter follows one fixed template. A throwaway Python checker in the scratchpad (not committed) validates structure, links, reading length and excerpt fidelity against the source.

**Tech Stack:** HTML5, CSS custom properties, inline SVG, highlight.js 11 (cdnjs), Python 3 stdlib for the checker, `sips` for image resizing.

**Spec:** `docs/superpowers/specs/2026-10-02-vulkan-guide-design.md`

## Global Constraints

- Written against commit `1380484`. Every page footer says: "Written against RustyVulkan commit 1380484."
- Each chapter must take 10 minutes or less to read: at most **2,400 words of prose** (code excluded), assuming roughly 240 wpm for technical text.
- Readers: solid programmers with a non-programming 3D background. Assume they know nothing of Rust and nothing of Vulkan. Two audiences: the ex-OpenGL C coder (deep dive) and the skimmer (big picture).
- Template order: header → "The big picture" → divider → "Deeper dive" → "Recap" → prev/next nav → footer.
- Callout classes: `callout gl` ("Coming from OpenGL"), `callout rust` ("Rust note"), `callout aside` (warnings/gaps).
- A Rust concept gets its full explanation once, where it first appears, and later mentions link back to that explanation (`02-rust-meets-c.html#…` or wherever it first appeared).
- Code excerpts are verbatim from the source at `1380484`, except for elisions marked `// ...` on their own line. Each excerpt has a `<figcaption>` with `src/path.rs:START-END`.
- No build step. External resources come only from cdnjs.cloudflare.com and Google Fonts. Pages must work opened from `file://`.
- Light theme by default, dark theme via `prefers-color-scheme`. No horizontal page scroll at 375px width (code blocks scroll internally).
- Known gaps from DEVELOPMENT.md "Areas still needing work" are flagged as gaps (`callout aside`), not presented as best practice.
- Do not modify anything under `src/`, `assets/`, or build files. Commit only when the user says to (user rule: no commits without an explicit "commit").

## Review Focus

1. **Excerpt drift or invention.** An excerpt that doesn't match the source misleads exactly the reader who goes on to open the file. The checker's excerpt test (Task 1) catches it, and every chapter task runs it.
2. **Skimmer can't stand alone.** If "The big picture" leans on code terms that only the deep dive defines, it fails the skimmer. Each chapter task's review step reads that section in isolation.
3. **Dark mode diagrams.** SVGs with hard-coded black or white disappear in one of the themes. The checker flags literal `#000`/`#fff`/`black`/`white` fill or stroke values inside `<svg>`.
4. **Phone width.** Wide code or SVGs causing page-level horizontal scroll. Task 8 does a browser check at 375px.
5. **Uncovered source files.** The spec says every file gets covered. The checker cross-references `git ls-files src assets/shaders build.rs scripts` against `<figcaption>` paths and in-text `<code>` paths.

---

## File Structure

```
docs/guide/
  index.html                  cover, hero screenshot, TOC (Task 1)
  style.css                   all styling + theme tokens (Task 1)
  guide.js                    highlight.js init only (Task 1)
  images/screenshot.jpg       2000px-wide resize of ~/Downloads/screenshot.jpg (Task 1)
  01-why-vulkan.html … 16-app-shell.html   (Tasks 2–7)
README.md                     one-line link to guide (Task 8)
<scratchpad>/check_guide.py   checker, not committed (Task 1)
```

`<scratchpad>` = `/private/tmp/claude-501/-Users-stephen-git-RustyVulkan/64ac60e5-3984-4627-a93f-b26212d98bae/scratchpad`

## Chapter template (every chapter task uses this verbatim skeleton)

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NN · Chapter Title · RustyVulkan Guide</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
<link rel="stylesheet" href="style.css">
<link rel="stylesheet" media="(prefers-color-scheme: light)" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<link rel="stylesheet" media="(prefers-color-scheme: dark)" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
</head>
<body>
<header class="site"><a class="brand" href="index.html">RustyVulkan <span>Guide</span></a><a class="toc-link" href="index.html#contents">Contents</a></header>
<main class="chapter">
  <header class="chapter-head">
    <p class="kicker">Part N · Chapter NN · <span class="readtime">~M min read</span></p>
    <h1>Chapter Title</h1>
    <p class="lede">One sentence on what this chapter covers.</p>
  </header>

  <section id="big-picture">
    <h2><span class="tag">Overview</span> The big picture</h2>
    <!-- prose + optional <figure class="diagram"><svg …></svg><figcaption>…</figcaption></figure> -->
  </section>

  <div class="dive-divider"><span>Deeper dive</span></div>

  <section id="deep-dive">
    <h2>…</h2>
    <figure class="code">
      <pre><code class="language-rust">…verbatim excerpt, HTML-escaped…</code></pre>
      <figcaption>src/path/file.rs:10-42</figcaption>
    </figure>
    <aside class="callout gl"><p class="callout-title">Coming from OpenGL</p><p>…</p></aside>
    <aside class="callout rust" id="rust-…"><p class="callout-title">Rust note</p><p>…</p></aside>
  </section>

  <section id="recap" class="recap">
    <h2>Recap</h2>
    <ul><li>…</li></ul>
  </section>

  <nav class="pager"><a class="prev" href="…">← NN · Title</a><a class="next" href="…">NN · Title →</a></nav>
</main>
<footer class="site">Written against RustyVulkan commit <code>1380484</code>. · <a href="index.html#contents">Contents</a></footer>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/rust.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/glsl.min.js"></script>
<script src="guide.js"></script>
</body>
</html>
```

Chapter 1 has no `.prev` link (it points back to index instead). Chapter 16's `.next` points to `index.html#contents` with the text "Back to contents".

## Writing method for each chapter (applies to Tasks 2–7)

For each chapter:
1. Read every source file listed for that chapter in full before writing.
2. Draft "The big picture" (about 500–800 words) for the skimmer. Plain English, with an SVG diagram where a flow, ownership or layout is involved. Define every Vulkan term the first time it appears, as `<dfn>`.
3. Draft "Deeper dive" (the rest of the budget). Walk through the excerpts in reading order and explain *why* as well as *what*. Add GL callouts where a GL programmer would expect something different, and Rust callouts where the syntax would stump a C programmer.
4. Recap: 3–5 bullets.
5. Run the checker and fix every failure.
6. Re-read "The big picture" alone and confirm it makes sense with the deep dive deleted.

---

### Task 1: Scaffold: stylesheet, JS, screenshot, cover page, checker

**Files:**
- Create: `docs/guide/style.css`, `docs/guide/guide.js`, `docs/guide/index.html`, `docs/guide/images/screenshot.jpg`
- Create (not committed): `<scratchpad>/check_guide.py`

**Interfaces:**
- Produces: the CSS classes used by the template (`site`, `brand`, `chapter`, `chapter-head`, `kicker`, `lede`, `tag`, `dive-divider`, `code`, `diagram`, `callout gl|rust|aside`, `callout-title`, `recap`, `pager`, `prev`, `next`), CSS variables for SVG (`--fg`, `--muted`, `--accent`, `--accent-2`, `--surface`, `--line`, `--gl`, `--rust`), and `check_guide.py [paths...]`, which exits non-zero on failure.

- [ ] **Step 1: Get vulkan.org's palette.** Fetch https://www.vulkan.org and its main stylesheet with WebFetch. Record the header/background charcoal and the brand red. If neither can be found, use charcoal `#1d1d1f`, crimson `#a41e22`, lighter crimson `#c8323a` for dark mode.

- [ ] **Step 2: Write the checker** at `<scratchpad>/check_guide.py`. It takes the repo root as its working directory. For each HTML file given (default: all `docs/guide/*.html`), it checks:
  - **Structure (chapters only, `NN-*.html`):** the page has `#big-picture`, `.dive-divider`, `#deep-dive`, `#recap`, `.pager`, and a footer containing `1380484`.
  - **Links:** every relative `href`/`src` resolves to an existing file, and every `#fragment` exists as an `id` in the target.
  - **Length:** prose words (text content minus `<pre>`, `<svg>`, `<figcaption>`) ≤ 2400. Print the count for every page.
  - **Excerpts:** for each `figure.code`, parse the `figcaption` `path:START-END`, unescape the `<code>` text, and split it on lines that are exactly `// ...` (after strip). Each chunk, after stripping each line's trailing whitespace and the common leading indentation, must appear as a contiguous run of lines inside source lines START..END, compared after the same normalisation. Report the caption on failure.
  - **SVG colours:** inside `<svg>`, fail on `fill`/`stroke` values of `#000`, `#000000`, `#fff`, `#ffffff`, `black` or `white`. Colours must come from `var(--…)` or `currentColor`.
  - **Coverage (`--coverage` flag):** list files from `git ls-files src assets/shaders build.rs scripts` that appear in no figcaption and in no `<code>` element's text across all pages.

  Use `html.parser` from the stdlib only.

- [ ] **Step 3: Run the checker on an empty set to confirm it runs.** `python3 <scratchpad>/check_guide.py docs/guide/*.html` should report that there are no files, or `index.html` failing link checks once it exists.

- [ ] **Step 4: Resize the screenshot.**

```bash
mkdir -p docs/guide/images
sips -Z 2000 -s format jpeg -s formatOptions 82 ~/Downloads/screenshot.jpg --out docs/guide/images/screenshot.jpg
```
Expected: a file under 700 KB, 2000×1188.

- [ ] **Step 5: Write `style.css`.** Contents:
  - `:root` tokens for light mode, redefined under `@media (prefers-color-scheme: dark)`.
  - `body` with an explicit background. Inter for prose, JetBrains Mono for code.
  - A charcoal `header.site` with a 3px crimson bottom border.
  - Main column at `max-width: 72ch` with a 16px gutter. `figure.code pre` uses `overflow-x: auto`.
  - Callouts: a left border of 4px in `--gl` (muted blue, about `#3b6ea5`) or `--rust` (about `#ce422b`), on a tinted background.
  - `.dive-divider` is a full-width rule with a centred crimson pill label.
  - A cover hero: the screenshot full width with rounded corners and a subtle shadow, plus a charcoal title band.
  - TOC: a grid of part groups, each chapter shown as a card with number, title, one-liner and read time.
  - `svg` is `max-width:100%; height:auto`.
  - Pager: a flex layout with space-between.

- [ ] **Step 6: Write `guide.js`:**

```js
// Syntax highlighting is optional: pages read fine as plain monospace offline.
if (window.hljs) {
  document.querySelectorAll('pre code').forEach((el) => window.hljs.highlightElement(el));
}
```

- [ ] **Step 7: Write `index.html`.** Same head and footer as the template. It contains:
  - A hero: `images/screenshot.jpg` with alt text describing the scene (wooden hut on a grassy meadow under a blue sky, with ImGui sun-light and shadow-cascade panels).
  - The title "RustyVulkan: a guided tour" and a pitch paragraph.
  - A "How to read this guide" section. It explains the two layers (skim the big pictures, or dive), the two callout types with live examples of each style, and the prerequisites.
  - `<section id="contents">`, using the six parts from the spec. Each card links to its `NN-slug.html`, with the title, one-liner and `~M min`. Use the Chapters table in the spec for titles and filenames; read times start at `~10 min` and are corrected in Task 8.

- [ ] **Step 8: Review the cover in the browser.** Open `file:///Users/stephen/git/RustyVulkan/docs/guide/index.html` in the built-in browser (`preview_start` with url). Screenshot it in light mode, and check dark mode with `resize_window colorScheme: dark`. Fix anything broken. Chapter links will 404 until later tasks; that is expected.

- [ ] **Step 9: Hand over.** Do not commit. Report progress.

### Task 2: Part 1, Foundations (chapters 1–2)

**Files:** Create `docs/guide/01-why-vulkan.html`, `docs/guide/02-rust-meets-c.html`

**Interfaces:**
- Consumes: the template, CSS classes and checker from Task 1.
- Produces: Rust anchor ids that later chapters link to. They must exist exactly as follows: `02-rust-meets-c.html#rust-unsafe`, `#rust-handles-copy`, `#rust-result`, `#rust-drop`, `#rust-builders`, `#rust-modules`, `#rust-cfg`, `#rust-option`, `#rust-borrowing`.

**Sources to read:** `src/main.rs`, `src/*/mod.rs`, `Cargo.toml`, `README.md`, `DEVELOPMENT.md`, a skim of `src/vulkan/base.rs` (for the frame diagram), and `src/vulkan/base/setup.rs` top section (for `ash` usage examples).

- [ ] **Step 1: Chapter 1, "Why Vulkan, and a map of this repo".** Contents:
  - Big picture: what OpenGL hid (a global state machine, a driver doing validation and sync behind your back) versus Vulkan's explicitness.
  - An SVG of the repo layers: `main.rs` → `app/` → `graphics/` and `vulkan/` → ash → Vulkan loader → KosmicKrisp → Metal → GPU.
  - An SVG of "one frame end to end": acquire image → record shadow/main/UI passes → submit → present.
  - Deep dive: `main.rs` (configure_runtime, the bundle Resources path, configure_kosmickrisp, the `unsafe set_var` and why it's sound only before threads start), `mod` declarations, and the `Cargo.toml` dependencies explained one line each.
  - Rust callouts: `#rust-modules` (mod/use), `#rust-cfg` (`#[cfg(target_os)]`), `#rust-option` (Option and combinators like `and_then`/`filter`, as used in configure_runtime), plus `?` in brief with a link forward to ch 2's `#rust-result`.

- [ ] **Step 2: Chapter 2, "Rust meets a C API".** Contents:
  - Big picture: Vulkan is a C API of create/destroy pairs with opaque handles. `ash` is a thin, almost 1:1 wrapper. Rust's safety model can't see GPU lifetimes, so the code makes ownership explicit by convention (DEVELOPMENT.md's "Ownership and failure handling").
  - An SVG of the teardown order: App::drop → renderers/managers → VMA allocations → allocator → device → surface → instance.
  - Deep dive, with real excerpts from setup.rs/base.rs:
    - an `ash` builder call (`vk::...CreateInfo::default().x(..)`) and how it differs from filling a C struct with `sType`/`pNext`
    - `unsafe { device.create_… }`
    - `VkResult` → `Result` and `?`
    - handles being `Copy`
    - `Drop` impls and the `ShaderModule` Drop example
    - `std::mem::take` / `Option::take` for avoiding double-destroy
  - Callouts with the exact ids listed under Produces. Also add a GL callout about glGen*/glDelete* versus explicit create/destroy, and one about objects no longer being bound to a context.

- [ ] **Step 3: Run the checker.** `python3 <scratchpad>/check_guide.py docs/guide/01-why-vulkan.html docs/guide/02-rust-meets-c.html`. Expected: PASS for both, with word counts printed (each ≤ 2400). Links to chapter 3 will fail until Task 3 exists. That is acceptable at this stage; note it and move on.

- [ ] **Step 4: Run the skimmer test.** Read each `#big-picture` alone and rewrite any sentence that depends on the deep dive.

- [ ] **Step 5: Hand over.** No commit.

### Task 3: Part 2, Talking to the GPU (chapters 3–5)

**Files:** Create `03-instance-devices.html`, `04-swapchain.html`, `05-gpu-memory.html` in `docs/guide/`

**Interfaces:**
- Consumes: the Rust anchor ids from Task 2.
- Produces: anchors `05-gpu-memory.html#staging` and `#vma`, which chapters 8, 13 and 14 link to.

**Sources:** ch3: `src/vulkan/base/setup.rs`. ch4: `src/vulkan/swapchain.rs`, plus the surface creation in setup.rs. ch5: VMA usage in `base.rs`/`setup.rs`, `src/graphics/mesh.rs`, `src/graphics/texture.rs` (upload, mipmaps if present, sampler).

- [ ] **Step 1: Chapter 3.**
  - Big picture: instance (connection to the loader), layers (validation), extensions (portability on macOS/KosmicKrisp), physical device versus logical device, queue families and queues.
  - An SVG of instance → physical devices → logical device → queues.
  - Deep dive: walk setup.rs in order, covering what features and extensions it enables and why (e.g. dynamic rendering, synchronization2 if used). The debug messenger.
  - GL callout: "there's no context".

- [ ] **Step 2: Chapter 4.**
  - Big picture: a surface is the window as Vulkan sees it. The swapchain is a ring of images you borrow and give back. Present modes, explained as vsync analogies.
  - An SVG of the swapchain ring with acquire/present arrows.
  - Deep dive: capability selection (format, present mode, extent, image count), the image views, recreation, and the partial-failure ownership pattern that DEVELOPMENT.md calls out (the constructor recording ownership step by step). The swapchain tests, if any are in the file.
  - GL callout: `SwapBuffers` versus all of this.

- [ ] **Step 3: Chapter 5.**
  - Big picture: GPU memory types (device-local versus host-visible), why VMA exists, and buffers versus images. Staging: copy from CPU-visible memory to fast GPU memory via a command buffer. Image layouts, introduced lightly with a link forward to ch 10.
  - An SVG of the staging upload path.
  - Deep dive: vertex/index buffer creation in mesh.rs, the vertex struct with `#[repr(C)]` + bytemuck Pod (Rust callout `#rust-repr-c` lives here and is new), texture upload and the sampler in texture.rs, flushing non-coherent memory, and keeping staging buffers alive until the GPU is done.
  - Aside callout: DEVELOPMENT.md notes that some upload paths lack full partial-failure guards.
  - GL callout: `glBufferData` hid all of this.

- [ ] **Step 4: Run the checker** on the three files. Expected: PASS apart from forward links to not-yet-written chapters.
- [ ] **Step 5: Run the skimmer test** on all three big-picture sections.
- [ ] **Step 6: Hand over.** No commit.

### Task 4: Part 3, Drawing things (chapters 6–8)

**Files:** Create `06-shaders.html`, `07-pipelines.html`, `08-descriptors-materials.html` in `docs/guide/`

**Interfaces:**
- Consumes: `05-gpu-memory.html#staging` and `#vma`, plus the ch2 Rust anchors.
- Produces: `07-pipelines.html#pipeline-layout` and `08-descriptors-materials.html#descriptor-sets`, which chapters 11–15 link to.

**Sources:** ch6: `build.rs`, `src/graphics/shaders.rs`, `assets/shaders/main.vert`, `main.frag`. ch7: `src/graphics/pipeline.rs`. ch8: `src/graphics/material.rs`, `materialmanager.rs`, the texture cache parts of `texture.rs`.

- [ ] **Step 1: Chapter 6.**
  - Big picture: GL compiled GLSL at runtime inside the driver. Vulkan takes SPIR-V bytecode. Here, `build.rs` compiles GLSL with shaderc at build time.
  - An SVG of `.vert/.frag` → build.rs/shaderc → `.spv` → ShaderModule → pipeline.
  - Deep dive: build.rs (the SHADERS list, rerun-if-changed, where output goes), shaders.rs (loading, the Drop impl, linking to `#rust-drop`), and a guided read of main.vert/main.frag (in/out locations, set/binding, push constants), with GLSL excerpts using `language-glsl` and paths under `assets/shaders/`.
  - Rust callout: what a build script is.

- [ ] **Step 2: Chapter 7.**
  - Big picture: a pipeline bakes nearly all of GL's mutable draw state into one immutable object. Dynamic rendering means no render pass objects.
  - An SVG of the pipeline stages with the fixed-function state boxes labelled with the code's settings.
  - Deep dive: pipeline.rs in full order. Vertex input bindings/attributes matching the mesh vertex struct (link to `#rust-repr-c`), rasterizer, depth, blending, multisample, dynamic state (viewport/scissor), `PipelineRenderingCreateInfo` formats, and the pipeline layout (put `id="pipeline-layout"` on its heading).
  - GL callouts: glEnable/glBlendFunc and friends, and "why so much up front".

- [ ] **Step 3: Chapter 8.**
  - Big picture: how a shader finds its textures and uniforms. Descriptor set layouts, pools and sets as a "table of pointers", plus push constants for small per-draw data.
  - An SVG mapping shader `layout(set=, binding=)` to descriptor set slots to the actual texture/buffer.
  - Deep dive: material.rs, materialmanager.rs, the texture cache and shared sampler, and the ownership rule that "material descriptors must be released before cache teardown". Put `id="descriptor-sets"` on the relevant heading.
  - GL callout: glUniform / glActiveTexture / glBindTexture units.

- [ ] **Step 4: Run the checker.** Expected: PASS apart from forward links.
- [ ] **Step 5: Run the skimmer test.**
- [ ] **Step 6: Hand over.** No commit.

### Task 5: Part 4, Making frames (chapters 9–10)

**Files:** Create `09-frame-loop.html`, `10-render-graph.html` in `docs/guide/`

**Interfaces:**
- Produces: `09-frame-loop.html#frames-in-flight` and `10-render-graph.html#barriers`, which chapters 11–16 link to.

**Sources:** ch9: `src/vulkan/base.rs`. ch10: `src/vulkan/render_graph.rs`, `src/vulkan/attachments.rs`.

- [ ] **Step 1: Chapter 9.**
  - Big picture: CPU and GPU run asynchronously. Command buffers are recorded lists. Fences mean "CPU waits for GPU" and semaphores mean "GPU waits for GPU". Frames in flight.
  - An SVG timeline with CPU and GPU lanes, showing two frames in flight and where fence waits and semaphore signals happen.
  - Deep dive: base.rs frame begin/end, acquire, submit, present, handling out-of-date/suboptimal, recreation at frame boundaries, zero-size window skip, and why render failure is fatal (DEVELOPMENT.md). Teardown in base.rs. Put `id="frames-in-flight"` on the relevant heading.
  - GL callouts: glFinish/glFlush and the implicit sync GL did.

- [ ] **Step 2: Chapter 10.**
  - Big picture: images have layouts. Barriers tell the GPU "finish writing X before reading it as Y". The render graph here is a linear pass scheduler (shadow → main → UI) that records transitions for you (DEVELOPMENT.md: not a general graph compiler). Attachments: depth, MSAA colour, resolve.
  - An SVG of the passes in sequence, with each image's layout changing between them.
  - Deep dive: render_graph.rs pass declarations, how transitions are computed and recorded (`synchronization2` barriers, if used), and attachments.rs (creation, MSAA resolve, recreation on resize). Put `id="barriers"` on the relevant heading.
  - GL callout: GL tracked all of this for you, and paid for it.

- [ ] **Step 3: Run the checker.** Expected: PASS apart from forward links.
- [ ] **Step 4: Run the skimmer test.**
- [ ] **Step 5: Hand over.** No commit.

### Task 6: Part 5a, The scene (chapters 11–13)

**Files:** Create `11-shadows.html`, `12-main-pass-scene.html`, `13-procedural-world.html` in `docs/guide/`

**Sources:**
- ch11: `src/vulkan/shadow_pass.rs`, `src/graphics/shadow_math.rs`, `assets/shaders/shadow_depth.vert`, plus the shadow sampling in `main.frag`.
- ch12: `src/vulkan/main_pass.rs`, `src/graphics/camera.rs`, `meshmanager.rs`, `assimp_loader.rs`, `import.rs`, `src/app/scene.rs`, and the material shaders (`vertex_color.*`, `infinite_plane.*`).
- ch13: `src/graphics/terrain.rs`, `sky.rs`, `rocks.rs`, and shaders `terrain.*`, `sky.*`, `rock.*`.

- [ ] **Step 1: Chapter 11.**
  - Big picture: shadow mapping is "render depth from the sun". Cascades split the view frustum so near shadows stay sharp. Reference the screenshot's Shadow Cascades panel, with the image embedded from `images/screenshot.jpg`.
  - An SVG of the camera frustum sliced into 4 cascades, each with its own light-space box.
  - Deep dive: the shadow_math.rs split scheme and light matrix fitting (and its tests), the shadow_pass.rs depth-only pipeline, array layers per cascade, depth bias, and the sampling in the shader.

- [ ] **Step 2: Chapter 12.**
  - Big picture: the main pass draws everything. The camera builds view/projection matrices. Meshes are loaded through assimp (FBX/GLB) or glTF and deduplicated in a manager. The scene is a list of what to draw.
  - An SVG of the data flow from file → loader → MeshManager / MaterialManager → scene record → draw call.
  - Deep dive: camera.rs (Vulkan's clip space versus GL's: Y flip and 0..1 depth, with a GL callout), the meshmanager and assimp_loader highlights, import.rs, scene.rs construction, and the main_pass.rs draw loop (binding the pipeline, descriptor sets, push constants, `cmd_draw_indexed`).

- [ ] **Step 3: Chapter 13.**
  - Big picture: terrain is a heightfield generated on the CPU, the sky is a fullscreen panorama, and rocks are scattered instances.
  - An SVG of how the terrain grid is laid out, or of the sky's fullscreen triangle technique (whichever the code actually does).
  - Deep dive: terrain.rs generation and normals, sky.rs (equirectangular sampling), and rocks.rs placement. Link to `05-gpu-memory.html#staging` for uploads.

- [ ] **Step 4: Run the checker.** Expected: PASS apart from forward links.
- [ ] **Step 5: Run the skimmer test.**
- [ ] **Step 6: Hand over.** No commit.

### Task 7: Part 5b + 6, Grass, ImGui, the app shell (chapters 14–16)

**Files:** Create `14-grass.html`, `15-imgui.html`, `16-app-shell.html` in `docs/guide/`

**Sources:**
- ch14: `src/graphics/grass.rs`, `grass/generation.rs`, `upload.rs`, `pipeline.rs`, `wind.rs`, `tests.rs`, and shaders `grass.vert`, `grass.frag`, `grass_mid.frag`.
- ch15: `src/vulkan/imgui_renderer.rs`, `imgui_renderer/pipeline.rs`, `imgui_renderer/resources.rs`, `src/vulkan/ui_pass.rs`, and `imgui.vert/frag`.
- ch16: `src/app/app.rs`, `src/app/input.rs`, `src/app/mod.rs`, `scripts/package_macos.sh`, `PACKAGING.md`, `packaging/macos/*`.

- [ ] **Step 1: Chapter 14.**
  - Big picture: why millions of blades are drawn with instancing (one blade mesh, many per-instance records), LOD tiers (near/mid), and wind animated in the vertex shader.
  - An SVG of a blade mesh × per-instance buffer → GPU, with the LOD rings around the camera.
  - Deep dive: candidate generation and density, the upload (link to `#staging`), the instance attribute layout, the pipeline differences from ch 7, the wind parameters (CPU side and shader side), and how the tests pin the geometry. Rust callout on `#[cfg(test)]` and `mod tests`.

- [ ] **Step 2: Chapter 15.**
  - Big picture: ImGui produces vertex lists every frame, and this renderer turns them into draws in its own pass on top of the scene. It reads the screenshot's panels as examples.
  - Deep dive: the font atlas texture, per-frame dynamic vertex/index buffers (linking `#frames-in-flight`), scissor rects per draw command, how the cascade preview textures are shown in the UI, and ui_pass.rs.
  - Aside: DEVELOPMENT.md says ImGui resource creation lacks complete failure guards.

- [ ] **Step 3: Chapter 16.**
  - Big picture: how winit drives everything (ApplicationHandler, resumed/window_event/about_to_wait), how input becomes camera movement, and how resize, minimise, fullscreen and the MSAA toggle are handled. Shutdown order. Packaging as a .app bundle with KosmicKrisp inside.
  - An SVG of the event loop, with arrows to the render paths.
  - Deep dive: app.rs (new, run, the event handlers, the resume guard, App::drop ordering, linking `#rust-drop`), input.rs, and package_macos.sh with the Info.plist and ICD JSON.
  - Closing section: "Where to go next", covering DEVELOPMENT.md's areas needing work and the checks to run.

- [ ] **Step 4: Run the checker on all pages, with coverage.** `python3 <scratchpad>/check_guide.py --coverage docs/guide/*.html`. Expected: every link resolves now that all pages exist, every excerpt matches, every page ≤ 2400 words, and the coverage list is empty. Fold any uncovered file into the closest chapter (at least an in-text `<code>` mention with a sentence on its role).
- [ ] **Step 5: Run the skimmer test.**
- [ ] **Step 6: Hand over.** No commit.

### Task 8: Finish: read times, README link, browser QA

**Files:** Modify `docs/guide/index.html` and all chapter `.readtime` spans, plus `README.md` (add one line after the DEVELOPMENT.md line).

- [ ] **Step 1: Set the read times.** For each chapter, take ceil((prose words + code lines × 2) / 230) minutes. Write that into the chapter `.readtime` and the matching TOC card. Every value must be ≤ 10. If one exceeds 10, trim its prose.
- [ ] **Step 2: Add the README link.** Add the line: `See [docs/guide/index.html](docs/guide/index.html) for a chapter-by-chapter guide to how the renderer works.`
- [ ] **Step 3: Re-run the full checker with `--coverage`.** Expected: all PASS.
- [ ] **Step 4: Browser QA.** In the built-in browser, open the index and chapters 1, 7, 9 and 14 at desktop width in light and dark, and at the 375px mobile preset. Check that:
  - no page-level horizontal scroll (`document.documentElement.scrollWidth <= innerWidth` via javascript_tool)
  - the SVGs are legible in both themes
  - code is highlighted
  - prev/next works

  Reset the viewport to desktop afterwards.
- [ ] **Step 5: Hand over.** Report the file list and what was verified. Commit only when the user asks.
