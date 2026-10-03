# RustyVulkan

An educational Rust renderer using Ash, winit, and Vulkan dynamic rendering.
The scene demonstrates procedural terrain, instanced vegetation, cascaded
shadows, shared textures, and an ImGui interface.

RustyVulkan uses the KosmicKrisp Vulkan-on-Metal driver on macOS.

## macOS requirements

- Apple Silicon
- macOS 26 or newer (KosmicKrisp requires Metal 4)
- Xcode Command Line Tools and Rust
- The macOS Vulkan SDK (the **System Global Installation** option is recommended)
- Homebrew package `assimp`

Install the Homebrew dependencies with:

```sh
brew install assimp
```

Verify the driver before running the project:

```sh
VK_DRIVER_FILES=/usr/local/share/vulkan/icd.d/libkosmickrisp_icd.json \
    vulkaninfo --summary
```

Then run normally:

```sh
cargo run
```

The application selects the SDK's KosmicKrisp ICD automatically. It checks
`$VULKAN_SDK/share/vulkan/icd.d` first and then the System Global Installation
under `/usr/local`. If the global option was not installed, source the SDK's
`setup-env.sh` before building. An existing `VK_DRIVER_FILES` value is preserved
so a driver can still be selected explicitly for diagnostics.

See [PACKAGING.md](PACKAGING.md) for creating a standalone macOS application.
See [DEVELOPMENT.md](DEVELOPMENT.md) for the code map, ownership rules, and checks.
