# Packaging for macOS

Run:

```sh
./scripts/package_macos.sh
```

The script builds the release executable and creates both
`dist/RustyVulkan.app` and `dist/RustyVulkan-macos-arm64.zip`. The application
contains its assets, Assimp, the Vulkan loader, and KosmicKrisp, so the receiving
Mac does not need Rust, Homebrew, or the Vulkan SDK installed.

The packaging host needs Homebrew's `assimp` package plus a Vulkan SDK
installation containing the loader and KosmicKrisp. The script first checks
`$VULKAN_SDK/lib`, then the System Global Installation location at `/usr/local/lib`.

The current package targets Apple Silicon and macOS 26 or newer. It is ad-hoc
signed rather than notarised, so a recipient may need to right-click the app
and choose **Open** the first time. Normal public distribution would require an
Apple Developer ID certificate and Apple notarisation.
