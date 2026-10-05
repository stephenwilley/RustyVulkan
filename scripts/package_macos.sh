#!/bin/zsh

# Build a Finder-launchable Apple Silicon application and a shareable zip.
# Homebrew is needed only for Assimp on the machine assembling the bundle. The
# Vulkan loader and KosmicKrisp are copied from the Vulkan SDK.
set -euo pipefail

SCRIPT_DIR=${0:A:h}
PROJECT_DIR=${SCRIPT_DIR:h}
DIST_DIR="$PROJECT_DIR/dist"
APP_BUNDLE="$DIST_DIR/RustyVulkan.app"
CONTENTS="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RESOURCES_DIR="$CONTENTS/Resources"
FRAMEWORKS_DIR="$CONTENTS/Frameworks"
ZIP_PATH="$DIST_DIR/RustyVulkan-macos-arm64.zip"

if [[ $(uname -m) != "arm64" ]]; then
    print -u2 "This script currently packages the Apple Silicon build only."
    exit 1
fi

cd "$PROJECT_DIR"
cargo build --release --locked

ASSIMP_PREFIX=$(brew --prefix assimp)
ASSIMP_SOURCE="$ASSIMP_PREFIX/lib/libassimp.6.dylib"
if [[ -n ${VULKAN_SDK:-} && -f "$VULKAN_SDK/lib/libvulkan.1.dylib" ]]; then
    VULKAN_LIB_DIR="$VULKAN_SDK/lib"
else
    VULKAN_LIB_DIR="/usr/local/lib"
fi
VULKAN_SOURCE="$VULKAN_LIB_DIR/libvulkan.1.dylib"
KOSMICKRISP_SOURCE="$VULKAN_LIB_DIR/libvulkan_kosmickrisp.dylib"

for dependency in "$ASSIMP_SOURCE" "$VULKAN_SOURCE" "$KOSMICKRISP_SOURCE"; do
    if [[ ! -f "$dependency" ]]; then
        print -u2 "Missing required library: $dependency"
        exit 1
    fi
done

# This is the only directory the script replaces.
rm -rf "$APP_BUNDLE"
rm -f "$ZIP_PATH"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR/vulkan/icd.d" "$FRAMEWORKS_DIR"

cp target/release/RustyVulkan "$MACOS_DIR/RustyVulkan"
ditto assets "$RESOURCES_DIR/assets"
# SPIR-V is embedded in the executable. Omit any legacy source-tree build output.
rm -rf "$RESOURCES_DIR/assets/shaders/spv"
cp packaging/macos/Info.plist "$CONTENTS/Info.plist"
cp packaging/macos/libkosmickrisp_icd.json \
    "$RESOURCES_DIR/vulkan/icd.d/libkosmickrisp_icd.json"
cp "$ASSIMP_SOURCE" "$FRAMEWORKS_DIR/libassimp.6.dylib"
cp "$VULKAN_SOURCE" "$FRAMEWORKS_DIR/libvulkan.1.dylib"
cp "$KOSMICKRISP_SOURCE" "$FRAMEWORKS_DIR/libvulkan_kosmickrisp.dylib"

# Replace machine-local library references with paths inside the bundle.
VULKAN_LINK_REFERENCE=$(otool -L "$MACOS_DIR/RustyVulkan" | \
    awk '$1 ~ /libvulkan\.1(\.[0-9]+)*\.dylib$/ { print $1; exit }')
if [[ -z "$VULKAN_LINK_REFERENCE" ]]; then
    print -u2 "Could not find the Vulkan loader reference in the executable."
    exit 1
fi
install_name_tool -change "$ASSIMP_SOURCE" \
    @executable_path/../Frameworks/libassimp.6.dylib "$MACOS_DIR/RustyVulkan"
install_name_tool -change "$VULKAN_LINK_REFERENCE" \
    @executable_path/../Frameworks/libvulkan.1.dylib "$MACOS_DIR/RustyVulkan"
install_name_tool -id @executable_path/../Frameworks/libassimp.6.dylib \
    "$FRAMEWORKS_DIR/libassimp.6.dylib"
install_name_tool -id @executable_path/../Frameworks/libvulkan.1.dylib \
    "$FRAMEWORKS_DIR/libvulkan.1.dylib"
install_name_tool -id @executable_path/../Frameworks/libvulkan_kosmickrisp.dylib \
    "$FRAMEWORKS_DIR/libvulkan_kosmickrisp.dylib"

# Ad-hoc signing prevents modified-binary errors. A publicly distributed build
# would additionally need an Apple Developer ID signature and notarisation.
codesign --force --sign - "$FRAMEWORKS_DIR/libassimp.6.dylib"
codesign --force --sign - "$FRAMEWORKS_DIR/libvulkan.1.dylib"
codesign --force --sign - "$FRAMEWORKS_DIR/libvulkan_kosmickrisp.dylib"
codesign --force --deep --sign - "$APP_BUNDLE"

if otool -L "$MACOS_DIR/RustyVulkan" | grep -q '/opt/homebrew'; then
    print -u2 "The packaged executable still refers to a Homebrew library."
    exit 1
fi
codesign --verify --deep --strict "$APP_BUNDLE"

ditto -c -k --sequesterRsrc --keepParent "$APP_BUNDLE" "$ZIP_PATH"
print "Created: $APP_BUNDLE"
print "Created: $ZIP_PATH"
