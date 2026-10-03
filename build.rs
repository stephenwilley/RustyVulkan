// --------------------------------------------------------------------------------------
// build.rs – Shader Compilation Step
//
// Created: August 2025
// Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//
// This build script runs automatically before compilation. It looks in `assets/shaders/`
// and compiles the GLSL files listed in `SHADERS`
// into SPIR-V binaries using the `shaderc` crate.
//
// The output `.spv` files are written alongside the originals for later loading by Vulkan.
//
// Note:
//   • Add new shader source files to `SHADERS` below
//   • Cargo reruns this script when a listed source or this script changes
//   • This runs automatically during `cargo build`
//
// This script ensures that all shaders are ready to go before linking the final binary.
// --------------------------------------------------------------------------------------

use std::{env, fs, path::Path};

// This explicit list is also the shader manifest: adding a shader here makes Cargo watch it.
// Generated SPIR-V is deliberately not watched, otherwise writing it retriggers this script.
const SHADERS: &[&str] = &[
    "grass.vert",
    "grass.frag",
    "grass_mid.frag",
    "imgui.vert",
    "imgui.frag",
    "infinite_plane.vert",
    "infinite_plane.frag",
    "lambert_no_tex.frag",
    "main.vert",
    "main.frag",
    "passthrough.vert",
    "rock.vert",
    "rock.frag",
    "shadow_depth.vert",
    "sky.vert",
    "sky.frag",
    "terrain.vert",
    "terrain.frag",
    "vertex_color.vert",
    "vertex_color.frag",
];

fn compile_shaders() {
    let shader_dir = Path::new("assets/shaders");
    let compiler = shaderc::Compiler::new().unwrap();
    let spv_dir = shader_dir.join("spv");
    fs::create_dir_all(&spv_dir).unwrap();

    for shader_name in SHADERS {
        let path = shader_dir.join(shader_name);
        println!("cargo:rerun-if-changed={}", path.display());

        let shader_kind = match path.extension().and_then(|s| s.to_str()) {
            Some("vert") => Some(shaderc::ShaderKind::Vertex),
            Some("frag") => Some(shaderc::ShaderKind::Fragment),
            Some("geom") => Some(shaderc::ShaderKind::Geometry),
            Some("comp") => Some(shaderc::ShaderKind::Compute),
            Some("tesc") => Some(shaderc::ShaderKind::TessControl),
            Some("tese") => Some(shaderc::ShaderKind::TessEvaluation),
            _ => None,
        };

        let Some(shader_kind) = shader_kind else {
            continue;
        };

        let spv_path = spv_dir.join(format!("{shader_name}.spv"));
        // Cargo already decides when inputs changed. A second timestamp cache
        // can retain stale binaries after a checkout or a compiler-option edit.

        println!("📝 Compiling shader: {:?}", path);

        let source = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("📝 Failed to read shader source: {:?}", path));

        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_target_env(shaderc::TargetEnv::Vulkan, 0);

        let binary_result = compiler
            .compile_into_spirv(
                &source,
                shader_kind,
                path.file_name().unwrap().to_str().unwrap(),
                "main",
                Some(&options),
            )
            .expect("📝 Shader compilation failed");

        fs::write(&spv_path, binary_result.as_binary_u8()).unwrap();
    }
}

fn configure_macos_vulkan_linking() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let mut candidates = Vec::new();
    if let Some(sdk) = env::var_os("VULKAN_SDK") {
        candidates.push(std::path::PathBuf::from(sdk).join("lib"));
    }
    candidates.push(std::path::PathBuf::from("/usr/local/lib"));

    let library_dir = candidates
        .into_iter()
        .find(|directory| directory.join("libvulkan.dylib").is_file())
        .expect(
            "Vulkan loader not found. Install the macOS Vulkan SDK with System Global Installation enabled, or set VULKAN_SDK.",
        );

    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", library_dir.display());
}

fn main() {
    // Tell Cargo when to rerun this build script:
    println!("cargo:rerun-if-changed=build.rs");
    configure_macos_vulkan_linking();
    compile_shaders();
    println!("📝 Shader compilation completed successfully!");
}
