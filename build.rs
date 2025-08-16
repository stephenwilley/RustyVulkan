// --------------------------------------------------------------------------------------
// build.rs – Shader Compilation Step
//
// Created: July 2025
// Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//
// This build script runs automatically before compilation. It looks in `assets/shaders/`
// and compiles any `.vert`, `.frag`, `.comp`, `.geom`, `.tesc`, or `.tese` GLSL files
// into SPIR-V binaries using the `shaderc` crate.
//
// The output `.spv` files are written alongside the originals for later loading by Vulkan.
//
// Note:
//   • Only files with recognized shader extensions are compiled
//   • Each file is recompiled only if it's changed (via cargo:rerun-if-changed)
//   • This runs automatically during `cargo build`
//
// This script ensures that all shaders are ready to go before linking the final binary.
// --------------------------------------------------------------------------------------

use std::{fs, path::Path};

fn compile_shaders() {
    let shader_dir = Path::new("assets/shaders");
    let compiler = shaderc::Compiler::new().unwrap();

    for entry in fs::read_dir(shader_dir).unwrap() {
        let path = entry.unwrap().path();
        println!("📝 Compiling shader: {:?}", path);

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
            continue; // Skip unsupported extensions
        };

        let source = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("📝 Failed to read shader source: {:?}", path));

        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_target_env(shaderc::TargetEnv::Vulkan, 0);

        let binary_result = compiler.compile_into_spirv(
            &source,
            shader_kind,
            path.file_name().unwrap().to_str().unwrap(),
            "main",
            Some(&options),
        ).expect("📝 Shader compilation failed");

        let spv_dir = shader_dir.join("spv");
        fs::create_dir_all(&spv_dir).unwrap();

        // Grab the full filename (e.g. "lambert.frag"):
        let shader_name = path.file_name().unwrap().to_string_lossy();
        // Append ".spv" to it, giving "lambert.frag.spv":
        let spv_file_name = format!("{}.spv", shader_name);
        // Build the final path:
        let spv_path = spv_dir.join(spv_file_name);
        fs::write(&spv_path, binary_result.as_binary_u8()).unwrap();
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn main() {
    // Tell Cargo when to rerun this build script:
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/shaders");
    compile_shaders();
    println!("📝 Shader compilation completed successfully!");
}