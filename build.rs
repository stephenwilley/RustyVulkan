// --------------------------------------------------------------------------------------
// build.rs – Shader Compilation Step
//
// Created: July 2025
// Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
// Updated: August 2025
//
// This build script now uses the `slangc` command line compiler to translate
// Slang shader files located in `assets/shaders/` into SPIR-V binaries.  Any
// file with the extension `.slang` is parsed.  The expected naming convention is
// `<name>.<stage>.slang` where `<stage>` is one of `vert`, `frag`, `comp`,
// `geom`, `tesc`, or `tese`.  The compiled binaries are written to
// `assets/shaders/spv/<name>.<stage>.spv`.
//
// The script preserves the auto-build behaviour: shaders are rebuilt whenever
// their source changes and a friendly warning is printed if `slangc` is not
// available on the build machine.
// --------------------------------------------------------------------------------------

use std::{fs, path::Path, process::Command};

fn compile_shaders() {
    let shader_dir = Path::new("assets/shaders");

    // Ensure the slang compiler exists.  If not, emit a warning and bail out so
    // that builds can continue (precompiled shaders may already exist).
    if Command::new("slangc").arg("--version").status().is_err() {
        println!("cargo:warning=slangc not found - skipping shader compilation");
        return;
    }

    for entry in fs::read_dir(shader_dir).expect("Unable to read shader dir") {
        let path = entry.expect("Invalid dir entry").path();
        if path.extension().and_then(|s| s.to_str()) != Some("slang") {
            continue; // only compile .slang files
        }

        println!("📝 Compiling shader: {:?}", path);

        // File naming convention: <name>.<stage>.slang
        let file_name = path.file_name().unwrap().to_string_lossy();
        let parts: Vec<&str> = file_name.split('.').collect();
        if parts.len() < 3 {
            println!("cargo:warning=unrecognised shader filename format: {}", file_name);
            continue;
        }
        let name = parts[0];
        let stage = parts[1];

        let profile = match stage {
            "vert" => "vs_6_0",
            "frag" => "ps_6_0",
            "comp" => "cs_6_0",
            "geom" => "gs_6_0",
            "tesc" => "hs_6_0",
            "tese" => "ds_6_0",
            _ => {
                println!("cargo:warning=unknown shader stage in {}", file_name);
                continue;
            }
        };

        let spv_dir = shader_dir.join("spv");
        fs::create_dir_all(&spv_dir).unwrap();
        let spv_path = spv_dir.join(format!("{}.{}.spv", name, stage));

        let status = Command::new("slangc")
            .arg(&path)
            .arg("-target").arg("spirv")
            .arg("-profile").arg(profile)
            .arg("-entry").arg("main")
            .arg("-o").arg(&spv_path)
            .status()
            .expect("failed to run slangc");

        if !status.success() {
            panic!("slangc failed to compile {:?}", path);
        }

        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/shaders");
    compile_shaders();
    println!("📝 Shader compilation step complete");
}

