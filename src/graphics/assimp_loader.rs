//! --------------------------------------------------------------------------------------
//! Assimp Loader (assimp_loader.rs)
//!
//! Created: September 2025
//! Author: Stephen Willey
//!
//! Utilities for importing generic 3D formats using russimp/Assimp and converting them
//! into meshes/materials + a `SceneObject` compatible with the engine.
//!
//! This module is feature-gated behind `russimp`. Without the feature, the main import
//! function returns an error at runtime explaining how to enable it.
//!
//! --------------------------------------------------------------------------------------

use std::error::Error;
use crate::vulkan::base::VulkanBase;
use crate::graphics::meshmanager::MeshManager;
use crate::graphics::materialmanager::MaterialManager;
use crate::app::scene::{SceneObject, ScenePart, Transform as SceneTransform};

/// Import a model via russimp/Assimp into a single `SceneObject`.
/// When the `russimp` feature is disabled, this returns an error at runtime.
#[cfg(not(feature = "russimp"))]
pub fn import_model_as_object(
    path: &str,
    _vb: &VulkanBase,
    _meshes: &mut MeshManager,
    _mats: &mut MaterialManager,
) -> Result<SceneObject, Box<dyn Error>> {
    Err(format!(
        "russimp feature not enabled. Enable feature 'russimp' to import '{}'.",
        path
    )
    .into())
}

/// russimp/Assimp-backed importer (skeleton): load scene, build meshes/materials, walk nodes.
///
/// NOTE: Implementation to be filled in next pass. The signature and flow match glTF importer
/// so the app stays decoupled. This compiles when `russimp` is enabled and can be completed
/// without wider changes.
#[cfg(feature = "russimp")]
pub fn import_model_as_object(
    path: &str,
    vb: &VulkanBase,
    meshes: &mut MeshManager,
    mats: &mut MaterialManager,
) -> Result<SceneObject, Box<dyn Error>> {
    use cgmath::{Quaternion as CQuat, Vector3 as CVec3, Matrix3};
    use cgmath::{Rotation, InnerSpace};

    // 1) Load the scene with typical preprocessing flags
    use russimp::scene::{Scene, PostProcess};
    let scene = Scene::from_file(
        path,
        vec![
            PostProcess::Triangulate,
            PostProcess::CalculateTangentSpace,
            PostProcess::ImproveCacheLocality,
        ],
    )?;

    // Resolve relative texture paths against the model's directory
    use std::path::Path;
    let base_path = Path::new(path);
    let base_dir = base_path
        .parent()
        .unwrap_or_else(|| Path::new(""));
    let model_stem: String = base_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());

    // 2) Materials: map to your MaterialManager (basic albedo + normal for now)
    let mut material_ids: Vec<usize> = Vec::with_capacity(scene.materials.len());
    for (i, mat) in scene.materials.iter().enumerate() {
        let name = format!("{}_mat{}", model_stem, i);

        // Best-effort fetch of common texture types; PBR expansion comes next
        use russimp::material::TextureType;
        const FILENAME_PROPERTY: &str = "$tex.file";
        // Helper to fetch external texture filename from material properties
        let find_tex_path = |semantic: TextureType| -> Option<String> {
            // Find first property with key "$tex.file" and matching semantic
            let prop = mat
                .properties
                .iter()
                .find(|p| p.key == FILENAME_PROPERTY && p.semantic == semantic)?;
            if let russimp::material::PropertyTypeInfo::String(p) = &prop.data {
                let pth = Path::new(p);
                let abs = if pth.is_absolute() {
                    p.clone()
                } else {
                    base_dir.join(pth).to_string_lossy().into_owned()
                };
                Some(abs)
            } else {
                None
            }
        };
        // Prefer BaseColor then Diffuse for albedo
        let diffuse_tex = find_tex_path(TextureType::BaseColor)
            .or_else(|| find_tex_path(TextureType::Diffuse));
        let normal_tex = find_tex_path(TextureType::Normals);

        // Choose shader based on texture availability
        let use_textured = diffuse_tex.is_some();
        let (vs_path, fs_path) = if use_textured {
            ("assets/shaders/spv/main.vert.spv".into(),
             "assets/shaders/spv/main.frag.spv".into())
        } else {
            ("assets/shaders/spv/passthrough.vert.spv".into(),
             "assets/shaders/spv/lambert_no_tex.frag.spv".into())
        };

        let id = mats.request_material(
            vb,
            crate::graphics::materialmanager::MaterialProperties {
                name,
                vs_path,
                fs_path,
                diffuse_texture_path: if use_textured { diffuse_tex } else { None },
                normalmap_texture_path: if use_textured { normal_tex } else { None },
                depth_write: true,
                uv_tiling: None,
            },
        );
        material_ids.push(id);
    }
    if material_ids.is_empty() {
        let default_id = mats.request_material(
            vb,
            crate::graphics::materialmanager::MaterialProperties {
                name: "assimp_default_mat".into(),
                vs_path: "assets/shaders/spv/passthrough.vert.spv".into(),
                fs_path: "assets/shaders/spv/lambert_no_tex.frag.spv".into(),
                diffuse_texture_path: None,
                normalmap_texture_path: None,
                depth_write: true,
                uv_tiling: None,
            },
        );
        material_ids.push(default_id);
    }

    // 3) Build meshes for each Assimp mesh; store mesh_id by index
    let mut mesh_ids: Vec<usize> = Vec::with_capacity(scene.meshes.len());
    for (mi, ai_mesh) in scene.meshes.iter().enumerate() {
        let mut cpu = crate::graphics::mesh::Mesh::new();
        cpu.vertices.reserve(ai_mesh.vertices.len());

        // Extract vertex attributes with safe fallbacks
        for vi in 0..ai_mesh.vertices.len() {
            let p = &ai_mesh.vertices[vi];
            let n = ai_mesh
                .normals
                .get(vi)
                .cloned()
                .unwrap_or(russimp::Vector3D { x: 0.0, y: 1.0, z: 0.0 });
            let uv = ai_mesh
                .texture_coords
                .get(0)
                .and_then(|tc| tc.as_ref())
                .and_then(|tc| tc.get(vi))
                .cloned()
                .unwrap_or(russimp::Vector3D { x: 0.0, y: 0.0, z: 0.0 });
            let tan = ai_mesh
                .tangents
                .get(vi)
                .cloned()
                .unwrap_or(russimp::Vector3D { x: 0.0, y: 0.0, z: 0.0 });
            let bit = ai_mesh
                .bitangents
                .get(vi)
                .cloned()
                .unwrap_or(russimp::Vector3D { x: 0.0, y: 0.0, z: 0.0 });

            cpu.vertices.push(crate::graphics::mesh::Vertex {
                pos: [p.x as f32, p.y as f32, p.z as f32],
                normal: [n.x as f32, n.y as f32, n.z as f32],
                color: [1.0, 1.0, 1.0],
                uv: [uv.x as f32, uv.y as f32],
                tangent: [tan.x as f32, tan.y as f32, tan.z as f32],
                bitangent: [bit.x as f32, bit.y as f32, bit.z as f32],
            });
        }
        // Indices
        let mut indices = Vec::with_capacity(ai_mesh.faces.len() * 3);
        for f in &ai_mesh.faces {
            // Triangulate flag ensures 3 indices per face
            for idx in &f.0 {
                indices.push(*idx as u32);
            }
        }
        cpu.indices = indices;

        let name = if ai_mesh.name.is_empty() {
            format!("{}_mesh#{}", model_stem, mi)
        } else {
            format!("{}_{}", model_stem, ai_mesh.name)
        };
        let mesh_id = meshes.request_mesh_from_cpu(name, vb, cpu)?;
        mesh_ids.push(mesh_id);
    }

    // 4) Traverse nodes to gather parts with transforms
    fn decompose(transform: &russimp::Matrix4x4) -> SceneTransform {
        // Interpret as row-major with last column = translation (a4,b4,c4)
        let t = CVec3::new(transform.a4 as f32, transform.b4 as f32, transform.c4 as f32);
        // Upper-left 3x3 contains rotation*scale. Treat its columns as basis vectors.
        let c0 = CVec3::new(transform.a1 as f32, transform.b1 as f32, transform.c1 as f32);
        let c1 = CVec3::new(transform.a2 as f32, transform.b2 as f32, transform.c2 as f32);
        let c2 = CVec3::new(transform.a3 as f32, transform.b3 as f32, transform.c3 as f32);
        let s0 = c0.magnitude();
        let s1 = c1.magnitude();
        let s2 = c2.magnitude();
        // Uniform scale approximation to match SceneTransform design
        let scale = ((s0 + s1 + s2) / 3.0).max(1e-6);
        // Normalize columns to obtain rotation matrix
        let r0 = if s0 > 0.0 { c0 / s0 } else { CVec3::new(1.0, 0.0, 0.0) };
        let r1 = if s1 > 0.0 { c1 / s1 } else { CVec3::new(0.0, 1.0, 0.0) };
        let r2 = if s2 > 0.0 { c2 / s2 } else { CVec3::new(0.0, 0.0, 1.0) };
        let rot_m = Matrix3::from_cols(r0, r1, r2);
        let rot = CQuat::from(rot_m);
        SceneTransform { translation: t, rotation: rot, scale }
    }

    let mut parts: Vec<ScenePart> = Vec::new();
    let mut object_transform = SceneTransform::identity();
    if let Some(root) = &scene.root {
        // Factor the root node's transform into the object transform to preserve authoring scale/rot.
        let root_tf = decompose(&root.transformation);
        object_transform = root_tf;
        // Compute inverse uniform transform so parts are relative to object root.
        let inv_scale = if root_tf.scale != 0.0 { 1.0 / root_tf.scale } else { 0.0 };
        let inv_rot = CQuat::conjugate(root_tf.rotation);
        let inv_trans = inv_rot.rotate_vector(-root_tf.translation * inv_scale);
        let root_inv = SceneTransform { translation: inv_trans, rotation: inv_rot, scale: inv_scale };

        // DFS from the root, accumulating transforms.
        let mut stack: Vec<(std::rc::Rc<russimp::node::Node>, SceneTransform)> =
            vec![(root.clone(), SceneTransform::identity())];
        while let Some((node_rc, parent_tf)) = stack.pop() {
            let node = node_rc.as_ref();
            let local = decompose(&node.transformation);
            let s = parent_tf.scale * local.scale;
            let r = parent_tf.rotation * local.rotation;
            let translated = parent_tf.rotation.rotate_vector(local.translation * parent_tf.scale);
            let t = parent_tf.translation + translated;
            let world = SceneTransform { translation: t, rotation: r, scale: s };

            // Part transform relative to the root
            let rel = {
                let s = root_inv.scale * world.scale;
                let r = root_inv.rotation * world.rotation;
                let translated = root_inv.rotation.rotate_vector(world.translation * root_inv.scale);
                let t = root_inv.translation + translated;
                SceneTransform { translation: t, rotation: r, scale: s }
            };

            for &mi in &node.meshes {
                let mesh_id = *mesh_ids.get(mi as usize).ok_or("Missing mesh id")?;
                let mat_index = scene.meshes[mi as usize].material_index as usize;
                let mat_id = *material_ids.get(mat_index).unwrap_or(&material_ids[0]);
                parts.push(ScenePart { transform: rel, material_id: mat_id, mesh_id });
            }
            for child in node.children.borrow().iter() {
                stack.push((child.clone(), world));
            }
        }
    }

    println!(
        "[assimp] Imported '{}' => {} meshes, {} materials, {} parts",
        path,
        mesh_ids.len(),
        material_ids.len(),
        parts.len()
    );
    Ok(SceneObject { transform: object_transform, parts, visible: true })
}
