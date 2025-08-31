//! --------------------------------------------------------------------------------------
//! glTF Loader (gltf_loader.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! Utilities for importing glTF scenes and converting them into meshes and materials.
//!
//! --------------------------------------------------------------------------------------

use std::error::Error;
use gltf::import;
use gltf::image::Source;
use crate::vulkan::base::VulkanBase;
use crate::graphics::meshmanager::MeshManager;
use crate::graphics::materialmanager::{MaterialManager, MaterialProperties};
use crate::graphics::mesh::Mesh;
use crate::graphics::mesh::Vertex as MeshVertex;
use cgmath::Vector3;
use mikktspace;
use mikktspace::Geometry;
use crate::app::scene::{SceneObject, ScenePart, Transform as SceneTransform};
use std::collections::HashMap;
use cgmath::{Quaternion as CQuat, Vector3 as CVec3, Rotation};


/// Inline MikkTSpace Geometry adapter for CPU-side mesh.
struct MikkGeom<'a> {
    verts: &'a mut [MeshVertex],
    idxs:  &'a [u32],
}

impl<'a> Geometry for MikkGeom<'a> {
    fn num_faces(&self) -> usize {
        self.idxs.len() / 3
    }
    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }
    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        let i = self.idxs[face * 3 + vert] as usize;
        self.verts[i].pos
    }
    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        let i = self.idxs[face * 3 + vert] as usize;
        self.verts[i].normal
    }
    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        let i = self.idxs[face * 3 + vert] as usize;
        self.verts[i].uv
    }
    fn set_tangent_encoded(&mut self, t: [f32; 4], face: usize, vert: usize) {
        let i = self.idxs[face * 3 + vert] as usize;
        // store tangent xyz
        self.verts[i].tangent  = [t[0], t[1], t[2]];
        // reconstruct bitangent = sign * cross(tangent, normal)
        let n = Vector3::from(self.verts[i].normal);
        let tan = Vector3::from(self.verts[i].tangent);
        let b   = tan.cross(n) * -t[3];
        self.verts[i].bitangent = b.into();
    }
}


/// Loads a glTF and returns a single SceneObject containing one ScenePart per primitive.
/// The object and all parts default to identity transforms; caller can set object transform after.
pub fn import_gltf_as_object(
    path: &str,
    vb:   &VulkanBase,
    meshes: &mut MeshManager,
    mats:   &mut MaterialManager,
) -> Result<SceneObject, Box<dyn Error>> {
    // Load the document and buffers
    let base_dir = std::path::Path::new(path)
        .parent()
        .unwrap_or_else(|| std::path::Path::new(""))
        .to_path_buf();
    let (doc, buffers, _) = import(path)?;

    // 1) Materials: same approach as import_gltf
    let mut material_ids = Vec::new();
    for (i, mat) in doc.materials().enumerate() {
        let name = mat.name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("gltf_mat{}", i));
        let pbr = mat.pbr_metallic_roughness();
        let base_color = pbr
            .base_color_texture()
            .map(|t| match t.texture().source().source() {
                Source::Uri { uri, .. } => base_dir.join(uri)
                    .to_string_lossy()
                    .into_owned(),
                Source::View { .. }  => panic!("Embedded glTF images not supported"),
            });
        let normal_map = mat
            .normal_texture()
            .map(|t| match t.texture().source().source() {
                Source::Uri { uri, .. } => base_dir.join(uri)
                    .to_string_lossy()
                    .into_owned(),
                Source::View { .. }  => panic!("Embedded glTF images not supported"),
            });

        let mat_id = mats.request_material(
            vb,
            MaterialProperties {
                name,
                vs_path: "assets/shaders/spv/main.vert.spv".into(),
                fs_path: "assets/shaders/spv/main.frag.spv".into(),
                diffuse_texture_path: base_color,
                normalmap_texture_path: normal_map,
                depth_write: true,
                uv_tiling: None,
            }
        );
        material_ids.push(mat_id);
    }
    if material_ids.is_empty() {
        let default_id = mats.request_material(
            vb,
            MaterialProperties {
                name: "default_mat".into(),
                vs_path: "assets/shaders/spv/passthrough.vert.spv".into(),
                fs_path: "assets/shaders/spv/lambert_no_tex.frag.spv".into(),
                diffuse_texture_path: None,
                normalmap_texture_path: None,
                depth_write: true,
                uv_tiling: None,
            }
        );
        material_ids.push(default_id);
    }

    // 2) Build meshes once per glTF primitive and index them by (mesh_index, primitive_index)
    let mut prim_mesh_map: HashMap<(usize, usize), usize> = HashMap::new();
    for mesh in doc.meshes() {
        let m_idx = mesh.index();
        for (pi, prim) in mesh.primitives().enumerate() {
            let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<[f32;3]> = reader.read_positions()
                .ok_or("Positions missing")?
                .collect();
            let normals: Vec<[f32;3]> = reader.read_normals()
                .ok_or("Normals missing")?
                .collect();
            let uvs: Vec<[f32;2]> = reader.read_tex_coords(0)
                .map(|tc| tc.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0,0.0]; positions.len()]);
            let tangents: Option<Vec<[f32;4]>> =
                reader.read_tangents().map(|t| t.collect());

            let indices: Vec<u32> = if let Some(index_iter) = reader.read_indices() {
                index_iter.into_u32().collect()
            } else {
                (0..positions.len() as u32).collect()
            };

            let mut cpu_mesh = Mesh::new();
            cpu_mesh.vertices.reserve(positions.len());
            let maybe_tangents = tangents;
            for i in 0..positions.len() {
                let (tan, bitan) = if let Some(ts) = maybe_tangents.as_ref() {
                    let t4 = ts[i];
                    let t = [t4[0], t4[1], t4[2]];
                    let bsign = t4[3];
                    let n = normals[i];
                    let b = [
                        bsign * (n[1]*t[2] - n[2]*t[1]),
                        bsign * (n[2]*t[0] - n[0]*t[2]),
                        bsign * (n[0]*t[1] - n[1]*t[0]),
                    ];
                    (t, b)
                } else {
                    ([0.0;3], [0.0;3])
                };
                cpu_mesh.vertices.push(crate::graphics::mesh::Vertex {
                    pos:        positions[i],
                    normal:     normals[i],
                    color:      [1.0,1.0,1.0],
                    uv:         uvs[i],
                    tangent:    tan,
                    bitangent:  bitan,
                });
            }
            cpu_mesh.indices = indices;

            if maybe_tangents.is_none() {
                let orig_vertices = std::mem::take(&mut cpu_mesh.vertices);
                let orig_indices  = cpu_mesh.indices.clone();
                let mut uw_vertices = Vec::with_capacity(orig_indices.len());
                let mut uw_indices  = Vec::with_capacity(orig_indices.len());
                for tri in orig_indices.chunks(3) {
                    for &idx in tri {
                        let v = orig_vertices[idx as usize];
                        uw_vertices.push(v);
                        uw_indices.push((uw_vertices.len() - 1) as u32);
                    }
                }
                cpu_mesh.vertices = uw_vertices;
                cpu_mesh.indices  = uw_indices;

                let mut geom = MikkGeom {
                    verts: &mut cpu_mesh.vertices,
                    idxs:  &cpu_mesh.indices,
                };
                if !mikktspace::generate_tangents(&mut geom) {
                    panic!("MikkTSpace tangent generation failed");
                }
            }

            for v in cpu_mesh.vertices.iter_mut() {
                v.uv[1] = 1.0 - v.uv[1];
            }

            let mesh_name = mesh.name()
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("gltf_mesh#{}", pi));
            let mesh_id = meshes.request_mesh_from_cpu(mesh_name, vb, cpu_mesh)?;
            prim_mesh_map.insert((m_idx, pi), mesh_id);
        }
    }

    // 3) Traverse node hierarchy and collect parts with accumulated transforms
    fn combine_uniform(parent: &SceneTransform, local: &SceneTransform) -> SceneTransform {
        let s = parent.scale * local.scale;
        let r = parent.rotation * local.rotation;
        let translated = parent.rotation.rotate_vector(local.translation * parent.scale);
        let t = parent.translation + translated;
        SceneTransform { translation: t, rotation: r, scale: s }
    }

    fn node_local_transform(node: &gltf::Node) -> SceneTransform {
        let (t, r, s) = node.transform().decomposed();
        let tq = CQuat::new(r[3], r[0], r[1], r[2]);
        let ts = (s[0] + s[1] + s[2]) / 3.0; // uniform approximation
        SceneTransform {
            translation: CVec3::new(t[0], t[1], t[2]),
            rotation: tq,
            scale: ts,
        }
    }

    let mut parts: Vec<ScenePart> = Vec::new();
    // Use an explicit stack for DFS
    let root_scene = doc.default_scene().or_else(|| doc.scenes().next());
    if let Some(scene) = root_scene {
        // If there is exactly one root node, factor its transform into the SceneObject and
        // make all parts relative to that root. This avoids double-scaling when the app
        // also places/scales the object.
        let roots: Vec<gltf::Node> = scene.nodes().collect();
        let mut object_transform = SceneTransform::identity();
        let mut root_inv: Option<SceneTransform> = None;
        if roots.len() == 1 {
            let rt = node_local_transform(&roots[0]);
            object_transform = rt;
            // Inverse for uniform transform
            let inv_scale = if rt.scale != 0.0 { 1.0 / rt.scale } else { 0.0 };
            let inv_rot = CQuat::conjugate(rt.rotation);
            let inv_trans = inv_rot.rotate_vector(-rt.translation * inv_scale);
            root_inv = Some(SceneTransform { translation: inv_trans, rotation: inv_rot, scale: inv_scale });
        }

        let mut stack: Vec<(gltf::Node, SceneTransform)> = Vec::new();
        for n in scene.nodes() {
            stack.push((n, SceneTransform::identity()));
        }
        while let Some((node, parent_tf)) = stack.pop() {
            let local = node_local_transform(&node);
            let world = combine_uniform(&parent_tf, &local);

            if let Some(mesh) = node.mesh() {
                let m_idx = mesh.index();
                for (pi, prim) in mesh.primitives().enumerate() {
                    // Map material
                    let mat_index = prim.material().index().unwrap_or(0);
                    let mat_id = *material_ids.get(mat_index).unwrap_or(&material_ids[0]);
                    // Mesh lookup
                    let mesh_id = *prim_mesh_map.get(&(m_idx, pi)).expect("Missing mesh for primitive");
                    let part_tf = if let Some(inv) = root_inv {
                        combine_uniform(&inv, &world)
                    } else {
                        world
                    };
                    parts.push(ScenePart { transform: part_tf, material_id: mat_id, mesh_id });
                }
            }
            for child in node.children() {
                stack.push((child, world));
            }
        }

        return Ok(SceneObject { transform: if roots.len() == 1 { object_transform } else { SceneTransform::identity() }, parts, visible: true });
    } else {
        // Fallback: some glTFs lack a default/first scene; in that case, just emit all meshes
        // at identity, matching the behavior of the previous flat importer.
        for mesh in doc.meshes() {
            let m_idx = mesh.index();
            for (pi, prim) in mesh.primitives().enumerate() {
                let mat_index = prim.material().index().unwrap_or(0);
                let mat_id = *material_ids.get(mat_index).unwrap_or(&material_ids[0]);
                let mesh_id = *prim_mesh_map.get(&(m_idx, pi)).expect("Missing mesh for primitive");
                parts.push(ScenePart { transform: SceneTransform::identity(), material_id: mat_id, mesh_id });
            }
        }
    }

    Ok(SceneObject { transform: SceneTransform::identity(), parts, visible: true })
}
