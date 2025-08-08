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


// Inline MikkTSpace Geometry adapter for CPU-side mesh
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

/// A single primitive from a glTF file, tied to a mesh and a material.
pub struct GltfPrim {
    pub mesh_id: usize,
    pub mat_id:  usize,
}

/// Loads a glTF file, imports its geometry and materials into your managers,
/// and returns a list of primitives you can add to your scene.
/// # Arguments
/// * `path` - The path to the glTF file.
/// * `vb` - The VulkanBase struct.
/// * `meshes` - The MeshManager to use for creating meshes.
/// * `mats` - The MaterialManager to use for creating materials.
/// # Returns
/// * `Result<Vec<GltfPrim>, Box<dyn Error>>` - Returns a list of primitives on success, or an error on failure.
pub fn import_gltf(
    path: &str,
    vb:   &VulkanBase,
    meshes: &mut MeshManager,
    mats:   &mut MaterialManager,
) -> Result<Vec<GltfPrim>, Box<dyn Error>> {
    let base_dir = std::path::Path::new(path)
        .parent()
        .unwrap_or_else(|| std::path::Path::new(""))
        .to_path_buf();
    // 1) Read and parse the file
    let (doc, buffers, _) = import(path)?;

    // 2) Import materials, giving fallback names if unnamed
    let mut material_ids = Vec::new();
    for (i, mat) in doc.materials().enumerate() {
        let name = mat.name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("gltf_mat{}", i));
        // Base-color and normal textures are optional
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
        println!("GLTF Material: {:?}, {:?}", base_color, normal_map);

        let mat_id = mats.request_material(
            vb,
            MaterialProperties {
                name,
                vs_path: "assets/shaders/spv/point_light.vert.spv".into(),
                fs_path: "assets/shaders/spv/point_light.frag.spv".into(),
                diffuse_texture_path: base_color,
                normalmap_texture_path: normal_map,
                depth_write: true
            }
        );
        material_ids.push(mat_id);
    }
    // If no materials were defined, create a single default material
    if material_ids.is_empty() {
        let default_id = mats.request_material(
            vb,
            MaterialProperties {
                name: "default_mat".into(),
                vs_path: "assets/shaders/spv/passthrough.vert.spv".into(),
                fs_path: "assets/shaders/spv/lambert_no_tex.frag.spv".into(),
                diffuse_texture_path: None,
                normalmap_texture_path: None,
                depth_write: true
            }
        );
        material_ids.push(default_id);
    }

    // 3) Import meshes & build primitives
    let mut prims = Vec::new();
    for mesh in doc.meshes() {
        for (pi, prim) in mesh.primitives().enumerate() {
            // Build a CPU-side Mesh from this primitive
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
            let maybe_tangents = tangents;
            // Read indices if provided, otherwise generate a sequential index for each vertex
            let indices: Vec<u32> = if let Some(index_iter) = reader.read_indices() {
                index_iter.into_u32().collect()
            } else {
                // No indices: assume triangle list with each vertex in order
                (0..positions.len() as u32).collect()
            };

            let mut cpu_mesh = Mesh::new();
            cpu_mesh.vertices.reserve(positions.len());
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
                    // Gotta put something in there but we'll overwrite these zeroes shortly with
                    // computed tangents
                    ([0.0;3], [0.0;3])
                };
                cpu_mesh.vertices.push(crate::graphics::mesh::Vertex {
                    pos:        positions[i],
                    normal:     normals[i],
                    color:      [1.0,1.0,1.0],
                    // Store original UVs for tangent calc; flip V later for Vulkan
                    uv:         uvs[i],
                    tangent:    tan,
                    bitangent:  bitan,
                });
            }
            // Assign original indices
            cpu_mesh.indices = indices;

            // A lot of the following code took inspiration from the gltf-transform code to prevent seams
            if maybe_tangents.is_none() {
                // Unweld: duplicate vertices per triangle to avoid shared-vertex smoothing across UV seams
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

                // Generate tangents on the unwelded mesh
                let mut geom = MikkGeom {
                    verts: &mut cpu_mesh.vertices,
                    idxs:  &cpu_mesh.indices,
                };
                if !mikktspace::generate_tangents(&mut geom) {
                    panic!("MikkTSpace tangent generation failed");
                }
            }

            // Now flip V coordinate for Vulkan sampling
            for v in cpu_mesh.vertices.iter_mut() {
                v.uv[1] = 1.0 - v.uv[1];
            }

            // Upload via new helper
            let mesh_name = mesh.name()
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("gltf_mesh#{}", pi));
            let mesh_id = meshes.request_mesh_from_cpu(
                mesh_name, vb, cpu_mesh)?;

            // Pick the corresponding material ID (fall back to first)
            let mat_index = prim.material().index().unwrap_or(0);
            let mat_id = *material_ids.get(mat_index).unwrap_or(&material_ids[0]);

            prims.push(GltfPrim { mesh_id, mat_id });
        }
    }

    Ok(prims)
}