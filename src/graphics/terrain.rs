//! --------------------------------------------------------------------------------------
//! Procedural Terrain (terrain.rs)
//!
//! Builds a small, deterministic heightfield on the CPU.  The resulting `Mesh` is an
//! ordinary static mesh: Vulkan does not need to know that its vertex heights came from
//! noise rather than from a model file.
//!
//! --------------------------------------------------------------------------------------

use crate::graphics::mesh::{Mesh, Vertex};

/// Settings for one square patch of heightfield terrain, expressed in world units.
#[derive(Clone, Copy)]
pub struct TerrainSettings {
    /// Width and depth of the square terrain patch.
    pub world_size: f32,
    /// Number of quads along each edge.  There is one more vertex than quad.
    pub cells_per_side: usize,
    /// The noise height is always in `0.0..=max_height`.
    pub max_height: f32,
    /// Centre of the deliberately level area beneath the hut, in X/Z space.
    pub pad_centre: [f32; 2],
    /// Radius of the completely flat part of the pad.
    pub pad_radius: f32,
    /// Width of the smooth transition from flat pad to noisy terrain.
    pub pad_falloff: f32,
    /// Makes the noise repeatable while still allowing future scene variations.
    pub seed: u32,
}

impl Default for TerrainSettings {
    fn default() -> Self {
        Self {
            // One-metre cells keep this first terrain inexpensive while still
            // producing smooth lighting over the scene's visible area.
            world_size: 100.0,
            cells_per_side: 100,
            max_height: 0.25,
            pad_centre: [0.0, -2.0],
            pad_radius: 2.5,
            pad_falloff: 2.0,
            seed: 0x5EED_C0DE,
        }
    }
}

impl TerrainSettings {
    /// Returns the terrain surface height at this world-space X/Z position.
    ///
    /// Scene setup uses this exact function when placing objects, so an object
    /// cannot drift away from the mesh due to a separate approximation.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        terrain_height(x, z, *self)
    }
}

/// Builds the low, rolling terrain used by the outdoor scene.
///
/// Noise is sampled in world space rather than from a texture.  This makes the mesh
/// deterministic and lets us calculate a matching normal and tangent at every vertex.
pub fn build_heightfield(settings: TerrainSettings) -> Mesh {
    assert!(
        settings.world_size > 0.0,
        "terrain must have a positive size"
    );
    assert!(
        settings.cells_per_side > 0,
        "terrain needs at least one cell"
    );
    assert!(
        settings.max_height >= 0.0,
        "terrain height cannot be negative"
    );
    assert!(
        settings.pad_radius >= 0.0,
        "terrain pad radius cannot be negative"
    );
    assert!(
        settings.pad_falloff > 0.0,
        "terrain pad falloff must be positive"
    );

    let cells = settings.cells_per_side;
    let vertices_per_side = cells + 1;
    let cell_size = settings.world_size / cells as f32;
    let half_size = settings.world_size * 0.5;

    let mut mesh = Mesh {
        vertices: Vec::with_capacity(vertices_per_side * vertices_per_side),
        indices: Vec::with_capacity(cells * cells * 6),
    };

    for row in 0..vertices_per_side {
        // Rows travel toward -Z.  This preserves the existing plane's winding order,
        // so counter-clockwise triangles continue to face upward.
        let z = half_size - row as f32 * cell_size;
        for column in 0..vertices_per_side {
            let x = -half_size + column as f32 * cell_size;
            let height = settings.height_at(x, z);

            // A central difference samples the same height function on each side of
            // this vertex.  Its slope gives a normal for the smoothly lit surface.
            let slope_x = (settings.height_at(x + cell_size, z)
                - settings.height_at(x - cell_size, z))
                / (2.0 * cell_size);
            let slope_z = (settings.height_at(x, z + cell_size)
                - settings.height_at(x, z - cell_size))
                / (2.0 * cell_size);

            let normal = normalise([-slope_x, 1.0, -slope_z]);
            // The mesh UVs increase in +X and -Z, matching the original unit plane.
            let tangent = normalise([1.0, slope_x, 0.0]);
            let bitangent = normalise(cross(normal, tangent));
            // A low-frequency vertex tint gives the bare soil some uneven, earthy colour
            // without bringing another texture into this deliberately procedural scene.
            let dirt_variation = 0.72 + 0.28 * value_noise(x, z, 0.22, settings.seed.wrapping_add(9));

            mesh.vertices.push(Vertex {
                pos: [x, height, z],
                normal,
                color: [
                    0.16 * dirt_variation,
                    0.075 * dirt_variation,
                    0.025 * dirt_variation,
                ],
                // The sand material tiles ten times across this 100 m patch.
                uv: [column as f32 / cells as f32, row as f32 / cells as f32],
                tangent,
                bitangent,
            });
        }
    }

    for row in 0..cells {
        for column in 0..cells {
            let top_left = (row * vertices_per_side + column) as u32;
            let top_right = top_left + 1;
            let bottom_left = top_left + vertices_per_side as u32;
            let bottom_right = bottom_left + 1;

            mesh.indices.extend_from_slice(&[
                top_left,
                top_right,
                bottom_right,
                bottom_right,
                bottom_left,
                top_left,
            ]);
        }
    }

    mesh
}

/// Returns a deterministic height in `0.0..=max_height`, including the hut pad.
fn terrain_height(x: f32, z: f32, settings: TerrainSettings) -> f32 {
    // Three smooth layers create broad landforms with a little local variation.
    // Their weights add to one, so the final height remains within the requested range.
    let noise = 0.68 * value_noise(x, z, 0.035, settings.seed)
        + 0.24 * value_noise(x, z, 0.09, settings.seed.wrapping_add(1))
        + 0.08 * value_noise(x, z, 0.21, settings.seed.wrapping_add(2));
    let noisy_height = settings.max_height * noise;

    let dx = x - settings.pad_centre[0];
    let dz = z - settings.pad_centre[1];
    let distance_from_pad = (dx * dx + dz * dz).sqrt();
    let blend_to_terrain = smoothstep(
        settings.pad_radius,
        settings.pad_radius + settings.pad_falloff,
        distance_from_pad,
    );

    // `0.0` inside the pad keeps the hut's already-grounded base visible.  The
    // smoothstep avoids a hard circular edge where the procedural ground resumes.
    noisy_height * blend_to_terrain
}

/// Smooth, grid-based value noise.  Unlike random noise, neighbouring samples blend.
fn value_noise(x: f32, z: f32, frequency: f32, seed: u32) -> f32 {
    let x = x * frequency;
    let z = z * frequency;
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let x_fraction = x - x0 as f32;
    let z_fraction = z - z0 as f32;
    let x_blend = smoothstep(0.0, 1.0, x_fraction);
    let z_blend = smoothstep(0.0, 1.0, z_fraction);

    let lower = lerp(hash_2d(x0, z0, seed), hash_2d(x0 + 1, z0, seed), x_blend);
    let upper = lerp(
        hash_2d(x0, z0 + 1, seed),
        hash_2d(x0 + 1, z0 + 1, seed),
        x_blend,
    );
    lerp(lower, upper, z_blend)
}

/// Converts integer grid coordinates to a stable pseudo-random value in `0.0..=1.0`.
fn hash_2d(x: i32, z: i32, seed: u32) -> f32 {
    let mut bits =
        (x as u32).wrapping_mul(0x8D_A6_B343) ^ (z as u32).wrapping_mul(0xD8_16_3841) ^ seed;
    bits ^= bits >> 16;
    bits = bits.wrapping_mul(0x7F4A_7C15);
    bits ^= bits >> 15;
    bits = bits.wrapping_mul(0x846C_A68B);
    bits ^= bits >> 16;
    bits as f32 / u32::MAX as f32
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, amount: f32) -> f32 {
    a + (b - a) * amount
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalise(vector: [f32; 3]) -> [f32; 3] {
    let length = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    [vector[0] / length, vector[1] / length, vector[2] / length]
}

#[cfg(test)]
mod tests {
    use super::{build_heightfield, TerrainSettings};

    #[test]
    fn terrain_stays_low_and_keeps_the_hut_pad_level() {
        let settings = TerrainSettings::default();
        let mesh = build_heightfield(settings);

        assert_eq!(mesh.vertices.len(), 101 * 101);
        assert_eq!(mesh.indices.len(), 100 * 100 * 6);
        assert!(mesh
            .vertices
            .iter()
            .all(|vertex| (0.0..=settings.max_height).contains(&vertex.pos[1])));

        let pad_centre = mesh
            .vertices
            .iter()
            .find(|vertex| vertex.pos[0] == 0.0 && vertex.pos[2] == -2.0)
            .expect("the one-metre grid contains the pad centre");
        assert_eq!(pad_centre.pos[1], 0.0);
        assert_eq!(settings.height_at(0.0, -2.0), 0.0);
    }
}
