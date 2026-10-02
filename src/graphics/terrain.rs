//! --------------------------------------------------------------------------------------
//! Procedural Terrain (terrain.rs)
//!
//! Builds a small, deterministic heightfield on the CPU.  The resulting `Mesh` is an
//! ordinary static mesh: Vulkan does not need to know that its vertex heights came from
//! noise rather than from a model file.
//!
//! --------------------------------------------------------------------------------------

use crate::graphics::mesh::{Mesh, Vertex};

/// One deterministic anchor for a cluster assembled from shared low-poly rock models.
#[derive(Clone, Copy, Debug)]
pub struct RockClusterPlacement {
    pub position: [f32; 2],
    pub rotation_degrees: f32,
    pub scale: f32,
    pub variant: usize,
    /// Grass fades out before reaching the outermost stones.
    pub clearing_radius: f32,
}

/// Art-directed anchors keep the rocks useful as landmarks. Scene setup combines several
/// imported shapes at each anchor, varying their scale and rotation deterministically.
pub const ROCK_CLUSTERS: &[RockClusterPlacement] = &[
    RockClusterPlacement {
        position: [-13.0, -12.0],
        rotation_degrees: 18.0,
        scale: 1.05,
        variant: 0,
        clearing_radius: 2.4,
    },
    RockClusterPlacement {
        position: [14.0, -17.0],
        rotation_degrees: 76.0,
        scale: 0.82,
        variant: 1,
        clearing_radius: 2.0,
    },
    RockClusterPlacement {
        position: [-23.0, 7.0],
        rotation_degrees: 141.0,
        scale: 1.35,
        variant: 2,
        clearing_radius: 3.0,
    },
    RockClusterPlacement {
        position: [21.0, 13.0],
        rotation_degrees: 32.0,
        scale: 1.15,
        variant: 0,
        clearing_radius: 2.6,
    },
    RockClusterPlacement {
        position: [-8.0, -28.0],
        rotation_degrees: 103.0,
        scale: 0.72,
        variant: 1,
        clearing_radius: 1.8,
    },
    RockClusterPlacement {
        position: [31.0, -7.0],
        rotation_degrees: 212.0,
        scale: 1.45,
        variant: 2,
        clearing_radius: 3.2,
    },
    RockClusterPlacement {
        position: [-34.0, -20.0],
        rotation_degrees: 55.0,
        scale: 1.10,
        variant: 0,
        clearing_radius: 2.5,
    },
    RockClusterPlacement {
        position: [9.0, 27.0],
        rotation_degrees: 167.0,
        scale: 0.95,
        variant: 1,
        clearing_radius: 2.2,
    },
    RockClusterPlacement {
        position: [-20.0, 31.0],
        rotation_degrees: 246.0,
        scale: 1.55,
        variant: 2,
        clearing_radius: 3.4,
    },
    RockClusterPlacement {
        position: [37.0, 25.0],
        rotation_degrees: 11.0,
        scale: 1.20,
        variant: 0,
        clearing_radius: 2.8,
    },
    RockClusterPlacement {
        position: [-43.0, 4.0],
        rotation_degrees: 92.0,
        scale: 0.88,
        variant: 1,
        clearing_radius: 2.1,
    },
    RockClusterPlacement {
        position: [27.0, -38.0],
        rotation_degrees: 188.0,
        scale: 1.40,
        variant: 2,
        clearing_radius: 3.1,
    },
    RockClusterPlacement {
        position: [-49.0, 28.0],
        rotation_degrees: 301.0,
        scale: 1.25,
        variant: 0,
        clearing_radius: 2.8,
    },
    RockClusterPlacement {
        position: [52.0, 5.0],
        rotation_degrees: 128.0,
        scale: 1.05,
        variant: 1,
        clearing_radius: 2.4,
    },
    RockClusterPlacement {
        position: [-12.0, 48.0],
        rotation_degrees: 219.0,
        scale: 1.60,
        variant: 2,
        clearing_radius: 3.5,
    },
    RockClusterPlacement {
        position: [47.0, -30.0],
        rotation_degrees: 344.0,
        scale: 1.30,
        variant: 0,
        clearing_radius: 3.0,
    },
];

/// Settings for one square patch of heightfield terrain, expressed in world units.
#[derive(Clone, Copy)]
pub struct TerrainSettings {
    /// Width and depth of the square terrain patch.
    pub world_size: f32,
    /// Number of quads along each edge.  There is one more vertex than quad.
    pub cells_per_side: usize,
    /// Maximum height of the small-scale noise on the valley floor.
    pub max_height: f32,
    /// Radius of the relatively open valley floor before its enclosing hills begin.
    pub valley_floor_radius: f32,
    /// Radius at which the enclosing hills reach their maximum height.
    pub valley_crest_radius: f32,
    /// Height added by the hills at their crest.
    pub valley_wall_height: f32,
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
            // The outer half of this larger patch forms the distant valley walls.
            // One-metre cells are still sufficient for their broad, smooth slopes.
            world_size: 200.0,
            // One-metre cells are sufficient for the broad contours; fine colour
            // texture is evaluated by the fragment shader rather than extra geometry.
            cells_per_side: 200,
            max_height: 2.1,
            valley_floor_radius: 36.0,
            valley_crest_radius: 82.0,
            // The distant wall adds three metres above the local rolling terrain.
            valley_wall_height: 3.0,
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

    /// Fraction of candidate grass retained at this point. Two overlapping noise scales
    /// make small dry pockets without carving broad empty bands through the field.
    pub fn grass_density_at(&self, x: f32, z: f32) -> f32 {
        let broad = value_noise(x, z, 0.055, self.seed.wrapping_add(101));
        let detail = value_noise(x, z, 0.17, self.seed.wrapping_add(102));
        let moisture = broad * 0.45 + detail * 0.55;
        let biome_density = smoothstep(0.20, 0.47, moisture);
        biome_density * rock_grass_clearance(x, z)
    }

    /// Vertex colour beneath the grass. Dense regions expose dark, fertile soil;
    /// dry regions become warm sandy earth, while slopes and rock skirts turn greyer.
    fn ground_color_at(&self, x: f32, z: f32, slope: f32) -> [f32; 3] {
        let grass = self.grass_density_at(x, z);
        let dry_soil = [0.22, 0.145, 0.065];
        let fertile_soil = [0.085, 0.075, 0.025];
        let mut color = mix_color(dry_soil, fertile_soil, grass);

        let rock_skirt = 1.0 - rock_grass_clearance(x, z);
        let steep_ground = smoothstep(0.28, 0.72, slope);
        let rock_amount = rock_skirt.max(steep_ground * 0.75);
        color = mix_color(color, [0.25, 0.235, 0.19], rock_amount);

        let variation = 0.82 + 0.30 * value_noise(x, z, 0.31, self.seed.wrapping_add(109));
        [
            color[0] * variation,
            color[1] * variation,
            color[2] * variation,
        ]
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
        settings.valley_floor_radius >= 0.0,
        "valley floor radius cannot be negative"
    );
    assert!(
        settings.valley_crest_radius > settings.valley_floor_radius,
        "valley crest must lie outside the valley floor"
    );
    assert!(
        settings.valley_crest_radius < settings.world_size * 0.5,
        "valley crest must leave room for the outer slope"
    );
    assert!(
        settings.valley_wall_height >= 0.0,
        "valley wall height cannot be negative"
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
            let slope = (slope_x * slope_x + slope_z * slope_z).sqrt();

            mesh.vertices.push(Vertex {
                pos: [x, height, z],
                normal,
                color: settings.ground_color_at(x, z, slope),
                // Dedicated terrain shaders use these as stable world-space detail coordinates.
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

/// Returns the deterministic noise and valley-wall height, including the hut pad.
fn terrain_height(x: f32, z: f32, settings: TerrainSettings) -> f32 {
    let natural_height = natural_terrain_height(x, z, settings);
    // Flatten to the height the unmodified terrain would have at the hut centre.  This
    // creates a building pad without digging the old, conspicuous zero-height bowl.
    let pad_height =
        natural_terrain_height(settings.pad_centre[0], settings.pad_centre[1], settings);

    let dx = x - settings.pad_centre[0];
    let dz = z - settings.pad_centre[1];
    let distance_from_pad = (dx * dx + dz * dz).sqrt();
    let blend_to_terrain = smoothstep(
        settings.pad_radius,
        settings.pad_radius + settings.pad_falloff,
        distance_from_pad,
    );

    lerp(pad_height, natural_height, blend_to_terrain)
}

/// Combines local noise and the distant valley wall before any building-pad flattening.
fn natural_terrain_height(x: f32, z: f32, settings: TerrainSettings) -> f32 {
    // Broad rolling forms dominate. A ridged version of the medium octave adds occasional
    // shoulders and shallow gullies without turning the valley floor into sharp noise.
    let broad = value_noise(x, z, 0.018, settings.seed);
    let medium = value_noise(x, z, 0.047, settings.seed.wrapping_add(1));
    let detail = value_noise(x, z, 0.13, settings.seed.wrapping_add(2));
    let ridge = 1.0 - (medium * 2.0 - 1.0).abs();
    let landform = 0.58 * broad + 0.24 * medium + 0.12 * ridge * ridge + 0.06 * detail;
    let noisy_height = settings.max_height * smoothstep(0.08, 0.92, landform);

    // A fourth-power radius produces a rounded square: it follows the square mesh's
    // boundary more closely than a circle, without introducing sharp diagonal corners.
    let boundary_radius = (x.powi(4) + z.powi(4)).sqrt().sqrt();
    let half_size = settings.world_size * 0.5;
    let climb = smoothstep(
        settings.valley_floor_radius,
        settings.valley_crest_radius,
        boundary_radius,
    );
    let outer_slope = 1.0 - smoothstep(settings.valley_crest_radius, half_size, boundary_radius);
    // Multiplying two smooth curves gives a rounded crest: the hill rises away from
    // the playable centre, then falls back toward the base terrain before the mesh ends.
    let valley_wall = settings.valley_wall_height * climb * outer_slope;

    noisy_height + valley_wall
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
        (x as u32).wrapping_mul(0x8DA6_B343) ^ (z as u32).wrapping_mul(0xD816_3841) ^ seed;
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

fn mix_color(a: [f32; 3], b: [f32; 3], amount: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], amount),
        lerp(a[1], b[1], amount),
        lerp(a[2], b[2], amount),
    ]
}

/// One means grass is unaffected; zero is the bare centre of a rock cluster.
fn rock_grass_clearance(x: f32, z: f32) -> f32 {
    ROCK_CLUSTERS.iter().fold(1.0_f32, |clearance, cluster| {
        let dx = x - cluster.position[0];
        let dz = z - cluster.position[1];
        let distance = (dx * dx + dz * dz).sqrt();
        let local_clearance = smoothstep(
            cluster.clearing_radius * 0.72,
            cluster.clearing_radius + 0.8,
            distance,
        );
        clearance.min(local_clearance)
    })
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
    use super::{TerrainSettings, build_heightfield};

    #[test]
    fn terrain_stays_within_its_height_range_and_keeps_the_hut_pad_level() {
        let settings = TerrainSettings::default();
        let mesh = build_heightfield(settings);

        assert_eq!(mesh.vertices.len(), 201 * 201);
        assert_eq!(mesh.indices.len(), 200 * 200 * 6);
        assert!(mesh.vertices.iter().all(|vertex| {
            (0.0..=settings.max_height + settings.valley_wall_height).contains(&vertex.pos[1])
        }));

        let pad_centre = mesh
            .vertices
            .iter()
            .find(|vertex| vertex.pos[0] == 0.0 && vertex.pos[2] == -2.0)
            .expect("the one-metre grid contains the pad centre");
        let pad_height = settings.height_at(0.0, -2.0);
        assert_eq!(pad_centre.pos[1], pad_height);
        assert!(pad_height > 0.0);
        // Every point inside the pad radius lands on the same natural centre height.
        assert_eq!(settings.height_at(1.0, -2.0), pad_height);
    }

    #[test]
    fn biome_has_grassy_and_bare_regions_and_clears_rock_centres() {
        let settings = TerrainSettings::default();
        let mut minimum = 1.0_f32;
        let mut maximum = 0.0_f32;
        for z in (-60..=60).step_by(4) {
            for x in (-60..=60).step_by(4) {
                let density = settings.grass_density_at(x as f32, z as f32);
                minimum = minimum.min(density);
                maximum = maximum.max(density);
            }
        }
        assert!(minimum < 0.05, "the biome should contain bare clearings");
        assert!(maximum > 0.90, "the biome should contain dense grass");
        for cluster in super::ROCK_CLUSTERS {
            assert_eq!(
                settings.grass_density_at(cluster.position[0], cluster.position[1]),
                0.0
            );
        }
    }

    #[test]
    fn valley_wall_rises_to_a_crest_then_rounds_down_at_the_boundary() {
        let settings = TerrainSettings {
            // Removing noise and moving the pad away isolates the valley profile.
            max_height: 0.0,
            pad_centre: [1_000.0, 1_000.0],
            ..TerrainSettings::default()
        };

        assert_eq!(settings.height_at(settings.valley_floor_radius, 0.0), 0.0);
        assert!(settings.height_at(60.0, 0.0) > 0.0);
        assert_eq!(
            settings.height_at(settings.valley_crest_radius, 0.0),
            settings.valley_wall_height
        );
        assert!(settings.height_at(90.0, 0.0) < settings.valley_wall_height);
        assert_eq!(settings.height_at(settings.world_size * 0.5, 0.0), 0.0);
    }
}
