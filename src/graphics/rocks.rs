//! Reusable rock assets and deterministic cluster composition.
//!
//! Each source model is imported once. Cluster members clone only scene handles and apply
//! different transforms, so repeated rocks share their mesh, material, and texture atlas.

use crate::graphics::terrain::RockClusterPlacement;

pub const ROCK_ASSET_PATHS: [&str; 5] = [
    "assets/meshes/rocks/rock-medium-a.gltf",
    "assets/meshes/rocks/rock-medium-b.gltf",
    "assets/meshes/rocks/rock-large-a.gltf",
    "assets/meshes/rocks/rock-large-b.gltf",
    "assets/meshes/rocks/rock-large-c.gltf",
];

/// Assets in each group use one atlas, and therefore one Vulkan material/pipeline.
pub const ROCK_MATERIAL_NAMES: [&str; 5] = [
    "QuaterniusRockMedium",
    "QuaterniusRockMedium",
    "QuaterniusRockLarge",
    "QuaterniusRockLarge",
    "QuaterniusRockLarge",
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RockMember {
    pub variant: usize,
    pub offset: [f32; 2],
    pub rotation_degrees: f32,
    /// Desired longest dimension in metres; scene setup normalises each source asset first.
    pub size: f32,
}

/// Builds three stones around one art-directed anchor. The anchor controls the overall
/// footprint while the index changes the asset combination without runtime randomness.
pub fn cluster_members(cluster_index: usize, placement: RockClusterPlacement) -> [RockMember; 3] {
    let seed = cluster_index as f32 * 1.618_034;
    let main_variant = (placement.variant + cluster_index) % ROCK_ASSET_PATHS.len();
    let secondary_angle = placement.rotation_degrees + 55.0 + seed.sin() * 24.0;
    let tertiary_angle = placement.rotation_degrees - 68.0 + seed.cos() * 21.0;

    [
        RockMember {
            variant: main_variant,
            offset: [0.0, 0.0],
            rotation_degrees: placement.rotation_degrees,
            size: placement.scale * 1.95,
        },
        RockMember {
            variant: (main_variant + 2) % ROCK_ASSET_PATHS.len(),
            offset: polar_offset(placement.scale * 1.02, secondary_angle),
            rotation_degrees: secondary_angle + 37.0,
            size: placement.scale * 1.12,
        },
        RockMember {
            variant: (main_variant + 4) % ROCK_ASSET_PATHS.len(),
            offset: polar_offset(placement.scale * 0.88, tertiary_angle),
            rotation_degrees: tertiary_angle - 19.0,
            size: placement.scale * 0.80,
        },
    ]
}

fn polar_offset(distance: f32, angle_degrees: f32) -> [f32; 2] {
    let angle = angle_degrees.to_radians();
    [angle.cos() * distance, angle.sin() * distance]
}

#[cfg(test)]
mod tests {
    use super::{ROCK_ASSET_PATHS, cluster_members};
    use crate::graphics::terrain::ROCK_CLUSTERS;

    #[test]
    fn cluster_combinations_use_every_shared_model() {
        let mut used = [false; ROCK_ASSET_PATHS.len()];
        for (index, placement) in ROCK_CLUSTERS.iter().copied().enumerate() {
            for member in cluster_members(index, placement) {
                used[member.variant] = true;
                assert!(member.size > 0.0);
            }
        }
        assert!(used.into_iter().all(|is_used| is_used));
    }
}
