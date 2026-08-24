//! CPU-side procedural vegetation generation and chunk visibility tests.
//!
//! Everything here produces ordinary owned Vec values or small copyable metadata.
//! Vulkan ownership begins later, when the parent renderer uploads those vectors.

use super::*;
use cgmath::{Matrix4, Vector4};
pub(super) fn pack_instance_params(rotation: f32, width: f32, tint: f32, phase: f32) -> [u16; 4] {
    let to_unorm = |value: f32| (value.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16;
    [
        to_unorm(rotation / std::f32::consts::TAU),
        to_unorm(width / MAX_BLADE_WIDTH),
        to_unorm(tint),
        to_unorm(phase / std::f32::consts::TAU),
    ]
}

pub(super) fn pack_position_height(x: f32, y: f32, z: f32, height: f32) -> [u16; 4] {
    let pack_range = |value: f32, range: [f32; 2]| {
        let normalised = (value - range[0]) / (range[1] - range[0]);
        (normalised.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16
    };
    [
        pack_range(x, INSTANCE_X_RANGE),
        pack_range(y, INSTANCE_Y_RANGE),
        pack_range(z, INSTANCE_Z_RANGE),
        pack_range(height, INSTANCE_HEIGHT_RANGE),
    ]
}

#[cfg(test)]
pub(super) fn unpack_position_height(packed: [u16; 4]) -> [f32; 4] {
    let unpack_range = |value: u16, range: [f32; 2]| {
        range[0] + value as f32 / u16::MAX as f32 * (range[1] - range[0])
    };
    [
        unpack_range(packed[0], INSTANCE_X_RANGE),
        unpack_range(packed[1], INSTANCE_Y_RANGE),
        unpack_range(packed[2], INSTANCE_Z_RANGE),
        unpack_range(packed[3], INSTANCE_HEIGHT_RANGE),
    ]
}

/// Builds one ribbon with the requested segment count. Instance rotation makes many
/// individually flat ribbons read as volumetric grass.
pub(super) fn build_blade_vertices(segments: usize) -> Vec<GrassVertex> {
    let mut vertices = Vec::with_capacity((segments + 1) * 2);

    for segment in 0..=segments {
        let height_fraction = segment as f32 / segments as f32;
        // Narrow toward the tip so the ribbon reads as a blade rather than a rectangle.
        let half_width = 0.5 * (1.0 - 0.82 * height_fraction);
        for side in [-1.0, 1.0] {
            vertices.push(GrassVertex {
                local_position: [half_width * side, height_fraction, 0.0],
                local_normal: [0.0, 0.0, 1.0],
                flower_head: 0.0,
            });
        }
    }
    vertices
}

pub(super) fn build_blade_indices_for_segments(segments: usize) -> Vec<u32> {
    let mut indices = Vec::with_capacity(segments * 6);

    for segment in 0..segments {
        let lower_left = (segment * 2) as u32;
        let lower_right = lower_left + 1;
        let upper_left = lower_left + 2;
        let upper_right = lower_left + 3;
        indices.extend_from_slice(&[
            lower_left,
            lower_right,
            upper_right,
            upper_right,
            upper_left,
            lower_left,
        ]);
    }
    indices
}

/// Builds a six-sided stalk with a rounded capsule at the top.  The mesh is expressed in
/// unit height and width; each instance supplies its final height and radius.
pub(super) fn build_reed_vertices() -> Vec<GrassVertex> {
    const SIDES: usize = 6;
    const HEAD_RINGS: &[(f32, f32, f32)] = &[
        // (height, radius, vertical normal component)
        (0.68, 0.18, -0.66),
        (0.73, 0.55, -0.31),
        (0.88, 0.55, 0.31),
        (0.94, 0.18, 0.66),
    ];
    let mut vertices = Vec::with_capacity(SIDES * (2 + HEAD_RINGS.len()) + 2);

    // The narrow green stalk uses two rings.  Its top overlaps the flower head slightly.
    for height in [0.0, 0.99] {
        for side in 0..SIDES {
            let angle = side as f32 / SIDES as f32 * std::f32::consts::TAU;
            vertices.push(GrassVertex {
                local_position: [angle.cos() * 0.16, height, angle.sin() * 0.16],
                local_normal: [angle.cos(), 0.0, angle.sin()],
                flower_head: 0.0,
            });
        }
    }

    // These rings approximate an elongated capsule, which keeps the flower deliberately
    // simple and stylised while still lighting like a small solid object.
    for &(height, radius, normal_y) in HEAD_RINGS {
        let horizontal_normal = (1.0 - normal_y * normal_y).sqrt();
        for side in 0..SIDES {
            let angle = side as f32 / SIDES as f32 * std::f32::consts::TAU;
            vertices.push(GrassVertex {
                local_position: [angle.cos() * radius, height, angle.sin() * radius],
                local_normal: [
                    angle.cos() * horizontal_normal,
                    normal_y,
                    angle.sin() * horizontal_normal,
                ],
                flower_head: 1.0,
            });
        }
    }

    vertices.push(GrassVertex {
        local_position: [0.0, 0.66, 0.0],
        local_normal: [0.0, -1.0, 0.0],
        flower_head: 1.0,
    });
    vertices.push(GrassVertex {
        // The cap sits just below the stalk tip, leaving a small green point visible.
        local_position: [0.0, 0.96, 0.0],
        local_normal: [0.0, 1.0, 0.0],
        flower_head: 1.0,
    });
    vertices
}

pub(super) fn build_reed_indices() -> Vec<u32> {
    const SIDES: usize = 6;
    const RING_COUNT: usize = 6;
    let mut indices = Vec::with_capacity(4 * SIDES * 6 + SIDES * 6);

    // The capsule overlaps the stalk, so there is deliberately no surface joining their
    // rings.  That avoids a visible seam and keeps their distinct material flags separate.
    for (lower_ring, upper_ring) in [(0, 1), (2, 3), (3, 4), (4, 5)] {
        for side in 0..SIDES {
            let next_side = (side + 1) % SIDES;
            let lower_left = (lower_ring * SIDES + side) as u32;
            let lower_right = (lower_ring * SIDES + next_side) as u32;
            let upper_left = (upper_ring * SIDES + side) as u32;
            let upper_right = (upper_ring * SIDES + next_side) as u32;
            indices.extend_from_slice(&[
                lower_left,
                upper_right,
                lower_right,
                upper_right,
                lower_left,
                upper_left,
            ]);
        }
    }

    let lower_cap = (RING_COUNT * SIDES) as u32;
    let upper_cap = lower_cap + 1;
    let flower_first_ring = 2 * SIDES;
    let flower_last_ring = (RING_COUNT - 1) * SIDES;
    for side in 0..SIDES {
        let next_side = (side + 1) % SIDES;
        indices.extend_from_slice(&[
            lower_cap,
            (flower_first_ring + side) as u32,
            (flower_first_ring + next_side) as u32,
            upper_cap,
            (flower_last_ring + next_side) as u32,
            (flower_last_ring + side) as u32,
        ]);
    }
    indices
}

/// Builds regular-but-jittered grass in contiguous 5×5 m chunks.  Keeping each chunk's
/// instances together lets `first_instance` select it without copying GPU data per frame.
pub(super) fn build_chunked_grass_instances(terrain: TerrainSettings) -> ChunkedGrassInstances {
    let spacing = GRASS_FIELD_SIZE / GRASS_GRID_SIDE as f32;
    let half_size = GRASS_FIELD_SIZE * 0.5;
    // The central field keeps every candidate while the much larger outer area is
    // thinned, so reserving the complete candidate count would waste considerable RAM.
    let candidate_count = GRASS_GRID_SIDE * GRASS_GRID_SIDE * GRASS_DENSITY_LAYERS;
    let mut instances = Vec::with_capacity(candidate_count * 3 / 4);
    let mut chunks = Vec::with_capacity(GRASS_CHUNKS_PER_SIDE * GRASS_CHUNKS_PER_SIDE);
    let cells_per_chunk = GRASS_GRID_SIDE / GRASS_CHUNKS_PER_SIDE;
    let chunk_radius = GRASS_CHUNK_SIZE * std::f32::consts::FRAC_1_SQRT_2 + 0.8;

    for chunk_row in 0..GRASS_CHUNKS_PER_SIDE {
        for chunk_column in 0..GRASS_CHUNKS_PER_SIDE {
            let first_instance = instances.len() as u32;
            let first_row = chunk_row * cells_per_chunk;
            let first_column = chunk_column * cells_per_chunk;
            let mut mid_subset = Vec::with_capacity(cells_per_chunk * cells_per_chunk / 2);
            let mut remaining = Vec::with_capacity(cells_per_chunk * cells_per_chunk * 2);

            for row in first_row..first_row + cells_per_chunk {
                for column in first_column..first_column + cells_per_chunk {
                    for layer in 0..GRASS_DENSITY_LAYERS {
                        let id = ((row * GRASS_GRID_SIDE + column) * GRASS_DENSITY_LAYERS + layer)
                            as u32;
                        // Two diagonally opposed offsets turn each former placement cell into a
                        // small staggered pair instead of placing two blades at the same root.
                        let (cell_offset_x, cell_offset_z) = if layer == 0 {
                            (0.25, 0.75)
                        } else {
                            (0.75, 0.25)
                        };
                        let jitter_x = (hash01(id, 0) - 0.5) * spacing * 0.45;
                        let jitter_z = (hash01(id, 1) - 0.5) * spacing * 0.45;
                        let x = -half_size + (column as f32 + cell_offset_x) * spacing + jitter_x;
                        let z =
                            -2.0 - half_size + (row as f32 + cell_offset_z) * spacing + jitter_z;

                        // Distant blades occupy fewer candidate roots.  The decision is
                        // stable for each blade, so no grass pops in or moves between frames.
                        if hash01(id, 31) > grass_density_at(x, z) {
                            continue;
                        }

                        if is_clearing(x, z) {
                            continue;
                        }

                        let instance = GrassInstance {
                            position_height: pack_position_height(
                                x,
                                terrain.height_at(x, z) + 0.003,
                                z,
                                0.17 + hash01(id, 2) * 0.21,
                            ),
                            rotation_width_tint_phase: pack_instance_params(
                                hash01(id, 3) * std::f32::consts::TAU,
                                0.025 + hash01(id, 4) * 0.03,
                                hash01(id, 5),
                                hash01(id, 6) * std::f32::consts::TAU,
                            ),
                        };
                        // The mid LOD uses a stable random half, avoiding visible rows. Half
                        // density preserves the field's colour while its two-triangle ribbon
                        // remains much cheaper than the near blade's eight triangles.
                        if hash01(id, 30) < 0.50 {
                            mid_subset.push(instance);
                        } else {
                            remaining.push(instance);
                        }
                    }
                }
            }

            // Put the thinned subset first. Its range is also the beginning of the full
            // range, allowing all grass LODs to share one GPU instance buffer.
            let mid_count = mid_subset.len() as u32;
            let near_count = (mid_subset.len() + remaining.len()) as u32;
            instances.extend(mid_subset);
            instances.extend(remaining);

            chunks.push(VegetationChunk {
                centre: [
                    -half_size + (chunk_column as f32 + 0.5) * GRASS_CHUNK_SIZE,
                    -2.0 - half_size + (chunk_row as f32 + 0.5) * GRASS_CHUNK_SIZE,
                ],
                radius: chunk_radius,
                near_grass: InstanceRange {
                    first: first_instance,
                    count: near_count,
                },
                mid_grass: InstanceRange {
                    first: first_instance,
                    count: mid_count,
                },
                reeds: InstanceRange::default(),
            });
        }
    }

    ChunkedGrassInstances { instances, chunks }
}

pub(super) fn grass_lod(distance: f32) -> GrassLod {
    if distance <= HIGH_DETAIL_GRASS_DISTANCE {
        GrassLod::Near
    } else if distance <= NEAR_GRASS_DISTANCE {
        GrassLod::Medium
    } else {
        GrassLod::Mid
    }
}

/// Returns the fraction of candidate grass roots retained at a world position.
pub(super) fn grass_density_at(x: f32, z: f32) -> f32 {
    // The field is centred two metres behind the origin to match its existing placement.
    // A square radius makes density reach the same value along all four field boundaries.
    let field_radius = x.abs().max((z + 2.0).abs());
    let half_size = GRASS_FIELD_SIZE * 0.5;
    let amount =
        ((field_radius - FULL_DENSITY_RADIUS) / (half_size - FULL_DENSITY_RADIUS)).clamp(0.0, 1.0);
    let smooth_amount = amount * amount * (3.0 - 2.0 * amount);
    1.0 - (1.0 - OUTER_DENSITY) * smooth_amount
}

/// Places a few clusters of simple flower reeds.  Every stalk in a cluster shares its
/// flower colour, making it read as one small plant rather than nine random tall blades.
pub(super) fn build_reed_instances(
    terrain: TerrainSettings,
    chunks: &mut [VegetationChunk],
) -> Vec<GrassInstance> {
    let half_size = GRASS_FIELD_SIZE * 0.5;
    let mut per_chunk = vec![Vec::new(); chunks.len()];

    for tuft_index in 0..TALL_TUFT_COUNT {
        let tuft_id = tuft_index as u32;
        let centre_x = -half_size + hash01(tuft_id, 20) * GRASS_FIELD_SIZE;
        let centre_z = -2.0 - half_size + hash01(tuft_id, 21) * GRASS_FIELD_SIZE;

        if is_clearing(centre_x, centre_z) {
            continue;
        }

        for blade_index in 0..TALL_BLADES_PER_TUFT {
            let blade_id = tuft_id * TALL_BLADES_PER_TUFT as u32 + blade_index as u32;
            // A square-root radius keeps the cluster evenly filled instead of concentrating
            // blades at its centre.  Rotation then makes each tuft read as a small plant.
            let angle = hash01(blade_id, 22) * std::f32::consts::TAU;
            let radius = hash01(blade_id, 23).sqrt() * 0.28;
            let x = centre_x + angle.cos() * radius;
            let z = centre_z + angle.sin() * radius;

            per_chunk[chunk_index_for(centre_x, centre_z)].push(GrassInstance {
                position_height: pack_position_height(
                    x,
                    terrain.height_at(x, z) + 0.004,
                    z,
                    0.62 + hash01(blade_id, 24) * 0.18,
                ),
                rotation_width_tint_phase: pack_instance_params(
                    hash01(blade_id, 25) * std::f32::consts::TAU,
                    0.070 + hash01(blade_id, 26) * 0.018,
                    // Most clusters are cream-white; a few are a muted red variant.
                    if hash01(tuft_id, 27) < 0.22 { 0.0 } else { 1.0 },
                    hash01(blade_id, 28) * std::f32::consts::TAU,
                ),
            });
        }
    }

    let mut instances = Vec::with_capacity(TALL_TUFT_COUNT * TALL_BLADES_PER_TUFT);
    for (chunk, reed_instances) in chunks.iter_mut().zip(per_chunk) {
        chunk.reeds = InstanceRange {
            first: instances.len() as u32,
            count: reed_instances.len() as u32,
        };
        instances.extend(reed_instances);
    }
    instances
}

/// Returns the chunk containing a field-space root position.  Reed clusters use their
/// centre so all nine stalks remain in the same culling/LOD decision.
pub(super) fn chunk_index_for(x: f32, z: f32) -> usize {
    let half_size = GRASS_FIELD_SIZE * 0.5;
    let column = ((x + half_size) / GRASS_CHUNK_SIZE)
        .floor()
        .clamp(0.0, (GRASS_CHUNKS_PER_SIDE - 1) as f32) as usize;
    let row = ((z + 2.0 + half_size) / GRASS_CHUNK_SIZE)
        .floor()
        .clamp(0.0, (GRASS_CHUNKS_PER_SIDE - 1) as f32) as usize;
    row * GRASS_CHUNKS_PER_SIDE + column
}

/// Conservative 3D frustum test using the eight corners of a padded chunk box.
///
/// Testing homogeneous clip coordinates avoids error-prone plane extraction. A chunk is
/// rejected only when all eight corners lie outside the same clipping plane.
pub(super) fn chunk_intersects_frustum(
    view_projection: Matrix4<f32>,
    chunk: &VegetationChunk,
) -> bool {
    let half_width = GRASS_CHUNK_SIZE * 0.5 + 0.3;
    let mut corners = [Vector4::new(0.0, 0.0, 0.0, 1.0); 8];
    let mut index = 0;
    for y in [-0.05, 1.05] {
        for x in [chunk.centre[0] - half_width, chunk.centre[0] + half_width] {
            for z in [chunk.centre[1] - half_width, chunk.centre[1] + half_width] {
                corners[index] = view_projection * Vector4::new(x, y, z, 1.0);
                index += 1;
            }
        }
    }

    !corners.iter().all(|p| p.x < -p.w)
        && !corners.iter().all(|p| p.x > p.w)
        && !corners.iter().all(|p| p.y < -p.w)
        && !corners.iter().all(|p| p.y > p.w)
        && !corners.iter().all(|p| p.z < 0.0)
        && !corners.iter().all(|p| p.z > p.w)
}

pub(super) fn draw_command(
    index_count: u32,
    range: InstanceRange,
) -> vk::DrawIndexedIndirectCommand {
    vk::DrawIndexedIndirectCommand {
        index_count,
        instance_count: range.count,
        first_index: 0,
        vertex_offset: 0,
        first_instance: range.first,
    }
}

pub(super) fn is_clearing(x: f32, z: f32) -> bool {
    // Hut, cube, smaller cube, and sphere.  These are temporary scene-level placement
    // hints; a future placement system can provide these radii instead.
    const CLEARINGS: &[(f32, f32, f32)] = &[
        (0.0, -2.0, 3.0),
        (7.0, 0.0, 1.5),
        (6.0, -6.0, 1.3),
        (-6.0, 0.0, 1.4),
    ];
    CLEARINGS.iter().any(|&(centre_x, centre_z, radius)| {
        let dx = x - centre_x;
        let dz = z - centre_z;
        dx * dx + dz * dz < radius * radius
    })
}

/// Deterministic pseudo-random value in `0.0..=1.0`, without storing a CPU RNG.
pub(super) fn hash01(id: u32, stream: u32) -> f32 {
    let mut value = id.wrapping_mul(0x9E37_79B9) ^ stream.wrapping_mul(0x85EB_CA6B);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^= value >> 16;
    value as f32 / u32::MAX as f32
}
