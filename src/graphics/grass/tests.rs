use super::generation::{
    build_blade_indices_for_segments, build_blade_vertices, build_chunked_grass_instances,
    build_reed_indices, build_reed_instances, build_reed_vertices, chunk_intersects_frustum,
    field_edge_density_at, grass_lod, pack_position_height, unpack_position_height,
};
use super::{
    GRASS_CHUNKS_PER_SIDE, GRASS_DENSITY_LAYERS, GRASS_GRID_SIDE, GrassInstance, GrassLod,
    TALL_BLADES_PER_TUFT, WIND_MAP_SIZE, build_wind_map_pixels,
};
use crate::graphics::terrain::TerrainSettings;

#[test]
fn grass_uses_one_shared_blade_and_many_grounded_instances() {
    let vertices = build_blade_vertices(4);
    let indices = build_blade_indices_for_segments(4);
    let chunked = build_chunked_grass_instances(TerrainSettings::default());

    assert_eq!(vertices.len(), 10);
    assert_eq!(indices.len(), 24);
    assert_eq!(build_blade_vertices(2).len(), 6);
    assert_eq!(build_blade_indices_for_segments(2).len(), 12);
    assert_eq!(std::mem::size_of::<GrassInstance>(), 16);
    let candidate_count = GRASS_GRID_SIDE * GRASS_GRID_SIDE * GRASS_DENSITY_LAYERS;
    assert!(chunked.instances.len() > candidate_count / 5);
    assert!(chunked.instances.len() < candidate_count * 4 / 5);
    assert_eq!(
        chunked.chunks.len(),
        GRASS_CHUNKS_PER_SIDE * GRASS_CHUNKS_PER_SIDE
    );
    assert!(
        chunked
            .instances
            .iter()
            .all(|instance| unpack_position_height(instance.position_height)[3] > 0.0)
    );
    assert!(
        chunked
            .instances
            .iter()
            .any(|instance| unpack_position_height(instance.position_height)[3] < 0.25)
    );
    assert!(
        chunked
            .instances
            .iter()
            .any(|instance| unpack_position_height(instance.position_height)[3] > 0.35)
    );
    assert!(
        chunked
            .instances
            .iter()
            .any(|instance| unpack_position_height(instance.position_height)[0].abs() > 70.0)
    );
    assert_eq!(field_edge_density_at(0.0, -2.0), 1.0);
    assert_eq!(field_edge_density_at(80.0, -2.0), super::OUTER_DENSITY);
    let mid_count: usize = chunked
        .chunks
        .iter()
        .map(|chunk| chunk.mid_grass.count as usize)
        .sum();
    assert!(mid_count > chunked.instances.len() * 2 / 5);
    assert!(mid_count < chunked.instances.len() * 3 / 5);
}

#[test]
fn reeds_use_a_separate_capsule_mesh_and_stay_clustered() {
    let vertices = build_reed_vertices();
    let indices = build_reed_indices();
    let mut chunked = build_chunked_grass_instances(TerrainSettings::default());
    let instances = build_reed_instances(TerrainSettings::default(), &mut chunked.chunks);

    assert!(vertices.iter().any(|vertex| vertex.flower_head > 0.5));
    assert!(indices.len() > 100);
    assert_eq!(indices.len() % 3, 0, "every reed index group is a triangle");
    // Every side triangle must face the same way as its stored outward normals.  This
    // catches an easy-to-miss error where only one triangle of every quad is reversed.
    for (triangle_index, triangle) in indices[..4 * 6 * 6].as_chunks::<3>().0.iter().enumerate() {
        let a = vertices[triangle[0] as usize];
        let b = vertices[triangle[1] as usize];
        let c = vertices[triangle[2] as usize];
        let ab = [
            b.local_position[0] - a.local_position[0],
            b.local_position[1] - a.local_position[1],
            b.local_position[2] - a.local_position[2],
        ];
        let ac = [
            c.local_position[0] - a.local_position[0],
            c.local_position[1] - a.local_position[1],
            c.local_position[2] - a.local_position[2],
        ];
        let face_normal = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let average_normal = [
            a.local_normal[0] + b.local_normal[0] + c.local_normal[0],
            a.local_normal[1] + b.local_normal[1] + c.local_normal[1],
            a.local_normal[2] + b.local_normal[2] + c.local_normal[2],
        ];
        let winding_matches_normals = face_normal[0] * average_normal[0]
            + face_normal[1] * average_normal[1]
            + face_normal[2] * average_normal[2];
        assert!(
            winding_matches_normals > 0.0,
            "side triangle {triangle_index} is inward"
        );
    }
    assert!(instances.len() >= TALL_BLADES_PER_TUFT);
    assert!(
        instances
            .iter()
            .all(|instance| unpack_position_height(instance.position_height)[3] >= 0.619)
    );
    assert_eq!(
        chunked
            .chunks
            .iter()
            .map(|chunk| chunk.reeds.count as usize)
            .sum::<usize>(),
        instances.len(),
    );
}

#[test]
fn packed_instances_preserve_scene_scale_and_lod_boundaries() {
    let original = [37.25, 3.75, -61.5, 0.73];
    let unpacked = unpack_position_height(pack_position_height(
        original[0],
        original[1],
        original[2],
        original[3],
    ));
    for (actual, expected) in unpacked.into_iter().zip(original) {
        assert!((actual - expected).abs() < 0.003);
    }

    assert_eq!(grass_lod(12.0), GrassLod::Near);
    assert_eq!(grass_lod(12.01), GrassLod::Medium);
    assert_eq!(grass_lod(24.0), GrassLod::Medium);
    assert_eq!(grass_lod(24.01), GrassLod::Mid);
}

#[test]
fn grass_chunks_cull_side_and_distant_field_sections() {
    let chunked = build_chunked_grass_instances(TerrainSettings::default());
    let camera = crate::graphics::camera::Camera::new();
    let view_projection = *camera.get_projection() * *camera.get_view();
    let visible_chunks = chunked
        .chunks
        .iter()
        .filter(|chunk| chunk_intersects_frustum(view_projection, chunk))
        .count();

    assert!(visible_chunks > 0);
    assert!(visible_chunks < chunked.chunks.len());
}

#[test]
fn generated_wind_map_has_two_varying_channels() {
    let pixels = build_wind_map_pixels();
    assert_eq!(pixels.len(), (WIND_MAP_SIZE * WIND_MAP_SIZE * 4) as usize);
    let first_red = pixels[0];
    let first_green = pixels[1];
    assert!(
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .any(|pixel| pixel[0] != first_red)
    );
    assert!(
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .any(|pixel| pixel[1] != first_green)
    );
}
