// --------------------------------------------------------------------------------------
// grass.vert
//
// Expands one shared grass-ribbon mesh into many world-space individual blades.  Binding 0
// is per-vertex blade geometry; binding 1 advances once per instance.
// --------------------------------------------------------------------------------------
#version 450

#define MAX_LIGHTS 8

layout(push_constant) uniform Push {
    mat4 mvp;
    mat4 mv;
    // x = elapsed seconds, y = wind strength, z = LOD (0 near, 0.5 medium, 1 mid).
    vec4 grassParams;
} pc;

struct Light {
    vec3 position;
    float intensity;
    vec3 color;
    float _pad;
};

struct DirLight {
    vec3 direction;
    float intensity;
    vec3 color;
    float _pad;
};

layout(std140, set = 0, binding = 0) uniform GlobalUBO {
    DirLight sun;
    mat4 light_vp;
    Light lights[MAX_LIGHTS];
    uint light_count;
    uvec3 _pad0;
} ubo;

// A small, tileable two-channel field. Red contains broad gust shapes and green
// contains finer turbulence; both are generated once when GrassRenderer is created.
layout(set = 1, binding = 0) uniform sampler2D windMap;

layout(location = 0) in vec3 inLocalPosition;
layout(location = 1) in vec3 inLocalNormal;
layout(location = 2) in float inFlowerHead;
layout(location = 3) in vec4 inPositionHeight;
layout(location = 4) in vec4 inRotationWidthTintPhase;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec4 vShadowPosition;
layout(location = 2) out float vHeightFraction;
layout(location = 3) out float vTint;
layout(location = 4) out float vFlowerHead;

mat2 rotate2d(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, -s, s, c);
}

// NVIDIA's vegetation examples use smoothed triangle waves for inexpensive continuous
// detail motion. Unlike a thresholded sine pulse, this never pauses at one displacement.
float smooth_triangle(float phase) {
    float triangle = abs(fract(phase) * 2.0 - 1.0);
    float smoothed = triangle * triangle * (3.0 - 2.0 * triangle);
    return smoothed * 2.0 - 1.0;
}

void main() {
    float height_fraction = inLocalPosition.y;
    // Both instance attributes arrive as normalised 16-bit values. Decode the root
    // position from the same fixed field bounds used by Rust when packing it.
    vec4 position_height = vec4(
        mix(-80.5, 80.5, inPositionHeight.x),
        mix(0.0, 5.0, inPositionHeight.y),
        mix(-82.5, 78.5, inPositionHeight.z),
        mix(0.0, 1.0, inPositionHeight.w)
    );
    float rotation = inRotationWidthTintPhase.x * 6.28318530718;
    float blade_width = inRotationWidthTintPhase.y * 0.1;
    // Fewer mid-distance instances expose more dark soil. A modest width increase
    // preserves the field's average coverage without restoring near-LOD geometry cost.
    blade_width *= mix(1.0, 1.50, pc.grassParams.z);
    float wind_phase = inRotationWidthTintPhase.w;

    mat2 blade_rotation = rotate2d(rotation);
    vec2 local_xz = blade_rotation * inLocalPosition.xz * blade_width;
    vec2 normal_xz = blade_rotation * inLocalNormal.xz;

    // Ghost of Tsushima combines a broad scrolling noise field with finer moving detail.
    // These two texture reads replace the old sixteen sine-based hashes per vertex.
    const vec2 global_wind = vec2(0.8, 0.6);
    vec2 broad_uv = position_height.xz * 0.035 - global_wind * pc.grassParams.x * 0.11;
    vec2 detail_uv = position_height.xz * 0.14 - global_wind * pc.grassParams.x * 0.36;
    float broad = textureLod(windMap, broad_uv, 0.0).r;
    float detail = textureLod(windMap, detail_uv, 0.0).g * 2.0 - 1.0;

    // Small spatial direction changes stop the whole field leaning as one rigid sheet.
    float direction_offset = detail * 0.24;
    vec2 local_wind = rotate2d(direction_offset) * global_wind;

    // Broad bright patches become short, clearly visible gusts. Continuous triangle-wave
    // motion remains underneath, so grass bends, passes through rest, and always returns.
    float gust_envelope = smoothstep(0.58, 0.82, broad);
    float sway_phase = dot(position_height.xz, vec2(0.19, 0.27))
        + pc.grassParams.x * 0.62 + wind_phase;
    float calm_sway = 0.24 * (
        smooth_triangle(sway_phase)
        + 0.32 * smooth_triangle(sway_phase * 1.71 + 0.37)
    );
    float gust_lean = gust_envelope * (1.05 + detail * 0.16);
    // Mid-distance blades retain the large readable motion but skip some fine sway.
    calm_sway *= mix(1.0, 0.72, pc.grassParams.z);
    float bend = (calm_sway * (1.0 + gust_envelope * 0.45) + gust_lean)
        * pc.grassParams.y * height_fraction * height_fraction;

    vec3 world_position = vec3(
        position_height.x + local_xz.x + local_wind.x * bend,
        position_height.y + height_fraction * position_height.w - abs(bend) * 0.12,
        position_height.z + local_xz.y + local_wind.y * bend
    );

    vec4 position_view = pc.mv * vec4(world_position, 1.0);
    // Shadow projection is linear across each triangle, so doing this here avoids a matrix
    // multiply for every grass fragment.
    vShadowPosition = ubo.light_vp * position_view;
    vNormal = normalize(mat3(pc.mv) * vec3(normal_xz.x, inLocalNormal.y, normal_xz.y));
    vHeightFraction = height_fraction;
    vTint = inRotationWidthTintPhase.z;
    vFlowerHead = inFlowerHead;
    gl_Position = pc.mvp * vec4(world_position, 1.0);
}
