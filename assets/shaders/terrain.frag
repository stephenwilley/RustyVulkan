// Lit terrain with inexpensive procedural colour detail. The broad biome colours
// are generated on the CPU, so this shader only adds texture at metre/centimetre scale.
#version 450
#define MAX_LIGHTS 8
#define SHADOW_CASCADE_COUNT 4

layout(push_constant) uniform Push { mat4 mvp; mat4 mv; vec4 uv_tiling; } pc;

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
    mat4 light_vp[SHADOW_CASCADE_COUNT];
    vec4 cascade_splits;
    Light lights[MAX_LIGHTS];
    uint light_count; uvec3 _pad0;
} ubo;

layout(set = 0, binding = 1) uniform sampler2DArray shadowMap;

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec3 vFragPosView;
layout(location = 3) in vec2 vTerrainPosition;
layout(location = 0) out vec4 outColor;

float hash21(vec2 point) {
    vec3 p = fract(vec3(point.xyx) * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float value_noise(vec2 point) {
    vec2 cell = floor(point);
    vec2 fraction = fract(point);
    fraction = fraction * fraction * (3.0 - 2.0 * fraction);
    float lower = mix(hash21(cell), hash21(cell + vec2(1.0, 0.0)), fraction.x);
    float upper = mix(hash21(cell + vec2(0.0, 1.0)), hash21(cell + vec2(1.0)), fraction.x);
    return mix(lower, upper, fraction.y);
}

float shadow_factor(vec3 frag_pos_view) {
    float view_depth = -frag_pos_view.z;
    if (view_depth > ubo.cascade_splits.w) return 1.0;
    int cascade = view_depth > ubo.cascade_splits.z ? 3
                : view_depth > ubo.cascade_splits.y ? 2
                : view_depth > ubo.cascade_splits.x ? 1 : 0;
    vec4 light_pos = ubo.light_vp[cascade] * vec4(frag_pos_view, 1.0);
    if (light_pos.w <= 0.0) return 1.0;

    vec3 ndc = light_pos.xyz / light_pos.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0
        || ndc.z <= 0.0 || ndc.z >= 1.0) return 1.0;

    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0).xy);
    float lit_samples = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float stored_depth = texture(shadowMap,
                vec3(uv + vec2(x, y) * texel, float(cascade))).r;
            lit_samples += (ndc.z - 0.0005 <= stored_depth) ? 1.0 : 0.35;
        }
    }
    return lit_samples / 9.0;
}

void main() {
    float broad_detail = value_noise(vTerrainPosition * 0.58);
    float fine_detail = value_noise(vTerrainPosition * 3.1 + 17.0);
    float grit_detail = value_noise(vTerrainPosition * 11.0 + 43.0);
    float colour_detail = 0.80 + broad_detail * 0.24 + fine_detail * 0.11;
    vec3 albedo = vColor * colour_detail;
    // Two continuous noise scales break up the soil before distinct little pebbles are
    // added below. Fragment detail keeps the one-metre terrain mesh inexpensive.
    float coarse_grit = smoothstep(0.62, 0.92, grit_detail);
    float micro_grit = value_noise(vTerrainPosition * 27.0 + 91.0);
    albedo *= 0.91 + micro_grit * 0.17 - coarse_grit * 0.16;

    // One jittered pebble candidate per 20 cm cell. fwidth softens sub-pixel edges so the
    // stronger grit does not turn into distance shimmer.
    vec2 pebble_grid = vTerrainPosition * 5.0;
    vec2 pebble_cell = floor(pebble_grid);
    vec2 pebble_offset = vec2(
        hash21(pebble_cell + 13.7),
        hash21(pebble_cell + 47.1)
    ) - 0.5;
    float pebble_seed = hash21(pebble_cell + 79.3);
    float pebble_distance = length(fract(pebble_grid) - 0.5 - pebble_offset * 0.45);
    float pebble_radius = mix(0.08, 0.23, hash21(pebble_cell + 101.9));
    float pebble_edge = max(fwidth(pebble_distance), 0.015);
    float pebble = step(0.70, pebble_seed)
        * (1.0 - smoothstep(pebble_radius - pebble_edge, pebble_radius + pebble_edge,
                            pebble_distance));
    vec3 pebble_colour = mix(vec3(0.075, 0.065, 0.050), vec3(0.29, 0.27, 0.22),
                             hash21(pebble_cell + 131.2));
    albedo = mix(albedo, pebble_colour, pebble * 0.58);

    vec3 normal = normalize(vNormal);
    vec3 view_dir = normalize(-vFragPosView);
    vec3 lighting = 0.14 * albedo;

    vec3 sun_dir = normalize(-ubo.sun.direction);
    float sun_diffuse = max(dot(normal, sun_dir), 0.0);
    vec3 sun_half = normalize(sun_dir + view_dir);
    float sun_specular = pow(max(dot(normal, sun_half), 0.0), 48.0);
    lighting += shadow_factor(vFragPosView) * ubo.sun.intensity * ubo.sun.color
        * (sun_diffuse * albedo + 0.025 * sun_specular);

    for (uint i = 0; i < ubo.light_count; ++i) {
        vec3 to_light = ubo.lights[i].position - vFragPosView;
        float distance_to_light = length(to_light);
        vec3 light_dir = to_light / max(distance_to_light, 0.0001);
        float diffuse = max(dot(normal, light_dir), 0.0);
        float attenuation = ubo.lights[i].intensity
            / (1.0 + 0.001 * distance_to_light * distance_to_light);
        lighting += attenuation * ubo.lights[i].color * diffuse * albedo;
    }

    outColor = vec4(clamp(lighting, 0.0, 1.0), 1.0);
}
