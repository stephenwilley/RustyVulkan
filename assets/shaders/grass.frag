// --------------------------------------------------------------------------------------
// grass.frag
//
// Opaque, double-sided grass lighting.  Geometry rather than alpha cards gives stable
// depth testing and avoids the expensive transparency sorting dense grass would need.
// --------------------------------------------------------------------------------------
#version 450

#define MAX_LIGHTS 8
#define SHADOW_CASCADE_COUNT 4

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
    uint light_count;
    uvec3 _pad0;
} ubo;

layout(set = 0, binding = 1) uniform sampler2DArray shadowMap;

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vFragPosView;
layout(location = 2) in float vHeightFraction;
layout(location = 3) in float vTint;
layout(location = 4) in float vFlowerHead;
layout(location = 5) in float vColourNoise;

layout(location = 0) out vec4 outColor;

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
    float visible = 0.0;
    const vec2 offsets[4] = vec2[](
        vec2(-0.5, -0.5), vec2(0.5, -0.5),
        vec2(-0.5,  0.5), vec2(0.5,  0.5)
    );
    for (int i = 0; i < 4; ++i) {
        float shadow_depth = texture(shadowMap,
            vec3(uv + offsets[i] * texel, float(cascade))).r;
        visible += (ndc.z - 0.001 <= shadow_depth) ? 1.0 : 0.45;
    }
    return visible * 0.25;
}

void main() {
    vec3 normal = normalize(vNormal);
    // Both sides of a zero-thickness blade share the vertex shader's upward-biased
    // vegetation normal. Flipping it on back faces would recreate the dark-side artifact.

    // Taller portions are brighter; per-instance tint breaks up a uniform green field.
    vec3 root_green = vec3(0.035, 0.16, 0.015);
    vec3 tip_green = vec3(0.20, 0.52, 0.045);
    vec3 albedo = mix(root_green, tip_green, smoothstep(0.0, 1.0, vHeightFraction));
    if (vFlowerHead > 0.5) {
        // A cluster chooses one of two deliberately simple stylised flower-head colours.
        vec3 muted_red = vec3(0.58, 0.10, 0.055);
        vec3 warm_white = vec3(0.92, 0.83, 0.58);
        albedo = mix(muted_red, warm_white, vTint);
    } else {
        // Broad warm/cool patches look grown rather than sprayed uniformly. A narrower
        // per-blade tint remains to stop neighbouring blades being perfectly identical.
        vec3 cool_patch = vec3(0.88, 1.03, 0.78);
        vec3 warm_patch = vec3(1.08, 0.97, 0.72);
        albedo *= mix(cool_patch, warm_patch, smoothstep(0.12, 0.88, vColourNoise));
        albedo *= mix(0.90, 1.10, vTint);
        albedo *= 0.85;
    }

    vec3 lighting = 0.20 * albedo;
    vec3 light_direction = normalize(-ubo.sun.direction);
    float diffuse = max(dot(normal, light_direction), 0.0);
    lighting += shadow_factor(vFragPosView) * ubo.sun.intensity * ubo.sun.color * diffuse * albedo;

    outColor = vec4(clamp(lighting, 0.0, 1.0), 1.0);
}
