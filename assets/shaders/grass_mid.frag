// --------------------------------------------------------------------------------------
// grass_mid.frag
//
// Distant blades cover very few pixels. They retain sun lighting and a single shadow
// lookup, avoiding the four-tap nearby filter where it cannot be seen.
// --------------------------------------------------------------------------------------
#version 450
#extension GL_GOOGLE_include_directive : require
#include "global.glsl"

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
    float shadow_depth = texture(shadowMap, vec3(uv, float(cascade))).r;
    return (ndc.z - 0.001 <= shadow_depth) ? 1.0 : 0.45;
}

void main() {
    vec3 normal = normalize(vNormal);
    // Match the near shader: the two sides share one stylised vegetation normal.

    vec3 root_green = vec3(0.035, 0.16, 0.015);
    vec3 tip_green = vec3(0.20, 0.52, 0.045);
    // Match the near shader's vertical colour curve so an LOD change alters geometry,
    // not the average material colour of the whole chunk.
    vec3 albedo = mix(root_green, tip_green, smoothstep(0.0, 1.0, vHeightFraction));
    vec3 cool_patch = vec3(0.88, 1.03, 0.78);
    vec3 warm_patch = vec3(1.08, 0.97, 0.72);
    albedo *= mix(cool_patch, warm_patch, smoothstep(0.12, 0.88, vColourNoise));
    albedo *= mix(0.90, 1.10, vTint);
    albedo *= 0.85;

    float diffuse = max(dot(normal, normalize(-ubo.sun.direction)), 0.0);
    vec3 lighting = 0.20 * albedo;
    lighting += shadow_factor(vFragPosView)
        * ubo.sun.intensity * ubo.sun.color * diffuse * albedo;
    outColor = vec4(clamp(lighting, 0.0, 1.0), 1.0);
}
