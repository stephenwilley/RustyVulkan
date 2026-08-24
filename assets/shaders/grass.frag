// --------------------------------------------------------------------------------------
// grass.frag
//
// Opaque, double-sided grass lighting.  Geometry rather than alpha cards gives stable
// depth testing and avoids the expensive transparency sorting dense grass would need.
// --------------------------------------------------------------------------------------
#version 450

#define MAX_LIGHTS 8

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

layout(set = 0, binding = 1) uniform sampler2D shadowMap;

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec4 vShadowPosition;
layout(location = 2) in float vHeightFraction;
layout(location = 3) in float vTint;
layout(location = 4) in float vFlowerHead;

layout(location = 0) out vec4 outColor;

float shadow_factor(vec4 light_pos) {
    if (light_pos.w <= 0.0) return 1.0;

    vec3 ndc = light_pos.xyz / light_pos.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0
        || ndc.z <= 0.0 || ndc.z >= 1.0) return 1.0;

    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));
    float visible = 0.0;
    const vec2 offsets[4] = vec2[](
        vec2(-0.5, -0.5), vec2(0.5, -0.5),
        vec2(-0.5,  0.5), vec2(0.5,  0.5)
    );
    for (int i = 0; i < 4; ++i) {
        float shadow_depth = texture(shadowMap, uv + offsets[i] * texel).r;
        visible += (ndc.z - 0.001 <= shadow_depth) ? 1.0 : 0.45;
    }
    return visible * 0.25;
}

void main() {
    vec3 normal = normalize(vNormal);
    if (!gl_FrontFacing) normal = -normal;

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
        albedo *= mix(0.78, 1.18, vTint);
    }

    vec3 lighting = 0.20 * albedo;
    vec3 light_direction = normalize(-ubo.sun.direction);
    float diffuse = max(dot(normal, light_direction), 0.0);
    lighting += shadow_factor(vShadowPosition) * ubo.sun.intensity * ubo.sun.color * diffuse * albedo;

    outColor = vec4(clamp(lighting, 0.0, 1.0), 1.0);
}
