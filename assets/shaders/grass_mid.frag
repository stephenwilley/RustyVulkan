// --------------------------------------------------------------------------------------
// grass_mid.frag
//
// Distant blades cover very few pixels. They retain sun lighting and a single shadow
// lookup, avoiding the four-tap nearby filter where it cannot be seen.
// --------------------------------------------------------------------------------------
#version 450

#define MAX_LIGHTS 8

struct Light { vec3 position; float intensity; vec3 color; float _pad; };
struct DirLight { vec3 direction; float intensity; vec3 color; float _pad; };

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
    float shadow_depth = texture(shadowMap, uv).r;
    return (ndc.z - 0.001 <= shadow_depth) ? 1.0 : 0.45;
}

void main() {
    vec3 normal = normalize(vNormal);
    if (!gl_FrontFacing) normal = -normal;

    vec3 root_green = vec3(0.035, 0.16, 0.015);
    vec3 tip_green = vec3(0.20, 0.52, 0.045);
    vec3 albedo = mix(root_green, tip_green, vHeightFraction);
    albedo *= mix(0.78, 1.18, vTint);

    float diffuse = max(dot(normal, normalize(-ubo.sun.direction)), 0.0);
    vec3 lighting = 0.20 * albedo;
    lighting += shadow_factor(vShadowPosition)
        * ubo.sun.intensity * ubo.sun.color * diffuse * albedo;
    outColor = vec4(clamp(lighting, 0.0, 1.0), 1.0);
}
