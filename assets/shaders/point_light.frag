// --------------------------------------------------------------------------------------
// point_light.frag – Fragment Shader
//
// Blinn-Phong shading with multiple point lights, computed per-fragment.
// Uses tangent-space normal mapping.
//
// GLSL 450
// --------------------------------------------------------------------------------------
#version 450
// Keep in sync with Rust `MAX_LIGHTS`
#define MAX_LIGHTS 8

layout(push_constant) uniform Push { mat4 mvp; mat4 mv; } pc;

struct Light {
    vec3 position;
    float intensity;
    vec3 color;
    float _pad;    // keep 16-byte stride
};

struct DirLight {
    vec3 direction; // in view space
    float intensity;
    vec3 color;
    float _pad1;
};

layout(set = 0, binding = 0) uniform GlobalUBO {
    DirLight sun;
    Light    lights[MAX_LIGHTS];
    uint     light_count;
    uvec3    _pad0;
} ubo;

layout(set = 1, binding = 0) uniform sampler2D diffuseMap;
layout(set = 1, binding = 1) uniform sampler2D normalMap;

layout(location = 0) in vec2 UV;
layout(location = 1) in vec3 vT;
layout(location = 2) in vec3 vB;
layout(location = 3) in vec3 vN;
layout(location = 4) in vec3 vFragPosView;

layout(location = 0) out vec4 outColor;

void main() {
    // Albedo
    vec3 albedo = texture(diffuseMap, UV).rgb;

    // Tangent-space normal from normal map
    vec3 normalTangent = normalize(texture(normalMap, UV).rgb * 2.0 - 1.0);

    // Build TBN and view dir in tangent space
    mat3 TBN = transpose(mat3(vT, vB, vN));
    vec3 V = normalize(TBN * (-vFragPosView));

    float shininess = 64.0;
    vec3 ambient = 0.1 * albedo;
    vec3 lighting = ambient;

    // Directional light (view-space direction). L points from fragment toward light.
    vec3 Ldir = normalize(TBN * (-ubo.sun.direction));
    vec3 Hdir = normalize(Ldir + V);
    float diff_dir = max(dot(normalTangent, Ldir), 0.0);
    float spec_dir = pow(max(dot(normalTangent, Hdir), 0.0), shininess);
    lighting += ubo.sun.intensity * ubo.sun.color * (diff_dir * albedo + spec_dir * vec3(1.0));

    for (uint i = 0; i < ubo.light_count; ++i) {
        // Light vector in view space and tangent space
        vec3 Lview = ubo.lights[i].position - vFragPosView;
        float dist = length(Lview);
        vec3 L = normalize(TBN * Lview);

        // Half vector (per-fragment)
        vec3 H = normalize(L + V);

        // Lambert + Blinn-Phong
        float diff = max(dot(normalTangent, L), 0.0);
        float spec = pow(max(dot(normalTangent, H), 0.0), shininess);

        // Quadratic attenuation (tweak as desired)
        float attenuation = ubo.lights[i].intensity / (1.0 + 0.001 * dist * dist);

        vec3 lightColor = ubo.lights[i].color;
        vec3 diffuse  = diff * albedo;
        vec3 specular = spec * vec3(1.0);

        lighting += attenuation * lightColor * (diffuse + specular);
    }

    outColor = vec4(clamp(lighting, 0.0, 1.0), 1.0);
}
