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

layout(std140, set = 0, binding = 0) uniform GlobalUBO {
    DirLight sun;                                  // directional light first
    mat4     light_vp;                             // light VP
    Light    lights[MAX_LIGHTS];                   // array of point lights
    uint     light_count; uvec3 _pad0;             // count + pad
} ubo;

layout(set = 0, binding = 1) uniform sampler2D shadowMap;
layout(set = 1, binding = 0) uniform sampler2D diffuseMap;
layout(set = 1, binding = 1) uniform sampler2D normalMap;

layout(location = 0) in vec2 UV;
layout(location = 1) in vec3 vT;
layout(location = 2) in vec3 vB;
layout(location = 3) in vec3 vN;
layout(location = 4) in vec3 vFragPosView;

layout(location = 0) out vec4 outColor;

float shadow_factor(vec3 fragPosView) {
    // Transform from VIEW space directly to LIGHT clip using ubo.light_vp (pre-multiplied with inverse(view))
    vec4 lightPos = ubo.light_vp * vec4(fragPosView, 1.0);

    // Perspective divide to NDC
    if (lightPos.w <= 0.0) {
        return 1.0; // behind light; treat as lit
    }
    vec3 ndc = lightPos.xyz / lightPos.w;
    // Convert to [0,1]
    vec2 uv = ndc.xy * 0.5 + 0.5;
    float depth = ndc.z; // already in [0,1] after Vulkan depth correction in CPU

    // Outside shadow map
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }

    // Small receiver bias to reduce acne (raster depth bias handles most)
    float bias = 0.0005;

    // Manual PCF with a few jittered taps (compat path for platforms without compare samplers).
    // Poisson-ish offsets (in texels):
    const vec2 OFFS[6] = vec2[](
        vec2( 0.0,  0.0),
        vec2( 0.35, 0.12),
        vec2(-0.28,-0.34),
        vec2(-0.57, 0.49),
        vec2( 0.62,-0.41),
        vec2( 0.15, 0.68)
    );
    ivec2 ts = textureSize(shadowMap, 0);
    vec2 texel = 1.0 / vec2(ts);
    float radius = 1.5; // in texels; tweak per taste

    float sum = 0.0;
    for (int i = 0; i < 6; ++i) {
        vec2 uvOff = uv + OFFS[i] * texel * radius;
        float sm = texture(shadowMap, uvOff).r;
        sum += (depth - bias <= sm) ? 1.0 : 0.4;
    }
    return sum / 6.0;
}

void main() {
    // Debug visualization removed
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
    float sun_visibility = shadow_factor(vFragPosView);
    lighting += sun_visibility * ubo.sun.intensity * ubo.sun.color * (diff_dir * albedo + spec_dir * vec3(1.0));

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
