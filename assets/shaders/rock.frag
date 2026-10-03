// Neutral-grey rock material. The source atlas contributes surface value and only a
// very small amount of its original hue, keeping five assets visually coherent.
#version 450
#define MAX_LIGHTS 8
#define SHADOW_CASCADE_COUNT 4

struct Light { vec3 position; float intensity; vec3 color; float _pad; };
struct DirLight { vec3 direction; float intensity; vec3 color; float _pad; };

layout(std140, set = 0, binding = 0) uniform GlobalUBO {
    DirLight sun;
    mat4 light_vp[SHADOW_CASCADE_COUNT];
    vec4 cascade_splits;
    Light lights[MAX_LIGHTS];
    uint light_count; uint _pad0, _pad1, _pad2;
} ubo;

layout(set = 0, binding = 1) uniform sampler2DArray shadowMap;
layout(set = 1, binding = 0) uniform sampler2D diffuseMap;

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec3 vFragPosView;
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
    vec3 atlas = texture(diffuseMap, vUV).rgb;
    float atlas_value = dot(atlas, vec3(0.2126, 0.7152, 0.0722));
    vec3 neutral_grey = vec3(mix(0.30, 0.48, atlas_value));
    // Four percent of the atlas chroma is enough to retain slight warm/cool identity
    // without allowing one source model to become visibly brown, green, or blue.
    vec3 albedo = neutral_grey + (atlas - vec3(atlas_value)) * 0.04;

    vec3 normal = normalize(vNormal);
    vec3 view_dir = normalize(-vFragPosView);
    vec3 lighting = 0.16 * albedo;

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
