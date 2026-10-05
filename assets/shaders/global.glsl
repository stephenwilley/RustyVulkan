// Shared set 0, binding 0 interface. Match src/graphics/gpu_data.rs.
// Scalar pads are deliberate: std140 gives uvec3 a different alignment.
#ifndef RUSTY_VULKAN_GLOBAL_GLSL
#define RUSTY_VULKAN_GLOBAL_GLSL

#define MAX_LIGHTS 8
#define SHADOW_CASCADE_COUNT 4

struct Light {
    vec3 position;
    float intensity;
    vec3 color;
    float _pad;
};

struct DirLight {
    vec3 direction; // view space
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
    uint _pad0, _pad1, _pad2;
} ubo;

#endif
