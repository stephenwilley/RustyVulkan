// --------------------------------------------------------------------------------------
// point_light.vert – Vertex Shader
//
// Transforms vertex attributes to view space and passes minimal data needed for
// correct per-fragment lighting with multiple point lights.
//
// GLSL 450
// --------------------------------------------------------------------------------------
#version 450
// Keep in sync with Rust `MAX_LIGHTS`
#define MAX_LIGHTS 8

layout(push_constant) uniform Push { mat4 mvp; mat4 mv; } pc;

// Declare Light before using it in UBO
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

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;
layout(location = 4) in vec3 inTangent;
layout(location = 5) in vec3 inBitangent;

layout(location = 0) out vec2 UV;
layout(location = 1) out vec3 vT;
layout(location = 2) out vec3 vB;
layout(location = 3) out vec3 vN;
layout(location = 4) out vec3 vFragPosView;

void main() {
    // View-space position of the fragment (for per-fragment light vectors)
    vFragPosView = (pc.mv * vec4(inPos, 1.0)).xyz;

    // Build T, B, N in view space for correct normal mapping
    mat3 mv3 = mat3(pc.mv);
    vT = normalize(mv3 * inTangent);
    vB = normalize(mv3 * inBitangent);
    vN = normalize(mv3 * inNormal);
    UV    = vec2(inUV.x, 1.0 - inUV.y); // flip Y for Vulkan
    gl_Position = pc.mvp * vec4(inPos, 1.0);
}
