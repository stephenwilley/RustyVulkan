// --------------------------------------------------------------------------------------
// point_light.vert – Vertex Shader
//
// This shader transforms vertex attributes (position, normal, tangent, bitangent)
// into view space and tangent space, and calculates the halfway direction for
// Blinn-Phong shading in the fragment shader. It also handles UV coordinate flipping.
//
// Shader stage: Vertex  
// GLSL version: 450
// --------------------------------------------------------------------------------------

#version 450

layout(push_constant) uniform Push { mat4 mvp; mat4 mv; } pc;
layout(set = 0, binding = 0) uniform GlobalUBO {
    vec3 lightPos;
    float lightIntensity;
} ubo;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;
layout(location = 4) in vec3 inTangent;
layout(location = 5) in vec3 inBitangent;

layout(location = 0) out vec3 vHalfwayDirTangent;
layout(location = 1) out vec3 vColor;
layout(location = 2) out vec2 outUV;
layout(location = 3) out vec3 vLightTangent;
layout(location = 4) out vec3 fragPosView;

void main() {
    fragPosView = (pc.mv * vec4(inPos, 1.0)).xyz;
    vec3 lightDirView = ubo.lightPos - fragPosView;
    vec3 viewDir = -fragPosView;
    vec3 halfwayDir = normalize(normalize(lightDirView) + normalize(viewDir));


    vec3 T = normalize(mat3(pc.mv) * inTangent);
    vec3 B = normalize(mat3(pc.mv) * inBitangent);
    vec3 N = normalize(mat3(pc.mv) * inNormal);
    mat3 invTBN = transpose(mat3(T, B, N));
    vec3 lightDirInTangentSpace = normalize(invTBN * lightDirView);
    vec3 halfwayDirInTangentSpace = normalize(invTBN * halfwayDir);

    vColor  = inColor;
    // Vulkan's lovely inverted Y.  Doing the flip here is more efficient than doing it in the frag shader because it's run fewer times.
    outUV   = vec2(inUV.x, 1 - inUV.y);
    vLightTangent = lightDirInTangentSpace;
    vHalfwayDirTangent = halfwayDirInTangentSpace;

    gl_Position  = pc.mvp * vec4(inPos, 1.0);
}