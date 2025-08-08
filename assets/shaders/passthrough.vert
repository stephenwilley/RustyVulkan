// --------------------------------------------------------------------------------------
// passthrough.vert – Vertex Shader
//
// This shader takes in vertex positions and colors, and outputs the position
// and color for each vertex to be used in the fragment shader.
// It is designed to work with a simple mesh, where each vertex has
// a position in 3D space and a color in RGB format.
//
// Shader stage: Vertex  
// GLSL version: 450
// --------------------------------------------------------------------------------------

#version 450

layout(push_constant) uniform Push { mat4 mvp; mat4 mv; } pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;
layout(location = 4) in vec3 inTangent;
layout(location = 5) in vec3 inBitangent;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vColor;
layout(location = 2) out vec2 outUV;
layout(location = 3) out vec3 vTangent;
layout(location = 4) out vec3 vBitangent;

void main() {
    vec3 T = normalize(mat3(pc.mv) * inTangent);
    vec3 B = normalize(mat3(pc.mv) * inBitangent);
    vec3 N = normalize(mat3(pc.mv) * inNormal);

    vNormal = N;
    vTangent = T;
    vBitangent = B;
    vColor  = inColor;
    outUV   = inUV;
    gl_Position  = pc.mvp * vec4(inPos, 1.0);
}