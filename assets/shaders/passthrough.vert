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

layout(push_constant) uniform Push { mat4 mvp; mat4 mv; vec3 lightDir; } pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vColor;

void main() {
    vNormal = normalize(mat3(pc.mv) * inNormal);
    vColor  = inColor;
    gl_Position  = pc.mvp * vec4(inPos, 1.0);
}