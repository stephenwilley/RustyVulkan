// --------------------------------------------------------------------------------------
// lambert.frag – Fragment Shader
//
// This shader implements a simple Lambertian reflectance model with a small ambient
// light component.
//
// Shader stage: Fragment  
// GLSL version: 450
// --------------------------------------------------------------------------------------

#version 450

layout(push_constant) uniform Push { mat4 mvp; mat4 mv; vec3 lightDir; } pc;

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 vTangent;
layout(location = 4) in vec3 vBitangent;

layout(location = 0) out vec4 outColor;

void main() {
    // Compute diffuse term with lightDir already in view space
    float diff = max(dot(vNormal, normalize(pc.lightDir)), 0.0);
    outColor  = vec4(clamp((diff * vColor) + (0.2 * vColor), 0.0, 1.0), 1.0);
}