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

layout(location = 0) out vec4 outColor;

void main() {
    float diff = max(dot(normalize(vNormal), normalize(pc.lightDir)), 0.0);
    vec3 lit  = (vColor * diff) + (vColor * 0.1);  // 0.1 is the ambient light factor
    outColor  = vec4(lit, 1.0);
}