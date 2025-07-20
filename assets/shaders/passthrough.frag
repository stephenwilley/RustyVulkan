// --------------------------------------------------------------------------------------
// passthrough.frag – Fragment Shader
//
// A minimal “pass-through” shader that takes the interpolated RGB colour from the
// vertex shader and outputs it with full opacity (alpha = 1.0).
//
// Shader stage: Fragment
// GLSL version: 450
// --------------------------------------------------------------------------------------

#version 450

layout(push_constant) uniform Push { mat4 mvp; vec3 lightDir; } pc;

layout(location = 1) in vec3 fragColour;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColour, 1.0);
}