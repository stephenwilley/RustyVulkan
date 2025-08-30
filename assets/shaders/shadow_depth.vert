// --------------------------------------------------------------------------------------
// shadow_depth.vert – Vertex Shader (depth-only)
//
// Transforms positions by a light view-projection matrix. Matches push-constant
// layout used in the main pass: two mat4's (mvp, mv). Here we only use `pc.mvp`.
// --------------------------------------------------------------------------------------
#version 450

layout(push_constant) uniform Push { mat4 mvp; mat4 mv; } pc;

layout(location = 0) in vec3 inPos;

void main() {
    gl_Position = pc.mvp * vec4(inPos, 1.0);
}

