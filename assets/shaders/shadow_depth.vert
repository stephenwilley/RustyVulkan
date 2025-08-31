#version 450

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
} push_constants;

void main() {
    gl_Position = push_constants.mvp * vec4(inPosition, 1.0);
}