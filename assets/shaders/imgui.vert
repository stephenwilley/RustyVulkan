// assets/shaders/imgui.vert
#version 450
layout(location = 0) in vec2 Position;
layout(location = 1) in vec2 UV;
layout(location = 2) in vec4 Color;
layout(push_constant) uniform PushConsts {
    mat4 ProjMtx;
} pc;
layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec4 frag_color;
void main() {
    frag_uv    = UV;
    frag_color = Color;
    gl_Position = pc.ProjMtx * vec4(Position, 0.0, 1.0);
}