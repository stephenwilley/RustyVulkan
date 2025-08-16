// --------------------------------------------------------------------------------------
// imgui.frag – Fragment Shader
//
// This shader is used by Dear ImGui to render UI elements. It applies a texture
// and vertex color to produce the final fragment color.
//
// Shader stage: Fragment
// GLSL version: 450
#version 450
layout(set = 0, binding = 0) uniform sampler2D Texture;
layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;
layout(location = 0) out vec4 out_color;
void main() {
    out_color = frag_color * texture(Texture, frag_uv);
}