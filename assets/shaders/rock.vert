// Textured rocks use their geometric normals: the deliberately faceted source geometry
// supplies the low-poly shading, while the atlas supplies only subtle colour variation.
#version 450

layout(push_constant) uniform Push { mat4 mvp; mat4 mv; vec4 uv_tiling; } pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec3 vFragPosView;

void main() {
    vUV = vec2(inUV.x, 1.0 - inUV.y);
    vNormal = normalize(mat3(pc.mv) * inNormal);
    vFragPosView = (pc.mv * vec4(inPos, 1.0)).xyz;
    gl_Position = pc.mvp * vec4(inPos, 1.0);
}
