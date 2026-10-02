// Dedicated terrain path: vertex colour carries the shared CPU biome result,
// while local X/Z remains available for stable procedural surface detail.
#version 450

layout(push_constant) uniform Push { mat4 mvp; mat4 mv; vec4 uv_tiling; } pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 vColor;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec3 vFragPosView;
layout(location = 3) out vec2 vTerrainPosition;

void main() {
    vColor = inColor;
    vNormal = normalize(mat3(pc.mv) * inNormal);
    vFragPosView = (pc.mv * vec4(inPos, 1.0)).xyz;
    vTerrainPosition = inPos.xz;
    gl_Position = pc.mvp * vec4(inPos, 1.0);
}
