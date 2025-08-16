// --------------------------------------------------------------------------------------
// infinite_plane.vert.slang – Vertex Shader
// --------------------------------------------------------------------------------------

[[vk::push_constant]]
cbuffer Push {
    float4x4 mvp;
    float4x4 mv;
};

struct VSInput {
    float3 inPos [[vk::location(0)]];
};

struct VSOutput {
    float3 fragPos3D [[vk::location(0)]];
    float3 cameraPos [[vk::location(1)]];
    float4 Position  [[vk::builtin_position]];
};

VSOutput main(VSInput IN) {
    VSOutput OUT;
    float3 cam = mul(inverse(mv), float4(0.0, 0.0, 0.0, 1.0)).xyz;
    float3 worldPos = float3(
        IN.inPos.x * 1000.0 + cam.x,
        0.0,
        IN.inPos.z * 1000.0 + cam.z
    );
    worldPos.y = 0.0;
    OUT.fragPos3D = worldPos;
    OUT.Position  = mul(mvp, float4(worldPos, 1.0));
    OUT.cameraPos = cam;
    return OUT;
}
