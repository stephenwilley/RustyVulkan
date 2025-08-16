// --------------------------------------------------------------------------------------
// imgui.vert.slang – Vertex Shader (Slang)
//
// Transforms ImGui vertices into clip space while passing through
// texture coordinates and colours.
// --------------------------------------------------------------------------------------

struct VSInput {
    float2 Position [[vk::location(0)]];
    float2 UV       [[vk::location(1)]];
    float4 Color    [[vk::location(2)]];
};

[[vk::push_constant]]
cbuffer PushConsts {
    float4x4 ProjMtx;
};

struct VSOutput {
    float2 frag_uv    [[vk::location(0)]];
    float4 frag_color [[vk::location(1)]];
    float4 Position   [[vk::builtin_position]];
};

VSOutput main(VSInput IN) {
    VSOutput OUT;
    OUT.frag_uv    = IN.UV;
    OUT.frag_color = IN.Color;
    OUT.Position   = mul(ProjMtx, float4(IN.Position, 0.0, 1.0));
    return OUT;
}
