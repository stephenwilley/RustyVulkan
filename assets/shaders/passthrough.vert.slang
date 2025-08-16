// --------------------------------------------------------------------------------------
// passthrough.vert.slang – Vertex Shader
// --------------------------------------------------------------------------------------

[[vk::push_constant]]
cbuffer Push {
    float4x4 mvp;
    float4x4 mv;
};

struct VSInput {
    float3 inPos      [[vk::location(0)]];
    float3 inNormal   [[vk::location(1)]];
    float3 inColor    [[vk::location(2)]];
    float2 inUV       [[vk::location(3)]];
    float3 inTangent  [[vk::location(4)]];
    float3 inBitangent[[vk::location(5)]];
};

struct VSOutput {
    float3 vNormal    [[vk::location(0)]];
    float3 vColor     [[vk::location(1)]];
    float2 outUV      [[vk::location(2)]];
    float3 vTangent   [[vk::location(3)]];
    float3 vBitangent [[vk::location(4)]];
    float4 Position   [[vk::builtin_position]];
};

VSOutput main(VSInput IN) {
    VSOutput OUT;
    float3x3 mv3 = (float3x3)mv;
    float3 T = normalize(mul(mv3, IN.inTangent));
    float3 B = normalize(mul(mv3, IN.inBitangent));
    float3 N = normalize(mul(mv3, IN.inNormal));

    OUT.vNormal    = N;
    OUT.vTangent   = T;
    OUT.vBitangent = B;
    OUT.vColor     = IN.inColor;
    OUT.outUV      = IN.inUV;
    OUT.Position   = mul(mvp, float4(IN.inPos, 1.0));
    return OUT;
}
