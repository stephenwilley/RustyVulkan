// --------------------------------------------------------------------------------------
// point_light.vert.slang – Vertex Shader
// --------------------------------------------------------------------------------------

#define MAX_LIGHTS 8

[[vk::push_constant]]
cbuffer Push {
    float4x4 mvp;
    float4x4 mv;
};

struct Light {
    float3 position;
    float intensity;
    float3 color;
    float _pad;
};

struct GlobalUBO {
    Light lights[MAX_LIGHTS];
    uint  light_count;
    uint3 _pad0;
};
[[vk::binding(0,0)]] ConstantBuffer<GlobalUBO> ubo;

struct VSInput {
    float3 inPos      [[vk::location(0)]];
    float3 inNormal   [[vk::location(1)]];
    float3 inColor    [[vk::location(2)]];
    float2 inUV       [[vk::location(3)]];
    float3 inTangent  [[vk::location(4)]];
    float3 inBitangent[[vk::location(5)]];
};

struct VSOutput {
    float2 UV            [[vk::location(0)]];
    float3 vT            [[vk::location(1)]];
    float3 vB            [[vk::location(2)]];
    float3 vN            [[vk::location(3)]];
    float3 vFragPosView  [[vk::location(4)]];
    float4 Position      [[vk::builtin_position]];
};

VSOutput main(VSInput IN) {
    VSOutput OUT;
    float3x3 mv3 = (float3x3)mv;
    OUT.vFragPosView = mul(mv, float4(IN.inPos,1.0)).xyz;
    OUT.vT = normalize(mul(mv3, IN.inTangent));
    OUT.vB = normalize(mul(mv3, IN.inBitangent));
    OUT.vN = normalize(mul(mv3, IN.inNormal));
    OUT.UV = float2(IN.inUV.x, 1.0 - IN.inUV.y);
    OUT.Position = mul(mvp, float4(IN.inPos,1.0));
    return OUT;
}
