// --------------------------------------------------------------------------------------
// lambert_no_tex.frag.slang – Fragment Shader
// --------------------------------------------------------------------------------------

[[vk::push_constant]]
cbuffer Push {
    float4x4 mvp;
    float4x4 mv;
};

struct GlobalUBO {
    float3 lightPos;
    float lightIntensity;
};
[[vk::binding(0,0)]] ConstantBuffer<GlobalUBO> ubo;

struct FSInput {
    float3 vNormal    [[vk::location(0)]];
    float3 vColor     [[vk::location(1)]];
    float2 uv         [[vk::location(2)]];
    float3 vTangent   [[vk::location(3)]];
    float3 vBitangent [[vk::location(4)]];
};

[[vk::location(0)]]
float4 main(FSInput IN) {
    float diff = max(dot(IN.vNormal, normalize(ubo.lightPos)), 0.0);
    float3 color = clamp((diff * IN.vColor) + (0.2 * IN.vColor), 0.0, 1.0);
    return float4(color, 1.0);
}
