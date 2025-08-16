// --------------------------------------------------------------------------------------
// point_light.frag.slang – Fragment Shader
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

[[vk::binding(1,0)]] Texture2D      diffuseMap;
[[vk::binding(1,1)]] SamplerState   diffuseSampler;
[[vk::binding(1,2)]] Texture2D      normalMap;
[[vk::binding(1,3)]] SamplerState   normalSampler;

struct FSInput {
    float2 UV           [[vk::location(0)]];
    float3 vT           [[vk::location(1)]];
    float3 vB           [[vk::location(2)]];
    float3 vN           [[vk::location(3)]];
    float3 vFragPosView [[vk::location(4)]];
};

[[vk::location(0)]]
float4 main(FSInput IN) {
    float3 albedo = diffuseMap.Sample(diffuseSampler, IN.UV).rgb;
    float3 normalTangent = normalize(normalMap.Sample(normalSampler, IN.UV).rgb * 2.0 - 1.0);

    float3x3 TBN = transpose(float3x3(IN.vT, IN.vB, IN.vN));
    float3 V = normalize(mul(TBN, -IN.vFragPosView));

    float shininess = 64.0;
    float3 ambient = 0.1 * albedo;
    float3 lighting = ambient;

    for (uint i = 0; i < ubo.light_count; ++i) {
        float3 Lview = ubo.lights[i].position - IN.vFragPosView;
        float dist = length(Lview);
        float3 L = normalize(mul(TBN, Lview));
        float3 H = normalize(L + V);
        float diff = max(dot(normalTangent, L), 0.0);
        float spec = pow(max(dot(normalTangent, H), 0.0), shininess);
        float attenuation = ubo.lights[i].intensity / (1.0 + 0.001 * dist * dist);
        float3 lightColor = ubo.lights[i].color;
        float3 diffuse  = diff * albedo;
        float3 specular = spec * float3(1.0, 1.0, 1.0);
        lighting += attenuation * lightColor * (diffuse + specular);
    }

    return float4(clamp(lighting, 0.0, 1.0), 1.0);
}
