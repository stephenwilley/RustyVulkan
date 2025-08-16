// --------------------------------------------------------------------------------------
// imgui.frag.slang – Fragment Shader (Slang)
//
// Samples a texture and multiplies it by the interpolated vertex colour.
// --------------------------------------------------------------------------------------

struct FSInput {
    float2 frag_uv    [[vk::location(0)]];
    float4 frag_color [[vk::location(1)]];
};

[[vk::binding(0,0)]] Texture2D      Texture;
[[vk::binding(0,1)]] SamplerState   TextureSampler;

[[vk::location(0)]]
float4 main(FSInput IN) {
    return IN.frag_color * Texture.Sample(TextureSampler, IN.frag_uv);
}
