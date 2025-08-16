// --------------------------------------------------------------------------------------
// infinite_plane.frag.slang – Fragment Shader
// --------------------------------------------------------------------------------------

struct FSInput {
    float3 fragPos3D [[vk::location(0)]];
    float3 cameraPos [[vk::location(1)]];
};

[[vk::location(0)]]
float4 main(FSInput IN) {
    float2 coord = IN.fragPos3D.xz * 1.0;
    float2 derivative = fwidth(coord);

    float thickness = 1.0;
    float2 gridDist = abs(frac(coord - 0.5) - 0.5) / (derivative * thickness);
    float distToLine = min(gridDist.x, gridDist.y);

    float lineAlpha = 1.0 - smoothstep(0.0, 1.0, distToLine);
    float4 color = float4(0.3, 0.3, 0.3, lineAlpha);

    float axisPixels = 1.5;
    float zAxisAlpha = 1.0 - smoothstep(0.0, axisPixels * (derivative.x / 1.0), abs(IN.fragPos3D.x));
    color.rgb = lerp(color.rgb, float3(0.3, 0.3, 1.0), zAxisAlpha);

    float xAxisAlpha = 1.0 - smoothstep(0.0, axisPixels * (derivative.y / 1.0), abs(IN.fragPos3D.z));
    color.rgb = lerp(color.rgb, float3(1.0, 0.3, 0.3), xAxisAlpha);

    float axisAlpha = max(zAxisAlpha, xAxisAlpha);
    color.a = max(color.a, axisAlpha);

    float dist = length(IN.cameraPos.xz - IN.fragPos3D.xz);
    float fadeDistance = 50.0;
    float fade = clamp(1.25 - dist / fadeDistance, 0.0, 1.0);
    return float4(color.rgb, color.a * fade);
}
