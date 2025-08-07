#version 450

layout(location = 0) in vec3 fragPos3D;
layout(location = 1) flat in vec3 cameraPos;
layout(location = 0) out vec4 outColor;

vec4 grid(vec3 fragPos3D, float scale) {
    // Scale the coordinates
    vec2 coord = fragPos3D.xz * scale;
    vec2 derivative = fwidth(coord);

    // Distance to nearest grid line in X/Z (scaled and screen-space)
    float thickness = 1.0;
    vec2 gridDist = abs(fract(coord - 0.5) - 0.5) / (derivative * thickness);
    float distToLine = min(gridDist.x, gridDist.y);

    vec4 color = vec4(0.3, 0.3, 0.3, 1.0 - min(distToLine, 1.0));

    // Calculate distance to the Z axis (x=0) and X axis (z=0)
    vec2 axisDist = abs(fragPos3D.xz) / (derivative * thickness);
    // Draw Z-axis (blue)
    float zAxisAlpha = 1.0 - clamp(axisDist.x, 0.0, 1.0);
    color.rgb = mix(color.rgb, vec3(0.3, 0.3, 1.0), zAxisAlpha);
    // Draw X-axis (red)
    float xAxisAlpha = 1.0 - clamp(axisDist.y, 0.0, 1.0);
    color.rgb = mix(color.rgb, vec3(1.0, 0.3, 0.3), xAxisAlpha);

    return color;
}

void main() {
    vec4 color = grid(fragPos3D, 1.0);
    float dist = length(cameraPos.xz - fragPos3D.xz);
    // Fade with distance but not for the first 1/4 of the distance
    float fadeDistance = 50.0;
    float fade = clamp(1.25 - dist / fadeDistance, 0.0, 1.0);
    outColor = vec4(color.rgb, color.a * fade);
}