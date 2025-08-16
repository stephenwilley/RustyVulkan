// --------------------------------------------------------------------------------------
// infinite_plane.frag – Fragment Shader
//
// This shader renders an infinite grid plane with major axes (X and Z) highlighted.
// It includes distance-based fading for the grid lines.
//
// Shader stage: Fragment
// GLSL version: 450
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

    float lineAlpha = 1.0 - smoothstep(0.0, 1.0, distToLine);
    vec4 color = vec4(0.3, 0.3, 0.3, lineAlpha);

    // Calculate distance to the Z axis (x=0) and X axis (z=0)
    float axisPixels = 1.5;         // desired axis half-width in pixels

    // Z-axis (blue) where x=0
    float zAxisAlpha = 1.0 - smoothstep(0.0, axisPixels * (derivative.x / scale), abs(fragPos3D.x));
    color.rgb = mix(color.rgb, vec3(0.3, 0.3, 1.0), zAxisAlpha);

    // X-axis (red) where z=0 (using derivative.y rather than z because it's a vec2 derivative calc)
    float xAxisAlpha = 1.0 - smoothstep(0.0, axisPixels * (derivative.y / scale), abs(fragPos3D.z));
    color.rgb = mix(color.rgb, vec3(1.0, 0.3, 0.3), xAxisAlpha);

    // Boost alpha coverage for axes so thickness matches axisPixels
    float axisAlpha = max(zAxisAlpha, xAxisAlpha);
    color.a = max(color.a, axisAlpha);

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