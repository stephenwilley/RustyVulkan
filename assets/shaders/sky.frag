// --------------------------------------------------------------------------------------
// sky.frag
//
// Maps a world-space direction onto a latitude/longitude panorama.  The texture wraps
// horizontally, so rotating through north never exposes a seam.
// --------------------------------------------------------------------------------------
#version 450

layout(set = 0, binding = 0) uniform sampler2D sky_panorama;

layout(push_constant) uniform SkyPush {
    mat4 inverse_view_projection;
    vec4 settings; // x = horizontal panorama rotation in radians
} pc;

layout(location = 0) in vec3 world_direction;
layout(location = 0) out vec4 out_color;

const float PI = 3.14159265358979323846;
const float TAU = 2.0 * PI;

void main() {
    vec3 direction = normalize(world_direction);
    float longitude = atan(direction.z, direction.x) / TAU + 0.5;
    float latitude = acos(clamp(direction.y, -1.0, 1.0)) / PI;
    vec2 uv = vec2(fract(longitude + pc.settings.x / TAU), latitude);
    out_color = texture(sky_panorama, uv);
}
