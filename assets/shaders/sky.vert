// --------------------------------------------------------------------------------------
// sky.vert
//
// Generates one fullscreen triangle from gl_VertexIndex.  The inverse camera matrix turns
// each screen position back into a world-space viewing direction for the panorama lookup.
// Camera translation is deliberately absent, making the sky appear infinitely distant.
// --------------------------------------------------------------------------------------
#version 450

layout(push_constant) uniform SkyPush {
    mat4 inverse_view_projection;
    vec4 settings; // x = horizontal panorama rotation in radians
} pc;

layout(location = 0) out vec3 world_direction;

void main() {
    const vec2 POSITIONS[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );

    vec2 screen_position = POSITIONS[gl_VertexIndex];
    gl_Position = vec4(screen_position, 0.0, 1.0);

    // A point on the far clip plane becomes a direction once transformed by the
    // inverse projection and rotation-only view matrix.
    vec4 world_position = pc.inverse_view_projection * vec4(screen_position, 1.0, 1.0);
    world_direction = world_position.xyz / world_position.w;
}
