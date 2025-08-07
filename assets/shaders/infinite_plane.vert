#version 450

layout(push_constant) uniform Push { mat4 mvp; mat4 mv; vec3 lightPos; float lightIntensity; } pc;

layout(location = 0) in vec3 inPos;

layout(location = 0) out vec3 fragPos3D;
layout(location = 1) flat out vec3 cameraPos;

void main() {
    // Place quad in world space on the XZ plane under the camera and make it big
    cameraPos = inverse(pc.mv)[3].xyz;
    vec4 worldPos = vec4(
        inPos.x * 1000.0 + cameraPos.x,
        0.0,
        inPos.z * 1000.0 + cameraPos.z,
        1.0
    );
    worldPos.y = 0.0; // force onto ground plane

    fragPos3D = worldPos.xyz;

    // Standard MVP for depth and screen position
    gl_Position = pc.mvp * worldPos;
}
