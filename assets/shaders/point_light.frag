// --------------------------------------------------------------------------------------
// point_light.frag – Fragment Shader
//
// Blinn-Phong shading with a point light source.
//
// Shader stage: Fragment  
// GLSL version: 450
// --------------------------------------------------------------------------------------

#version 450

layout(push_constant) uniform Push { mat4 mvp; mat4 mv; } pc;
layout(set = 0, binding = 0) uniform GlobalUBO {
    vec3 lightPos;
    float lightIntensity;
} ubo;

layout(set = 1, binding = 0) uniform sampler2D diffuseMap;
layout(set = 1, binding = 1) uniform sampler2D normalMap;

layout(location = 0) in vec3 vHalfwayDirTangent;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 vLightTangent;
layout(location = 4) in vec3 fragPosView;

layout(location = 0) out vec4 outColor;

void main() {
    // Get the texture color
    vec4 texColor = texture(diffuseMap, uv);

    // Compute the normal
    // Sample the normal map (RGB in [0,1]) and remap to [-1,1]
    vec3 normalSample = texture(normalMap, uv).rgb;
    vec3 normalTangent = normalize(normalSample * 2.0 - 1.0);

    // Calculate attenuation
    vec3 lightDir = ubo.lightPos - fragPosView;
    float dist = length(lightDir);
    float attenuation = ubo.lightIntensity / (1.0 + 0.001 * dist * dist);

    // Compute diffuse term
    float diff = max(dot(normalTangent, normalize(vLightTangent)), 0.0);

    // Compute specular term
    float shininess = 64.0;
    float spec = pow(max(dot(normalTangent, vHalfwayDirTangent), 0.0), shininess);      

    vec3 ambient = 0.1 * texColor.rgb;
    vec3 diffuse = diff * texColor.rgb;
    vec3 specular = spec * vec3(1.0);
    vec3 litColor = attenuation * (diffuse + specular) + ambient;

    outColor = vec4(clamp(litColor, 0.0, 1.0), 1.0);
}