#version 450

layout(set = 1, binding = 0) uniform EnvTransform {
    mat4 transform;
} envTransform;
layout(set = 1, binding = 1) uniform samplerCube envCubemap;

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 envLookupDir = (envTransform.transform * vec4(fragNormal, 0.0)).xyz;
    outColor = fragColor * texture(envCubemap, envLookupDir);
}
