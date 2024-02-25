#version 450
#
layout(set = 0, binding = 0) uniform CameraInfo {
    mat4 view;
    mat4 proj;
    vec4 position;
} camera;

layout(set = 1, binding = 0) uniform EnvTransform {
    mat4 transform;
} envTransform;
layout(set = 1, binding = 1) uniform samplerCube envCubemap;

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec4 worldPos;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 mirrorDir = reflect((worldPos - camera.position).xyz, normal);

    vec3 envLookupDir = (envTransform.transform * vec4(mirrorDir, 0.0)).xyz;
    outColor = fragColor * texture(envCubemap, envLookupDir);
}
