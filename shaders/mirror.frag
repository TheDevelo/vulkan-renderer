#version 450
#include "common.glsl"

layout(set = 0, binding = 0) uniform CameraInfo {
    mat4 view;
    mat4 proj;
    vec4 position;
} camera;

layout(set = 1, binding = 0) uniform MaterialConstantsUBO {
    MaterialConstants materialConstants;
};
layout(set = 1, binding = 1) uniform sampler2D normalMap;
layout(set = 1, binding = 2) uniform sampler2D displacementMap;

layout(set = 2, binding = 0) uniform EnvTransform {
    mat4 envTransform;
};
layout(set = 2, binding = 1) uniform samplerCube envCubemap;

layout(location = 0) in VertexOutput frag;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = getNormal(frag, materialConstants, normalMap);
    vec3 mirrorDir = reflect((frag.worldPos - camera.position).xyz, normal);

    vec3 envLookupDir = (envTransform * vec4(mirrorDir, 0.0)).xyz;
    outColor = tonemap(frag.color * texture(envCubemap, envLookupDir));
}
