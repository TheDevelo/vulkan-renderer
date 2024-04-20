#version 450
#include "common.glsl"

layout(scalar, set = 1, binding = 0) uniform EnvironmentInfoUBO {
    EnvironmentInfo envInfo;
};
layout(set = 1, binding = 4) uniform samplerCube sourceCubemap;

layout(location = 0) in vec3 localFragPos;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform pc {
    uint face;
    uint mipLevel;
};

void main() {
    vec3 envLookupDir = normalize(localFragPos);
    outColor = textureLod(sourceCubemap, envLookupDir, mipLevel);
}
