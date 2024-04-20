#version 450
#include "common.glsl"

layout(binding = 0) uniform CameraInfoUBO {
    CameraInfo camera;
};

layout(scalar, set = 1, binding = 0) uniform EnvironmentInfoUBO {
    EnvironmentInfo envInfo;
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inUV;
layout(location = 4) in vec4 inColor;

layout(location = 0) out vec3 localFragPos;

layout(push_constant) uniform pc {
    uint face;
    uint mipLevel;
};

const mat4 faceRotations[6] = mat4[6](
    // +X
    mat4(
        vec4(0, 0, -1, 0),
        vec4(0, -1, 0, 0),
        vec4(-1, 0, 0, 0),
        vec4(0, 0, 0, 1)
    ),
    // -X
    mat4(
        vec4(0, 0, 1, 0),
        vec4(0, -1, 0, 0),
        vec4(1, 0, 0, 0),
        vec4(0, 0, 0, 1)
    ),
    // +Y
    mat4(
        vec4(1, 0, 0, 0),
        vec4(0, 0, 1, 0),
        vec4(0, -1, 0, 0),
        vec4(0, 0, 0, 1)
    ),
    // -Y
    mat4(
        vec4(1, 0, 0, 0),
        vec4(0, 0, -1, 0),
        vec4(0, 1, 0, 0),
        vec4(0, 0, 0, 1)
    ),
    // +Z
    mat4(
        vec4(1, 0, 0, 0),
        vec4(0, -1, 0, 0),
        vec4(0, 0, -1, 0),
        vec4(0, 0, 0, 1)
    ),
    // -Z
    mat4(
        vec4(-1, 0, 0, 0),
        vec4(0, -1, 0, 0),
        vec4(0, 0, 1, 0),
        vec4(0, 0, 0, 1)
    )
);

void main() {
    localFragPos = inPosition;

    // Flip the model at the environment mirror plane
    vec4 mirrorPos = vec4(inPosition, 1.0);
    mirrorPos.z  = -(mirrorPos.z + envInfo.mirrorDist) - envInfo.mirrorDist;

    // Transform our env-space model into world-space
    // Would be more efficient to pre-compute the inverse, but I'm lazy and this shader will only be run on small models
    vec4 worldPos = inverse(envInfo.transform) * mirrorPos;

    // Move into camera-space based on the face and project
    gl_Position = camera.proj * faceRotations[face] * camera.view * worldPos;
}
