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
        vec4(0, 0, -1, 0),
        vec4(0, 1, 0, 0),
        vec4(0, 0, 0, 1)
    ),
    // -Y
    mat4(
        vec4(1, 0, 0, 0),
        vec4(0, 0, 1, 0),
        vec4(0, -1, 0, 0),
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

// Infinite projection matrix with FOV = 90, Aspect = 1, and zNear = 0.01
const mat4 projection = mat4(
    vec4(1, 0, 0, 0),
    vec4(0, -1, 0, 0),
    vec4(0, 0, -1, -1),
    vec4(0, 0, -0.01, 0)
);

void main() {
    localFragPos = inPosition;

    // Reflect the camera across the mirror plane
    vec4 cameraPos = envInfo.transform * camera.position;
    cameraPos.z = -(cameraPos.z + envInfo.mirrorDist) - envInfo.mirrorDist;

    // Move the world so that the reflected camera is the origin, and then rotate to face our face and project
    gl_Position = projection * faceRotations[face] * vec4(inPosition - cameraPos.xyz, 1.0);
}
