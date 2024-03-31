#version 450
#include "common.glsl"

layout(std430, binding = 0) buffer LightInfoSSBO {
    LightInfo lights[];
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inUV;
layout(location = 4) in vec4 inColor;

layout(push_constant) uniform pc {
    mat4 model;
    uint lightIndex;
};

void main() {
    vec4 worldPos = model * vec4(inPosition, 1.0);
    gl_Position = lights[lightIndex].projection * lights[lightIndex].transform * worldPos;
}
