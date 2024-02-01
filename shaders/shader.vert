#version 450

layout(binding = 0) uniform MVPMatrices {
    mat4 model;
    mat4 viewProj;
} mvp;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexUV;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexUV;

void main() {
    gl_Position = mvp.viewProj * mvp.model * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
    fragTexUV = inTexUV;
}
