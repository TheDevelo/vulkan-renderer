#version 450

layout(binding = 0) uniform MVPMatrices {
    mat4 view;
    mat4 proj;
} viewProj;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexUV;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexUV;

layout(push_constant) uniform pc {
    mat4 model;
};

void main() {
    gl_Position = viewProj.proj * viewProj.view * model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexUV = inTexUV;
}
