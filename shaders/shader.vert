#version 450

layout(binding = 0) uniform CameraInfo {
    mat4 view;
    mat4 proj;
    vec4 position;
} camera;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec4 worldPos;

layout(push_constant) uniform pc {
    mat4 model;
};

void main() {
    worldPos = model * vec4(inPosition, 1.0);
    gl_Position = camera.proj * camera.view * worldPos;
    fragColor = inColor;
    fragNormal = mat3(transpose(inverse(model))) * inNormal;
}
