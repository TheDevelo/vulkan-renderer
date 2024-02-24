#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    float light = dot(normalize(fragNormal), vec3(0,0,1)) * 0.5 + 0.5;
    outColor = vec4(fragColor.xyz * light, fragColor.w);
}
