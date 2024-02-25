#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(fragNormal);

    float light = dot(normal, vec3(0,0,1)) * 0.5 + 0.5;
    outColor = vec4(fragColor.xyz * light, fragColor.w);
}
