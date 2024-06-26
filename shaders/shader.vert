#version 450
#include "common.glsl"

layout(binding = 0) uniform CameraInfoUBO {
    CameraInfo camera;
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inUV;
layout(location = 4) in vec4 inColor;

layout(location = 0) out VertexOutput frag;

layout(push_constant) uniform pc {
    mat4 model;
};

void main() {
    frag.worldPos = model * vec4(inPosition, 1.0);
    gl_Position = camera.proj * camera.view * frag.worldPos;

    frag.color = inColor;
    frag.uv = inUV;

    frag.normal = normalize(mat3(transpose(inverse(model))) * inNormal);
    frag.tangent = normalize(mat3(model) * inTangent.xyz);
    frag.bitangent = cross(frag.normal, frag.tangent) * inTangent.w;
    mat3 TBN = transpose(mat3(frag.tangent, frag.bitangent, frag.normal)); // Matrix that turns directions in world space to directions in tangent space

    frag.viewDir = (frag.worldPos - camera.position).xyz;
    frag.tangentViewDir = TBN * frag.viewDir;
}
