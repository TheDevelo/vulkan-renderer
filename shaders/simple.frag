#version 450
#include "common.glsl"

layout(set = 1, binding = 0) uniform MaterialConstantsUBO {
    MaterialConstants materialConstants;
};
layout(set = 1, binding = 1) uniform sampler2D normalMap;
layout(set = 1, binding = 2) uniform sampler2D displacementMap;

layout(location = 0) in VertexOutput frag;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = getAdjustedUVs(frag, materialConstants, displacementMap);
    vec3 normal = getNormal(frag, materialConstants, normalMap, uv);

    float light = dot(normal, vec3(0,0,1)) * 0.5 + 0.5;
    outColor = tonemap(vec4(frag.color.rgb * light, frag.color.a));
}
