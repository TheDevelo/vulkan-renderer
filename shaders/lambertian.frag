#version 450
#include "common.glsl"

layout(binding = 0) uniform CameraInfoUBO {
    CameraInfo camera;
};

layout(set = 1, binding = 0) uniform MaterialConstantsUBO {
    MaterialConstants materialConstants;
};
layout(set = 1, binding = 1) uniform sampler2D normalMap;
layout(set = 1, binding = 2) uniform sampler2D displacementMap;
layout(set = 1, binding = 3) uniform sampler2D albedoMap;

layout(set = 2, binding = 0) uniform EnvironmentInfoUBO {
    EnvironmentInfo envInfo;
};
layout(set = 2, binding = 2) uniform samplerCube lambertianCubemap;

layout(location = 0) in VertexOutput frag;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = getAdjustedUVs(frag, materialConstants, displacementMap);
    vec3 normal = getNormal(frag, materialConstants, normalMap, uv);

    vec3 envLookupDir = (envInfo.transform * vec4(normal, 0.0)).xyz;
    vec4 diffuseColor;
    if (materialConstants.useAlbedoMap) {
        diffuseColor = texture(albedoMap, uv);
    }
    else {
        diffuseColor = vec4(materialConstants.albedo, 1.0);
    }
    outColor = tonemap(frag.color * diffuseColor * texture(lambertianCubemap, envLookupDir), camera.exposure);
}
