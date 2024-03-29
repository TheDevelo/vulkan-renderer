#version 450
#include "common.glsl"

layout(binding = 0) uniform CameraInfoUBO {
    CameraInfo camera;
};
layout(binding = 1) uniform sampler2D pbrBRDF;

layout(set = 1, binding = 0) uniform MaterialConstantsUBO {
    MaterialConstants materialConstants;
};
layout(set = 1, binding = 1) uniform sampler2D normalMap;
layout(set = 1, binding = 2) uniform sampler2D displacementMap;
layout(set = 1, binding = 3) uniform sampler2D albedoMap;
layout(set = 1, binding = 4) uniform sampler2D roughnessMap;
layout(set = 1, binding = 5) uniform sampler2D metalnessMap;

layout(set = 2, binding = 0) uniform EnvironmentInfoUBO {
    EnvironmentInfo envInfo;
};
layout(set = 2, binding = 2) uniform samplerCube lambertianCubemap;
layout(set = 2, binding = 3) uniform samplerCube ggxCubemap;

layout(std430, set = 3, binding = 0) buffer LightInfoSSBO {
    LightInfo lights[];
};

layout(location = 0) in VertexOutput frag;

layout(location = 0) out vec4 outColor;

// The PBR mixing formulae are inspired by Google's Filament documentation:
// https://google.github.io/filament/Filament.md.html#materialsystem
void main() {
    vec2 uv = getAdjustedUVs(frag, materialConstants, displacementMap);
    vec3 normal = getNormal(frag, materialConstants, normalMap, uv);
    float cosThetaV = max(dot(-normalize(frag.viewDir), normal), 0.0);

    // Get the albedo, roughness, and metalness
    vec4 albedo;
    float roughness;
    float metalness;
    if (materialConstants.useAlbedoMap) {
        albedo = texture(albedoMap, uv);
    }
    else {
        albedo = vec4(materialConstants.albedo, 1.0);
    }
    if (materialConstants.useRoughnessMap) {
        roughness = texture(roughnessMap, uv).r;
    }
    else {
        roughness = materialConstants.roughness;
    }
    if (materialConstants.useMetalnessMap) {
        metalness = texture(metalnessMap, uv).r;
    }
    else {
        metalness = materialConstants.metalness;
    }
    vec3 f0 = 0.04 * (1 - metalness) + albedo.rgb * metalness;
    vec2 brdf = texture(pbrBRDF, vec2(cosThetaV, roughness)).xy;

    // We perform our own trilinear filtering, since the hardware filtering from Vulkan seems to have stepping between LOD levels for some reason
    float roughnessLOD = roughness * float(envInfo.ggxMipLevels);
    float baseRoughnessLOD = floor(roughnessLOD);
    float roughnessT = roughnessLOD - baseRoughnessLOD;

    vec3 mirrorDir = reflect(frag.viewDir, normal);
    vec3 diffuseLookupDir = (envInfo.transform * vec4(normal, 0.0)).xyz;
    vec3 specularLookupDir = (envInfo.transform * vec4(mirrorDir, 0.0)).xyz;

    // Calculate the specular and diffuse radiances
    vec3 diffuse = albedo.rgb * (1 - metalness) * texture(lambertianCubemap, diffuseLookupDir).rgb;
    vec3 specularEnv = textureLod(ggxCubemap, specularLookupDir, baseRoughnessLOD).rgb * (1.0 - roughnessT) + textureLod(ggxCubemap, specularLookupDir, baseRoughnessLOD + 1.0).rgb * roughnessT;
    vec3 specular = specularEnv * (f0 * brdf.x + brdf.y);

    outColor = tonemap(frag.color * vec4(diffuse + specular, albedo.a), camera.exposure);
}
