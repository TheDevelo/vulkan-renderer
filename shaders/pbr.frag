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

layout(location = 0) in VertexOutput frag;

layout(location = 0) out vec4 outColor;

// The PBR mixing formulae are inspired by glTF's mixing formulae:
// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
void main() {
    vec2 uv = getAdjustedUVs(frag, materialConstants, displacementMap);
    vec3 normal = getNormal(frag, materialConstants, normalMap, uv);
    float cosThetaV = abs(dot(-normalize(frag.viewDir), normal));

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
    vec2 brdf = texture(pbrBRDF, vec2(cosThetaV, roughness)).xy;

    // We perform our own trilinear filtering, since the hardware filtering from Vulkan seems to have stepping between LOD levels for some reason
    float roughnessLOD = roughness * float(envInfo.ggxMipLevels);
    float baseRoughnessLOD = floor(roughnessLOD);
    float roughnessT = roughnessLOD - baseRoughnessLOD;

    // Calculate the specular and diffuse radiances
    vec3 mirrorDir = reflect(frag.viewDir, normal);
    vec3 diffuseLookupDir = (envInfo.transform * vec4(normal, 0.0)).xyz;
    vec3 specularLookupDir = (envInfo.transform * vec4(mirrorDir, 0.0)).xyz;

    vec4 diffuse = albedo * texture(lambertianCubemap, diffuseLookupDir);
    vec4 specular = textureLod(ggxCubemap, specularLookupDir, baseRoughnessLOD) * (1.0 - roughnessT) + textureLod(ggxCubemap, specularLookupDir, baseRoughnessLOD + 1.0) * roughnessT;

    // Calculate the dielectric and metal radiances using Schlicks'
    // We use a F0 = 0.04 for our dielectric material to represent an IOR of 1.5, which is a good representative choice
    float schlick = pow(1 - cosThetaV, 5.0);
    vec4 dielectric = mix(diffuse, specular * (brdf.x + brdf.y), 0.04 + 0.96 * schlick);
    vec4 metal = specular * (albedo * brdf.x + brdf.y);

    outColor = tonemap(frag.color * mix(dielectric, metal, metalness), camera.exposure);
}
