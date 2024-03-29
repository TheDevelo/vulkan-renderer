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

layout(std430, set = 3, binding = 0) buffer LightInfoSSBO {
    LightInfo lights[];
};

layout(location = 0) in VertexOutput frag;

layout(location = 0) out vec4 outColor;

vec4 lightContribution(LightInfo light, vec3 normal, vec4 position) {
    vec3 lightspaceNormal = normalize(mat3(transpose(inverse(light.transform))) * normal);
    vec3 lightspacePosition = (light.transform * position).xyz;

    if (light.type == 0) {
        // Sun Light
        // The light direction is (0, 0, -1)
        float cosWeight = max(0.0, dot(lightspaceNormal, vec3(0.0, 0.0, 1.0)));
        return vec4(light.tint, 1.0) * light.power * cosWeight;
    }
    else if (light.type == 1) {
        // Sphere Light
        // The light direction is normalize(lightspacePosition)
        float cosWeight = max(0.0, dot(lightspaceNormal, -normalize(lightspacePosition)));
        float falloff = 1.0 / max(light.radius, dot(lightspacePosition, lightspacePosition));
        vec4 lightContrib = vec4(light.tint, 1.0) * light.power * falloff * cosWeight / (4 * PI);

        if (light.useLimit) {
            float attenuation = max(0, 1 - pow(length(lightspacePosition) / light.limit, 4.0));
            lightContrib *= attenuation;
        }
        return lightContrib;
    }
    else if (light.type == 2) {
        // Spot Light
        // The light direction is normalize(lightspacePosition), but the spot direction is (0, 0, -1)
        // We calculate the light the same way as with the sphere light, but then attenuate by how close the light direction is to the spot direction
        float cosWeight = max(0.0, dot(lightspaceNormal, -normalize(lightspacePosition)));
        float falloff = 1.0 / max(light.radius, dot(lightspacePosition, lightspacePosition));
        vec4 lightContrib = vec4(light.tint, 1.0) * light.power * falloff * cosWeight / (4 * PI);

        // Attenuate by the spot direction
        float halfFov = light.fov / 2.0;
        float spotAngle = acos(dot(normalize(lightspacePosition), vec3(0.0, 0.0, -1.0)));
        float spotAttenuation = clamp((spotAngle - halfFov) / (halfFov * (1 - light.blend) - halfFov), 0.0, 1.0);
        lightContrib *= spotAttenuation;

        if (light.useLimit) {
            float attenuation = max(0, 1 - pow(length(lightspacePosition) / light.limit, 4.0));
            lightContrib *= attenuation;
        }
        return lightContrib;
    }
    else {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
}

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

    // Sample the light contribution from the environment map
    vec4 totalLight = texture(lambertianCubemap, envLookupDir);

    // Compute the light contribution from each analytic light
    for (int i = 0; i < lights.length(); i++) {
        totalLight += lightContribution(lights[i], normal, frag.worldPos);
    }

    outColor = tonemap(frag.color * diffuseColor * totalLight, camera.exposure);
}
