#define PI 3.14159265358979323846264338327950288
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

struct CameraInfo {
    mat4 view;
    mat4 proj;
    vec4 position;
    float exposure;
    bool tonemap;
};

struct MaterialConstants {
    vec3 albedo;
    float roughness;
    float metalness;
    bool useNormalMap;
    bool useDisplacementMap;
    bool useAlbedoMap;
    bool useRoughnessMap;
    bool useMetalnessMap;
};

struct AABB {
    vec3 minCorner;
    vec3 maxCorner;
};

struct EnvironmentInfo {
    mat4 transform;
    AABB localBBox;
    uint ggxMipLevels;
    bool isLocal;
    bool isEmpty;
};

struct LightInfo {
    mat4 transform;
    mat4 projection;
    vec3 tint;
    uint type; // 0 = Sun, 1 = Sphere, 2 = Spot, UINT_MAX = Disabled

    // Sun/Sphere/Spot Info
    float power; // Strength for Sun

    // Sun Info
    float angle;

    // Sphere/Spot Info
    float radius;
    float limit;
    bool useLimit;

    // Spot Info
    float fov;
    float blend;

    // Shadow Map Info if we have any
    bool useShadowMap;
    uint shadowMapIndex;

    // Padding to align to 256 bytes
    float padding[18];
};

struct VertexOutput {
    vec4 color;
    vec3 normal;
    vec3 tangent;
    vec3 bitangent;
    vec2 uv;
    vec3 viewDir;
    vec3 tangentViewDir;
    vec4 worldPos;
};

// Tonemapping operator is (an approximation of) the ACES Filmic curve
// Formula from https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec4 tonemap(vec4 linearColor, CameraInfo camera) {
    if (!camera.tonemap) {
        return linearColor;
    }

    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;

    linearColor.rgb *= camera.exposure; // Adjust the input color by the exposure level
    vec4 tonemappedColor = clamp((linearColor * (a * linearColor + b)) / (linearColor * (c * linearColor + d) + e), 0.0, 1.0);
    tonemappedColor.a = linearColor.a;

    return tonemappedColor;
}

// Gets displacement-adjusted UV coordinates using Parallax Occlusion Mapping
// https://learnopengl.com/Advanced-Lighting/Parallax-Mapping - Good resource covering POM
vec2 getAdjustedUVs(VertexOutput frag, MaterialConstants materialConstants, sampler2D displacementMap) {
    if (materialConstants.useDisplacementMap) {
        const float POMLayers = 4.0;
        const float layerStep = 1.0 / POMLayers;
        const float heightScale = 0.1;
        // deltaUV is change in UV taking a unit step in the normal direction
        vec2 deltaUV = frag.tangentViewDir.xy * layerStep * heightScale / frag.tangentViewDir.z;

        // Step through the layers until we "hit" the depth map
        float currentLayerDepth = 0.0;
        vec2 currentUV = frag.uv;
        float currentDepthValue = texture(displacementMap, currentUV).r;
        while (currentLayerDepth < currentDepthValue) {
            currentUV -= deltaUV;
            currentDepthValue = texture(displacementMap, currentUV).r;
            currentLayerDepth += layerStep;
        }

        // Use the previous layer's step to calculate a slope, and calculate the intersection with that slope
        vec2 prevUV = currentUV + deltaUV;
        float curDistToDisplacement = currentDepthValue - currentLayerDepth;
        float prevDistToDisplacement = texture(displacementMap, prevUV).r - (currentLayerDepth - layerStep);
        float intersectionT = curDistToDisplacement / (curDistToDisplacement - prevDistToDisplacement);

        return currentUV * (1.0 - intersectionT) + prevUV * intersectionT;
    }
    else {
        return frag.uv;
    }
}

vec3 getNormal(VertexOutput frag, MaterialConstants materialConstants, sampler2D normalMap, vec2 uv) {
    if (materialConstants.useNormalMap) {
        vec3 TBN = texture(normalMap, uv).xyz * 2.0 - 1.0;
        return TBN.x * normalize(frag.tangent) + TBN.y * normalize(frag.bitangent) + TBN.z * normalize(frag.normal);
    }
    else {
        return normalize(frag.normal);
    }
}

// Parallax correct the environment map lookup direction using the environment's bounding box
// This only applies for specular IBL, since the lambertian cubemap does not make sense for parallax correction
// NOTE: this also handles the mirror plane case, but for that we pre-distort the cubemap, so the lookup direction stays the same
vec3 parallaxEnvDir(EnvironmentInfo env, vec4 worldPosition, vec3 worldDirection) {
    vec3 direction = (env.transform * vec4(worldDirection, 0.0)).xyz;
    vec3 position = (env.transform * worldPosition).xyz;
    if (!env.isLocal) {
        // Global environment maps should not be parallax corrected.
        return direction;
    }

    // Determine the first point of intersection on the cube's bounding box
    // Note that we only need to check 3 planes as the other 3 are opposite of where direction points (assuming position is in the box)
    vec3 bboxPlanes = env.localBBox.minCorner;
    if (direction.x > 0) {
        bboxPlanes.x = env.localBBox.maxCorner.x;
    }
    if (direction.y > 0) {
        bboxPlanes.y = env.localBBox.maxCorner.y;
    }
    if (direction.z > 0) {
        bboxPlanes.z = env.localBBox.maxCorner.z;
    }

    float xIntersectionT = (bboxPlanes.x - position.x) / direction.x;
    float yIntersectionT = (bboxPlanes.y - position.y) / direction.y;
    float zIntersectionT = (bboxPlanes.z - position.z) / direction.z;

    // Get the intersection point using the minimum intersection time
    vec3 hitPosition = min(min(xIntersectionT, yIntersectionT), zIntersectionT) * direction + position;
    return normalize(hitPosition);
}

// GGX BRDF derived from Epic's 2013 SIGGRAPH PBR course notes
// Assume view is coming into our point instead of heading out
// NOTE: we return a vec3 since the BRDF varies per-channel for metals due to Fresnel
vec3 ggxBRDF(vec3 normal, vec3 view, vec3 light, float roughness, vec3 f0) {
    vec3 halfV = normalize(light - view);
    float alphaSquared = roughness * roughness * roughness * roughness;

    float NoH = dot(normal, halfV);
    float NoV = dot(normal, -view);
    float NoL = dot(normal, light);
    float VoH = dot(-view, halfV);

    // Calculate D(h) (the normal distribution function)
    float denom = (NoH * NoH * (alphaSquared - 1.0) + 1.0);
    float D = alphaSquared / (PI * denom * denom);

    // Calculate G(l, v, h) (the shadowing/masking term)
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float G = NoL * NoV / ((NoL * (1.0 - k) + k) * (NoV * (1.0 - k) + k));

    // Calculate F(v, h) (the fresnel term)
    vec3 F = f0 + (1.0 - f0) * pow(1.0 - VoH, 5.0);

    // Calculate the final BRDF
    return D * F * G / (4.0 * NoL * NoV);
}

// Calculates whether a pixel is in shadow from the shadow map
float getShadowValue(LightInfo light, sampler2DShadow shadowMap, vec4 position) {
    if (!light.useShadowMap) {
        return 1.0;
    }

    // Calculate the position of the pixel in the shadow map
    vec4 shadowMapPosition = light.projection * light.transform * position;

    // Do the perspective divide and shift the range from [-1,1] to [0,1]
    shadowMapPosition.xyz /= shadowMapPosition.w;
    shadowMapPosition.xy *= 0.5;
    shadowMapPosition.xy += 0.5;

    // Perform PCF with a 5x5 kernel to create a smooth shadow edge
    vec2 samplingOffset = 1.0 / vec2(textureSize(shadowMap, 0));
    float shadow = 0.0;
    for (int xOff = -2; xOff <= 2; xOff++) {
        for (int yOff = -2; yOff <= 2; yOff++) {
            vec3 offsetPosition = shadowMapPosition.xyz;
            offsetPosition.xy += vec2(xOff, yOff) * samplingOffset;
            shadow += texture(shadowMap, offsetPosition) / 25.0;
        }
    }

    // Rescale and threshold the shadow value to help prevent light peek near the edges of shadow-casting objects
    // Essentially treats shadow values < 0.5 as in full shadow.
    shadow = clamp(2.0 * shadow - 1.0, 0.0, 1.0);

    return shadow;
}

vec4 diffuseLightContribution(LightInfo light, sampler2DShadow shadowMap, vec3 normal, vec4 position) {
    vec3 lightspaceNormal = normalize(mat3(transpose(inverse(light.transform))) * normal);
    vec3 lightspacePosition = (light.transform * position).xyz;

    if (light.type == 0) {
        // Sun Light
        // The light direction is (0, 0, -1)
        float cosWeight = max(0.0, dot(lightspaceNormal, vec3(0.0, 0.0, 1.0)));
        return vec4(light.tint, 1.0) * light.power * cosWeight / PI;
    }
    else if (light.type == 1) {
        // Sphere Light
        // The light direction is normalize(lightspacePosition)
        float cosWeight = max(0.0, dot(lightspaceNormal, -normalize(lightspacePosition)));
        float falloff = 1.0 / max(light.radius, dot(lightspacePosition, lightspacePosition));
        vec4 lightContrib = vec4(light.tint, 1.0) * light.power * falloff * cosWeight / (4 * PI * PI);

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
        vec4 lightContrib = vec4(light.tint, 1.0) * light.power * falloff * cosWeight / (4 * PI * PI);

        // Attenuate by the spot direction
        float halfFov = light.fov / 2.0;
        float spotAngle = acos(dot(normalize(lightspacePosition), vec3(0.0, 0.0, -1.0)));
        float spotAttenuation = clamp((spotAngle - halfFov) / (halfFov * (1 - light.blend) - halfFov), 0.0, 1.0);
        lightContrib *= spotAttenuation;

        if (light.useLimit) {
            float attenuation = max(0, 1 - pow(length(lightspacePosition) / light.limit, 4.0));
            lightContrib *= attenuation;
        }

        // Return light attenuated by shadow value
        return lightContrib * getShadowValue(light, shadowMap, position);
    }
    else {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
}

vec4 specularLightContribution(LightInfo light, sampler2DShadow shadowMap,
                               float roughness, vec3 f0,
                               vec3 normal, vec3 view, vec4 position) {
    vec3 lightspaceNormal = normalize(mat3(transpose(inverse(light.transform))) * normal);
    vec3 lightspaceView = normalize(mat3(light.transform) * view);
    vec3 lightspaceMirror = reflect(lightspaceView, lightspaceNormal);
    vec3 lightspacePosition = (light.transform * position).xyz;
    float alpha = roughness * roughness;

    if (light.type == 0) {
        // Sun Light
        // First, find the representative light direction (from directions within angle/2 of -z) closest to mirror
        vec2 mirrorOrthogonal = normalize(lightspaceMirror.xy);
        float representativeAngle = min(light.angle / 2, acos(lightspaceMirror.z));
        vec3 lightDir = vec3(mirrorOrthogonal * sin(representativeAngle), cos(representativeAngle));

        // Now calculate the light contribution as BRDF * power * cosWeight
        float cosWeight = max(0.0, dot(lightspaceNormal, lightDir));
        vec4 brdf = vec4(ggxBRDF(lightspaceNormal, lightspaceView, lightDir, roughness, f0), 1.0);
        vec4 lightContrib = vec4(light.tint, 1.0) * light.power * cosWeight * brdf;

        // Lastly, we want to normalize the light contribution due to our representative point operation widening the BRDF (as outlined in Epic's course notes)
        // We multiply by (alpha / alpha')^2, where alpha' = clamp(alpha + solidAngle, 0, 1). solidAngle of a spherical sector of angle a is 2pi(1-cos a/2).
        float alphaPrime = clamp(alpha + 2 * PI * (1 - cos(light.angle / 2)), 0.0, 1.0);
        float normalizationFactor = alpha * alpha / (alphaPrime * alphaPrime);
        lightContrib *= normalizationFactor;

        return lightContrib;
    }
    else if (light.type == 1) {
        // Sphere Light
        // First, find the representative light direction on the sphere closest to mirror (as per Epic)
        vec3 closestRayDir = lightspacePosition - dot(lightspacePosition, lightspaceMirror) * lightspaceMirror;
        vec3 closestPointDir = closestRayDir * clamp(light.radius / length(closestRayDir), 0.0, 1.0) - lightspacePosition;
        vec3 lightDir = normalize(closestPointDir);

        // Now calculate the light contribution as BRDF * (power * falloff / 4 PI) * cosWeight
        float cosWeight = max(0.0, dot(lightspaceNormal, lightDir));
        float falloff = 1.0 / max(light.radius, dot(closestPointDir, closestPointDir));
        vec4 brdf = vec4(ggxBRDF(lightspaceNormal, lightspaceView, lightDir, roughness, f0), 1.0);
        vec4 lightContrib = vec4(light.tint, 1.0) * light.power * falloff * cosWeight * brdf / (4 * PI);

        if (light.useLimit) {
            float attenuation = max(0, 1 - pow(length(lightspacePosition) / light.limit, 4.0));
            lightContrib *= attenuation;
        }

        // Normalize based on (alpha / alpha')^2
        float alphaPrime = clamp(alpha + light.radius / (2 * length(closestPointDir)), 0.0, 1.0);
        float normalizationFactor = alpha * alpha / (alphaPrime * alphaPrime);
        lightContrib *= normalizationFactor;

        return lightContrib;
    }
    else if (light.type == 2) {
        // Spot Light
        // First, find the representative light direction on the sphere closest to mirror (as per Epic)
        vec3 closestRayDir = lightspacePosition - dot(lightspacePosition, lightspaceMirror) * lightspaceMirror;
        vec3 closestPointDir = closestRayDir * clamp(light.radius / length(closestRayDir), 0.0, 1.0) - lightspacePosition;
        vec3 lightDir = normalize(closestPointDir);

        // Now calculate the light contribution as BRDF * (power * falloff / 4 PI) * cosWeight
        float cosWeight = max(0.0, dot(lightspaceNormal, lightDir));
        float falloff = 1.0 / max(light.radius, dot(closestPointDir, closestPointDir));
        vec4 brdf = vec4(ggxBRDF(lightspaceNormal, lightspaceView, lightDir, roughness, f0), 1.0);
        vec4 lightContrib = vec4(light.tint, 1.0) * light.power * falloff * cosWeight * brdf / (4 * PI);

        // Attenuate by the spot direction
        float halfFov = light.fov / 2.0;
        float spotAngle = acos(dot(normalize(lightspacePosition), vec3(0.0, 0.0, -1.0)));
        float spotAttenuation = clamp((spotAngle - halfFov) / (halfFov * (1 - light.blend) - halfFov), 0.0, 1.0);
        lightContrib *= spotAttenuation;

        if (light.useLimit) {
            float attenuation = max(0, 1 - pow(length(lightspacePosition) / light.limit, 4.0));
            lightContrib *= attenuation;
        }

        // Normalize based on (alpha / alpha')^2
        float alphaPrime = clamp(alpha + light.radius / (2 * length(closestPointDir)), 0.0, 1.0);
        float normalizationFactor = alpha * alpha / (alphaPrime * alphaPrime);
        lightContrib *= normalizationFactor;

        // Return light attenuated by shadow value
        return lightContrib * getShadowValue(light, shadowMap, position);
    }
    else {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
}
