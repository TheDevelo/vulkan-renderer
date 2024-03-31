#define PI 3.14159265358979323846264338327950288

struct CameraInfo {
    mat4 view;
    mat4 proj;
    vec4 position;
    float exposure;
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

struct EnvironmentInfo {
    mat4 transform;
    uint ggxMipLevels;
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

    // Padding to align to 256 bytes
    float padding[20];
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
vec4 tonemap(vec4 linearColor, float exposure) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;

    linearColor.rgb *= exposure; // Adjust the input color by the exposure level
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

vec4 diffuseLightContribution(LightInfo light, vec3 normal, vec4 position) {
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
