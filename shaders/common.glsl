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
};

struct VertexOutput {
    vec4 color;
    vec3 normal;
    vec3 tangent;
    vec3 bitangent;
    vec2 uv;
    vec3 viewDir;
    vec3 tangentViewDir;
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
