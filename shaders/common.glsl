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

struct VertexOutput {
    vec4 color;
    vec3 normal;
    vec3 tangent;
    vec3 bitangent;
    vec2 uv;
    vec4 worldPos;
};

// Tonemapping operator is (an approximation of) the ACES Filmic curve
// Formula from https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec4 tonemap(vec4 linearColor) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;

    vec4 tonemappedColor = clamp((linearColor * (a * linearColor + b)) / (linearColor * (c * linearColor + d) + e), 0.0, 1.0);
    tonemappedColor.a = linearColor.a;

    return tonemappedColor;
}

vec3 getNormal(VertexOutput frag, MaterialConstants materialConstants, sampler2D normalMap) {
    if (materialConstants.useNormalMap) {
        vec3 TBN = texture(normalMap, frag.uv).xyz * 2.0 - 1.0;
        return TBN.x * normalize(frag.tangent) + TBN.y * normalize(frag.bitangent) + TBN.z * normalize(frag.normal);
    }
    else {
        return normalize(frag.normal);
    }
}
