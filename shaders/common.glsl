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
