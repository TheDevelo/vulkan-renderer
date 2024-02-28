#include "stb_image.h"
#include "stb_image_write.h"

#include "linear.hpp"
#include "util.hpp"

// NOTE: Face order goes +X,-X,+Y,-Y,+Z,-Z just as in Vulkan.
// https://registry.khronos.org/vulkan/specs/1.3/html/chap16.html#_cube_map_face_selection
// We also use the center of each pixel, so s_face is (x + 0.5) / width, and same for t_face.
Vec3<float> getNormalFrom(uint32_t face, uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
    float s = ((static_cast<float>(x) + 0.5) / static_cast<float>(width)) * 2.0 - 1.0;
    float t = ((static_cast<float>(y) + 0.5) / static_cast<float>(height)) * 2.0 - 1.0;
    switch (face) {
        case 0:
            return Vec3<float>(1.0, -t, -s);
        case 1:
            return Vec3<float>(-1.0, -t, s);
        case 2:
            return Vec3<float>(s, 1.0, t);
        case 3:
            return Vec3<float>(s, -1.0, -t);
        case 4:
            return Vec3<float>(s, -t, 1.0);
        case 5:
            return Vec3<float>(-s, -t, -1.0);
        default:
            PANIC("invalid face specified in cubemap");
    }
}

int main(int argc, char** argv) {
    const uint32_t lambertianSize = 8;
    // Load the cubemap into a float buffer
    int textureWidth, textureHeight, textureChannels;
    stbi_uc* pixels = stbi_load("cube.png", &textureWidth, &textureHeight, &textureChannels, STBI_rgb_alpha);
    if (!pixels) {
        PANIC("failed to load texture image!");
    }
    textureHeight /= 6;

    // Convert image from RGBE format
    std::vector<float> inputEnv;
    inputEnv.resize(textureWidth * textureHeight * 6 * 4);
    convertRGBEtoRGB(pixels, inputEnv.data(), textureWidth * textureHeight * 6);

    // Free our original image buffer
    stbi_image_free(pixels);

    // Lambertian Cube
    std::vector<float> lambertianCube(lambertianSize * lambertianSize * 6 * 4);
    for (uint32_t face = 0; face < 6; face++) {
        for (uint32_t y = 0; y < lambertianSize; y++) {
            for (uint32_t x = 0; x < lambertianSize; x++) {
                uint32_t i = ((face * lambertianSize + y) * lambertianSize + x) * 4;
                Vec3<float> normal = linear::normalize(getNormalFrom(face, x, y, lambertianSize, lambertianSize));
                Vec3<double> integral = Vec3<double>(0.0);
                printf("%u\n", i / 4);
                for (uint32_t face2 = 0; face2 < 6; face2++) {
                    for (uint32_t y2 = 0; y2 < static_cast<uint32_t>(textureHeight); y2++) {
                        for (uint32_t x2 = 0; x2 < static_cast<uint32_t>(textureWidth); x2++) {
                            uint32_t i2 = ((face2 * textureHeight + y2) * textureWidth + x2) * 4;
                            Vec3<float> normal2 = getNormalFrom(face2, x2, y2, textureWidth, textureHeight);
                            float jacobian = linear::dot(normal2, normal);
                            if (jacobian > 0.0) {
                                float length2 = linear::length2(normal);
                                jacobian /= (length2 * length2);
                                integral.x += inputEnv[i2] * jacobian;
                                integral.y += inputEnv[i2 + 1] * jacobian;
                                integral.z += inputEnv[i2 + 2] * jacobian;
                            }
                        }
                    }
                }
                integral = integral * Vec3<double>(4.0 * M_1_PI / (static_cast<double>(textureWidth) * static_cast<double>(textureHeight)));
                lambertianCube[i] = integral.x;
                lambertianCube[i + 1] = integral.y;
                lambertianCube[i + 2] = integral.z;
            }
        }
    }

    // Convert to RGBE format
    std::vector<uint8_t> lambertianCubeOut(lambertianSize * lambertianSize * 6 * 4);
    convertRGBtoRGBE(lambertianCube.data(), lambertianCubeOut.data(), lambertianSize * lambertianSize * 6);

    // Write out
    stbi_write_png("cube-out.png", lambertianSize, lambertianSize * 6, 4, lambertianCubeOut.data(), lambertianSize * 4);

    return 0;
}
