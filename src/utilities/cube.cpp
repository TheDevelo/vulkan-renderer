#include "stb_image.h"
#include "stb_image_write.h"

#include <charconv>
#include <filesystem>
#include <iostream>
#include <optional>

#include "linear.hpp"
#include "util.hpp"

struct Cubemap {
    std::vector<float> data;
    uint32_t width;
    uint32_t height;
};

struct CubePixel {
    uint32_t face;
    uint32_t x;
    uint32_t y;
};

// NOTE: Face order goes +X,-X,+Y,-Y,+Z,-Z just as in Vulkan.
// https://registry.khronos.org/vulkan/specs/1.3/html/chap16.html#_cube_map_face_selection
// We also use the center of each pixel, so s_face is (x + 0.5) / width, and same for t_face.
Vec3<float> getNormalFrom(CubePixel pixel, uint32_t width, uint32_t height) {
    float s = ((static_cast<float>(pixel.x) + 0.5) / static_cast<float>(width)) * 2.0 - 1.0;
    float t = ((static_cast<float>(pixel.y) + 0.5) / static_cast<float>(height)) * 2.0 - 1.0;
    switch (pixel.face) {
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

void normalToCube(Vec3<float> const& normal, uint32_t width, uint32_t height, CubePixel& pixel) {
    float s;
    float t;
    if (abs(normal.x) >= abs(normal.y) && abs(normal.x) >= abs(normal.z)) {
        if (normal.x >= 0.0) {
            pixel.face = 0;
            s = -normal.z / normal.x;
            t = -normal.y / normal.x;
        }
        else {
            pixel.face = 1;
            s = normal.z / -normal.x;
            t = -normal.y / -normal.x;
        }
    }
    else if (abs(normal.y) >= abs(normal.z)) {
        if (normal.y >= 0.0) {
            pixel.face = 2;
            s = normal.x / normal.y;
            t = normal.z / normal.y;
        }
        else {
            pixel.face = 3;
            s = normal.x / -normal.y;
            t = -normal.z / -normal.y;
        }
    }
    else {
        if (normal.z >= 0.0) {
            pixel.face = 4;
            s = normal.x / normal.z;
            t = -normal.y / normal.z;
        }
        else {
            pixel.face = 5;
            s = -normal.x / -normal.z;
            t = -normal.y / -normal.z;
        }
    }

    s = s * 0.5 + 0.5;
    t = t * 0.5 + 0.5;
    pixel.x = static_cast<uint32_t>(s * width);
    pixel.y = static_cast<uint32_t>(t * height);
}

void integrateLambertian(std::filesystem::path filePath, Cubemap const& inputEnv, uint32_t outputSize) {
    Cubemap lambertianEnv {
        .data = std::vector<float>(outputSize * outputSize * 6 * 4),
        .width = outputSize,
        .height = outputSize,
    };

    // Iterate over each pixel in the output cubemap
    for (uint32_t face = 0; face < 6; face++) {
        for (uint32_t y = 0; y < lambertianEnv.height; y++) {
            for (uint32_t x = 0; x < lambertianEnv.width; x++) {
                uint32_t i = ((face * lambertianEnv.height + y) * lambertianEnv.width + x) * 4;
                CubePixel pixel {
                    .face = face,
                    .x = x,
                    .y = y,
                };
                Vec3<float> normal = linear::normalize(getNormalFrom(pixel, lambertianEnv.width, lambertianEnv.height));
                Vec3<double> integral = Vec3<double>(0.0);
                printf("%u\n", i / 4);

                // Integrate over all the input cubemap pixels
                for (uint32_t face2 = 0; face2 < 6; face2++) {
                    for (uint32_t y2 = 0; y2 < inputEnv.height; y2++) {
                        for (uint32_t x2 = 0; x2 < inputEnv.width; x2++) {
                            uint32_t i2 = ((face2 * inputEnv.height + y2) * inputEnv.height + x2) * 4;
                            CubePixel pixel2 {
                                .face = face2,
                                .x = x2,
                                .y = y2,
                            };
                            Vec3<float> normal2 = getNormalFrom(pixel2, inputEnv.width, inputEnv.height);

                            float jacobian = linear::dot(normal2, normal);
                            if (jacobian > 0.0) {
                                float length2 = linear::length2(normal);
                                jacobian /= (length2 * length2);
                                integral.x += inputEnv.data[i2] * jacobian;
                                integral.y += inputEnv.data[i2 + 1] * jacobian;
                                integral.z += inputEnv.data[i2 + 2] * jacobian;
                            }
                        }
                    }
                }

                integral = integral * Vec3<double>(4.0 * M_1_PI / (static_cast<double>(inputEnv.width) * static_cast<double>(inputEnv.height)));
                lambertianEnv.data[i] = integral.x;
                lambertianEnv.data[i + 1] = integral.y;
                lambertianEnv.data[i + 2] = integral.z;
            }
        }
    }

    // Convert our lambertian cubemap to RGBE format
    std::vector<uint8_t> cubemapOut(lambertianEnv.width * lambertianEnv.height * 6 * 4);
    convertRGBtoRGBE(lambertianEnv.data.data(), cubemapOut.data(), lambertianEnv.width * lambertianEnv.height * 6);

    // Write to the output file
    filePath.replace_extension(".lambertian.png");
    stbi_write_png(filePath.c_str(), lambertianEnv.width, lambertianEnv.height * 6, 4, cubemapOut.data(), lambertianEnv.width * 4);
}

int main(int argc, char** argv) {
    // Parse command line options
    bool doIntegrateLambertian = false;
    bool doIntegrateGGX = false;
    uint32_t cubemapSize = 32;
    std::optional<std::string> inputFile;

    int currentIndex = 1;
    const std::vector<std::string_view> args(argv + 0, argv + argc);
    while (currentIndex < argc) {
        std::string_view currentArg = args[currentIndex];
        if (currentArg == "--lambertian") {
            doIntegrateLambertian = true;
        }
        else if (currentArg == "--ggx") {
            doIntegrateGGX = true;
        }
        else if (currentArg == "--cubemap-size") {
            currentIndex += 1;
            if (currentIndex >= argc) {
                PANIC("missing argument to --cubemap-size");
            }

            std::string_view sizeStr = args[currentIndex];
            auto sizeResult = std::from_chars(sizeStr.data(), sizeStr.data() + sizeStr.size(), cubemapSize);
            if (sizeResult.ec == std::errc::invalid_argument || sizeResult.ec == std::errc::result_out_of_range) {
                PANIC("invalid argument to --cubemap-size");
            }
        }
        else if (!inputFile.has_value()) {
            inputFile = currentArg;
        }
        else {
            PANIC("invalid command line argument: " + std::string(currentArg));
        }

        currentIndex += 1;
    }

    if (!inputFile.has_value()) {
        PANIC("command line arguments do not have an input file listed");
    }
    std::filesystem::path inputPath = inputFile.value();

    // Load the input cubemap image
    int textureWidth, textureHeight, textureChannels;
    stbi_uc* pixels = stbi_load(inputPath.c_str(), &textureWidth, &textureHeight, &textureChannels, STBI_rgb_alpha);
    if (!pixels) {
        PANIC("failed to load texture image!");
    }
    textureHeight /= 6;

    // Convert image from RGBE format
    Cubemap inputEnv {
        .data = std::vector<float>(textureWidth * textureHeight * 6 * 4),
        .width = static_cast<uint32_t>(textureWidth),
        .height = static_cast<uint32_t>(textureHeight),
    };
    convertRGBEtoRGB(pixels, inputEnv.data.data(), inputEnv.width * inputEnv.height * 6);

    // Free our original image buffer
    stbi_image_free(pixels);

    if (doIntegrateLambertian) {
        integrateLambertian(inputPath, inputEnv, cubemapSize);
    }
    if (doIntegrateGGX) {
        // TODO: GGX integration
    }

    return 0;
}
