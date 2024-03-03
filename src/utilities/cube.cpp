#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <barrier>
#include <charconv>
#include <execution>
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

    // Generate a list of pixels in our lambertian cubemap so we can parallel for_each over them
    std::vector<CubePixel> pixels(lambertianEnv.width * lambertianEnv.height * 6);
    for (uint32_t face = 0; face < 6; face++) {
        for (uint32_t y = 0; y < lambertianEnv.height; y++) {
            for (uint32_t x = 0; x < lambertianEnv.width; x++) {
                uint32_t i = (face * lambertianEnv.height + y) * lambertianEnv.width + x;
                pixels[i].face = face;
                pixels[i].x = x;
                pixels[i].y = y;
            }
        }
    }

    // Setup a progress bar for the execution for loop
    auto progressBarUpdate = [&]() noexcept {
        static uint32_t totalPercent = 0;
        static uint32_t pixelsCompleted = 0;
        pixelsCompleted += 1;
        if (pixelsCompleted * 100 >= (totalPercent + 1) * lambertianEnv.width * lambertianEnv.height * 6) {
            totalPercent += 1;

            if (totalPercent == 100) {
                std::cout << "#] Complete!" << std::endl;
            }
            else if (totalPercent % 10 == 0) {
                std::cout << "#" << std::flush;
            }
            else {
                std::cout << "=" << std::flush;
            }
        }
    };
    std::barrier progressBarBarrier(1, progressBarUpdate);
    std::cout << "Integrating Lambertian: [" << std::flush;

    // Iterate over each pixel in the lambertian cubemap in parallel
    std::for_each(std::execution::par, pixels.begin(), pixels.end(), [&](CubePixel& pixel) {
        uint32_t lambertianIndex = ((pixel.face * lambertianEnv.height + pixel.y) * lambertianEnv.width + pixel.x) * 4;
        Vec3<float> lambertianNormal = linear::normalize(getNormalFrom(pixel, lambertianEnv.width, lambertianEnv.height));
        Vec3<double> integral = Vec3<double>(0.0);

        // Integrate over all the input cubemap pixels
        for (uint32_t face = 0; face < 6; face++) {
            for (uint32_t y = 0; y < inputEnv.height; y++) {
                for (uint32_t x = 0; x < inputEnv.width; x++) {
                    uint32_t environmentIndex = ((face * inputEnv.height + y) * inputEnv.height + x) * 4;
                    CubePixel environmentPixel {
                        .face = face,
                        .x = x,
                        .y = y,
                    };
                    Vec3<float> environmentNormal = getNormalFrom(environmentPixel, inputEnv.width, inputEnv.height);

                    // Jacobian of our integral is max(0, |w * n|) / ||w||^4. We take the max to exclude points on the lower hemisphere.
                    // To not waste work on computing the length for points that don't contribute, we check just the dot product.
                    float jacobian = linear::dot(lambertianNormal, environmentNormal);
                    if (jacobian > 0.0) {
                        float length2 = linear::length2(environmentNormal);
                        jacobian /= (length2 * length2);
                        integral.x += inputEnv.data[environmentIndex] * jacobian;
                        integral.y += inputEnv.data[environmentIndex + 1] * jacobian;
                        integral.z += inputEnv.data[environmentIndex + 2] * jacobian;
                    }
                }
            }
        }

        integral = integral * Vec3<double>(4.0 * M_1_PI / (static_cast<double>(inputEnv.width) * static_cast<double>(inputEnv.height)));
        lambertianEnv.data[lambertianIndex] = integral.x;
        lambertianEnv.data[lambertianIndex + 1] = integral.y;
        lambertianEnv.data[lambertianIndex + 2] = integral.z;

        progressBarBarrier.arrive_and_wait();
    });

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
