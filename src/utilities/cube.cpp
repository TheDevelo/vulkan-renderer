#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <charconv>
#include <execution>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>

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
    pixel.x = std::min(static_cast<uint32_t>(s * width), width - 1);
    pixel.y = std::min(static_cast<uint32_t>(t * height), height - 1);
}

// Function to take the radical inverse of a function, with the digits scrambled according to perm
// Adapted from PBRT: https://pbr-book.org/3ed-2018/Sampling_and_Reconstruction/The_Halton_Sampler
float scrambledRadicalInverse(const std::vector<uint64_t> &perm, uint64_t a, uint64_t base) {
    const float invBase = 1.0 / base;
    uint64_t reversedDigits = 0;
    float invBaseN = 1.0;
    while (a) {
        uint64_t next  = a / base;
        uint64_t digit = a - next * base;
        reversedDigits = reversedDigits * base + perm[digit];
        invBaseN *= invBase;
        a = next;
    }
    return std::min((reversedDigits + invBase * perm[0] / (1 - invBase)) * invBaseN, 0.999999f);
}

// Generate the ith sample from the Halton sequence
// Since we only need 1 2D sample per iteration for our Monte Carlo purposes, just radical inverses with bases 2 and 3
inline Vec2<float> halton(uint64_t i, const std::vector<uint64_t>& perm2, const std::vector<uint64_t>& perm3) {
    return Vec2<float>(scrambledRadicalInverse(perm2, i, 2), scrambledRadicalInverse(perm3, i, 3));
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
    std::mutex progressBarMutex;
    uint32_t totalPercent = 0;
    uint32_t pixelsCompleted = 0;
    auto updateProgressBar = [&]() {
        pixelsCompleted += 1;
        while (pixelsCompleted * 100 >= (totalPercent + 1) * pixels.size()) {
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

        std::lock_guard<std::mutex> progressBarGuard(progressBarMutex);
        updateProgressBar();
    });

    // Convert our lambertian cubemap to RGBE format
    std::vector<uint8_t> cubemapOut(lambertianEnv.width * lambertianEnv.height * 6 * 4);
    convertRGBtoRGBE(lambertianEnv.data.data(), cubemapOut.data(), lambertianEnv.width * lambertianEnv.height * 6);

    // Write to the output file
    filePath.replace_extension(".lambertian.png");
    stbi_write_png(filePath.c_str(), lambertianEnv.width, lambertianEnv.height * 6, 4, cubemapOut.data(), lambertianEnv.width * 4);
}

// NOTE: minOutputSize is the minimum size a GGX cubemap can be. Since we are doing mip-maps, the last mip-level might be bigger than minOutputSize.
void integrateGGX(std::filesystem::path filePath, Cubemap const& inputEnv, uint32_t minOutputSize) {
    // Calculate the number of mipmap levels we want to generate. This includes a roughness 0 mipmap, which is just the input environment.
    uint32_t mipLevels = std::floor(std::log2(std::max(inputEnv.width, inputEnv.height)) - std::log2(minOutputSize));

    // Generate a GGX mip-map for each LOD level.
    uint32_t mipWidth = inputEnv.width / 2;
    uint32_t mipHeight = inputEnv.width / 2;
    for (uint32_t mipLevel = 1; mipLevel < mipLevels; mipLevel++) {
        Cubemap ggxEnv {
            .data = std::vector<float>(mipWidth * mipHeight * 6 * 4),
            .width = mipWidth,
            .height = mipHeight,
        };

        // Calculate the roughness value that our mip Level corresponds to
        // Level 0 should be 0 roughness, while Level (mipLevels - 1) should be 1 roughness
        float roughness = static_cast<float>(mipLevel) / (mipLevels - 1);
        float alpha = roughness * roughness;

        // Generate a list of pixels in our GGX cubemap so we can parallel for_each over them
        std::vector<CubePixel> pixels(ggxEnv.width * ggxEnv.height * 6);
        for (uint32_t face = 0; face < 6; face++) {
            for (uint32_t y = 0; y < ggxEnv.height; y++) {
                for (uint32_t x = 0; x < ggxEnv.width; x++) {
                    uint32_t i = (face * ggxEnv.height + y) * ggxEnv.width + x;
                    pixels[i].face = face;
                    pixels[i].x = x;
                    pixels[i].y = y;
                }
            }
        }

        // Setup a progress bar for the execution for loop
        std::mutex progressBarMutex;
        uint32_t totalPercent = 0;
        uint32_t pixelsCompleted = 0;
        auto updateProgressBar = [&]() {
            pixelsCompleted += 1;
            while (pixelsCompleted * 100 >= (totalPercent + 1) * pixels.size()) {
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
        std::cout << "Integrating GGX (Level " << mipLevel << " of " << mipLevels - 1 << "): [" << std::flush;

        // Generate permutations to use for our Halton sequence
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, 5);

        std::vector<uint64_t> perm2(2);
        perm2[0] = dist(gen) % 2;
        perm2[1] = 1 - perm2[0];

        std::vector<uint64_t> perm3(3);
        perm3[0] = dist(gen) % 3;
        perm3[1] = (perm3[0] + 1 + (dist(gen) % 2)) % 3;
        perm3[2] = 3 - perm3[0] - perm3[1];

        // Iterate over each pixel in the GGX cubemap in parallel
        // Inspired by the code samples from Epic's 2013 PBR course notes: https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
        std::for_each(std::execution::par, pixels.begin(), pixels.end(), [&](CubePixel& pixel) {
            uint32_t ggxIndex = ((pixel.face * ggxEnv.height + pixel.y) * ggxEnv.width + pixel.x) * 4;

            // Build a tangent space around the normal of this pixel
            Vec3<float> ggxNormal = linear::normalize(getNormalFrom(pixel, ggxEnv.width, ggxEnv.height));
            Vec3<float> tangentX;
            if (ggxNormal.z < 0.99) {
                tangentX = linear::normalize(linear::cross(Vec3<float>(0.0, 0.0, 1.0), ggxNormal));
            }
            else {
                tangentX = linear::normalize(linear::cross(Vec3<float>(1.0, 0.0, 0.0), ggxNormal));
            }
            Vec3<float> tangentY = linear::cross(ggxNormal, tangentX);

            // Calculate the left term of the PBR split sum using importance sampling
            const uint64_t numSamples = 1024;
            Vec3<double> splitSum = Vec3<double>(0.0);
            double splitSumWeight = 0.0;
            for (uint64_t sample = 0; sample < numSamples; sample++) {
                // Importance sample the GGX half-vector
                Vec2<float> randUV = halton(sample + numSamples * ggxIndex / 4, perm2, perm3);
                float phi = 2 * M_PI * randUV.x;
                float cosTheta = sqrt((1.0 - randUV.y) / (1.0 + (alpha * alpha - 1.0) * randUV.y));
                float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
                Vec3<float> halfVec = Vec3<float>(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

                // Transform the half vector to be around the GGX pixel normal instead of +Z
                halfVec = halfVec.x * tangentX + halfVec.y * tangentY + halfVec.z * ggxNormal;

                // Reflect the view direction around the half vector.
                // NOTE: We are approximating that the view direction is ggxNormal.
                // This is not true in reality, but provides the most accurate reflections when we are looking head on.
                Vec3<float> reflectionVec = 2.0f * linear::dot(ggxNormal, halfVec) * halfVec - ggxNormal;

                // Reject any reflection vectors that lie below the upper hemisphere around ggxNormal
                double cosReflection = linear::dot(reflectionVec, ggxNormal);
                if (cosReflection > 0.0) {
                    // Add the luminance of the reflection to our split sum
                    CubePixel environmentPixel;
                    normalToCube(reflectionVec, inputEnv.width, inputEnv.height, environmentPixel);
                    uint32_t environmentIndex = ((environmentPixel.face * inputEnv.height + environmentPixel.y) * inputEnv.width + environmentPixel.x) * 4;

                    splitSum.x += inputEnv.data[environmentIndex] * cosReflection;
                    splitSum.y += inputEnv.data[environmentIndex + 1] * cosReflection;
                    splitSum.z += inputEnv.data[environmentIndex + 2] * cosReflection;
                    splitSumWeight += cosReflection;
                }
            }
            splitSum = splitSum / Vec3<double>(splitSumWeight);

            ggxEnv.data[ggxIndex] = splitSum.x;
            ggxEnv.data[ggxIndex + 1] = splitSum.y;
            ggxEnv.data[ggxIndex + 2] = splitSum.z;

            std::lock_guard<std::mutex> progressBarGuard(progressBarMutex);
            updateProgressBar();
        });

        // Convert our lambertian cubemap to RGBE format
        std::vector<uint8_t> cubemapOut(ggxEnv.width * ggxEnv.height * 6 * 4);
        convertRGBtoRGBE(ggxEnv.data.data(), cubemapOut.data(), ggxEnv.width * ggxEnv.height * 6);

        // Write to the output file
        std::filesystem::path outputPath = filePath;
        outputPath.replace_extension(string_format(".ggx%u.png", mipLevel));
        stbi_write_png(outputPath.c_str(), ggxEnv.width, ggxEnv.height * 6, 4, cubemapOut.data(), ggxEnv.width * 4);

        mipWidth /= 2;
        mipHeight /= 2;
    }
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
        integrateGGX(inputPath, inputEnv, cubemapSize);
    }

    return 0;
}
