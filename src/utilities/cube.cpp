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
void integrateGGX(std::filesystem::path filePath, Cubemap const& inputEnv, uint32_t topWidth, uint32_t topHeight, uint32_t minOutputSize) {
    // Calculate the number of mipmap levels we want to generate. This includes a roughness 0 mipmap, which is just the input environment.
    uint32_t mipLevels = std::floor(std::log2(std::max(topWidth, topHeight)) - std::log2(minOutputSize));

    // Generate a GGX mip-map for each LOD level.
    uint32_t mipWidth = topWidth / 2;
    uint32_t mipHeight = topHeight / 2;
    for (uint32_t mipLevel = 1; mipLevel <= mipLevels; mipLevel++) {
        Cubemap ggxEnv {
            .data = std::vector<float>(mipWidth * mipHeight * 6 * 4),
            .width = mipWidth,
            .height = mipHeight,
        };

        // Calculate the roughness value that our mip Level corresponds to
        // Level 0 should be 0 roughness, while Level mipLevels should be 1 roughness
        float roughness = static_cast<float>(mipLevel) / mipLevels;
        float alphaSquared = roughness * roughness * roughness * roughness;

        // Calculate the integration cutoff for our GGX halfway vectors.
        // The cutoff is cos^2(theta), where theta is the angle such that 0 to theta encompasses tolerance% of the GGX probability
        // A higher tolerance means higher quality, but longer to integrate. 0.99 is a good balance between quality and performance from my testing.
        // The formula for the cutoff was derived after a few hours of finagling on WolframAlpha
        const float tolerance = 0.99;
        float cutoff = 1 - (alphaSquared * tolerance) / ((alphaSquared - 1) * tolerance + 1);

        // Precompute the normals for pixels on the input environment, and representatives for input environment patches
        // Precomputing here speeds up the main integration loop by a lot, since we need the normals to check against the cutoff.
        // Thus these get computed for each iteration, not just for the pixels that pass the cutoff.
        const uint32_t patchSize = std::min(std::min(inputEnv.width, inputEnv.height) / 16, 16u);
        std::vector<Vec3<float>> envNormals;
        std::vector<Vec3<float>> unitEnvNormals;
        std::vector<Vec3<float>> unitPatchNormals;

        envNormals.resize(inputEnv.width * inputEnv.height * 6);
        unitEnvNormals.resize(inputEnv.width * inputEnv.height * 6);
        unitPatchNormals.resize(inputEnv.width / patchSize * inputEnv.height / patchSize * 6);
        for (uint32_t face = 0; face < 6; face++) {
            for (uint32_t y = 0; y < inputEnv.height; y++) {
                for (uint32_t x = 0; x < inputEnv.width; x++) {
                    uint32_t i = (face * inputEnv.height + y) * inputEnv.width + x;
                    CubePixel environmentPixel { .face = face, .x = x, .y = y, };
                    envNormals[i] = getNormalFrom(environmentPixel, inputEnv.width, inputEnv.height);
                    unitEnvNormals[i] = linear::normalize(envNormals[i]);
                }
            }
        }
        for (uint32_t face = 0; face < 6; face++) {
            for (uint32_t y = 0; y < inputEnv.height / patchSize; y++) {
                for (uint32_t x = 0; x < inputEnv.width / patchSize; x++) {
                    uint32_t i = (face * inputEnv.height / patchSize + y) * inputEnv.width / patchSize + x;
                    CubePixel environmentPixel { .face = face, .x = x, .y = y, };
                    unitPatchNormals[i] = linear::normalize(getNormalFrom(environmentPixel, inputEnv.width / patchSize, inputEnv.height / patchSize));
                }
            }
        }

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
        std::cout << "Integrating GGX (Level " << mipLevel << " of " << mipLevels << "): [" << std::flush;

        // Iterate over each pixel in the GGX cubemap in parallel
        // Inspired by the code samples from Epic's 2013 PBR course notes: https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
        std::for_each(std::execution::par, pixels.begin(), pixels.end(), [&](CubePixel& pixel) {
            uint32_t ggxIndex = ((pixel.face * ggxEnv.height + pixel.y) * ggxEnv.width + pixel.x) * 4;
            Vec3<float> ggxNormal = linear::normalize(getNormalFrom(pixel, ggxEnv.width, ggxEnv.height));

            // Integrate the split sum.
            // As Epic gives it, its a sum over the importance-sampled luminances, but we can equivalently do this with quadrature by multiplying by the probability of each halfway-vector
            // To speed up the integration, we first split our input environment into patches, and check the normal of each patch against the cutoff.
            // That way, we can early out many pixels at once. This could throw away pixels that should make the cutoff, but only at the fringes where the pixels wouldn't contribute much anyways.
            Vec3<double> splitSum = Vec3<double>(0.0);
            double totalJacobian = 0.0;
            for (uint32_t face = 0; face < 6; face++) {
                for (uint32_t y = 0; y < inputEnv.height / patchSize; y++) {
                    for (uint32_t x = 0; x < inputEnv.width / patchSize; x++) {
                        uint32_t i = (face * inputEnv.height / patchSize + y) * inputEnv.width / patchSize + x;

                        // Check the patch against the cutoff
                        Vec3<float> halfway = unitPatchNormals[i] + ggxNormal;
                        float cosHalf = linear::dot(halfway, ggxNormal);
                        if (cosHalf * cosHalf > cutoff * linear::length2(halfway)) {
                            // Iterate over the pixels in the patch
                            for (uint32_t subY = 0; subY < patchSize; subY++) {
                                for (uint32_t subX = 0; subX < patchSize; subX++) {
                                    uint32_t totalX = x * patchSize + subX;
                                    uint32_t totalY = y * patchSize + subY;
                                    uint32_t environmentIndex = ((face * inputEnv.height + totalY) * inputEnv.height + totalX) * 4;

                                    // Check the pixel against the cutoff AND that the reflection doesn't go through the surface
                                    halfway = unitEnvNormals[environmentIndex / 4] + ggxNormal;
                                    cosHalf = linear::dot(halfway, ggxNormal);
                                    float cosRefl = linear::dot(unitEnvNormals[environmentIndex / 4], ggxNormal);
                                    if (cosHalf * cosHalf > cutoff * linear::length2(halfway) && cosRefl > 0.0) {
                                        // The pixel passed, so we can add to the integral.
                                        // Calculate the Jacobian due to differing solid angle (see integrateLambertian() for justication)
                                        float jacobian = linear::dot(ggxNormal, envNormals[environmentIndex / 4]);
                                        float length2 = linear::length2(envNormals[environmentIndex / 4]);
                                        jacobian /= (length2 * length2);

                                        // Calculate the GGX probability (D(h) * |n * h|)
                                        float NoH = linear::dot(linear::normalize(halfway), ggxNormal);
                                        float denom = NoH * NoH * (alphaSquared - 1) + 1;
                                        float ggxProb = alphaSquared * NoH / (M_PI * denom * denom);
                                        jacobian *= ggxProb;

                                        totalJacobian += jacobian;
                                        splitSum.x += inputEnv.data[environmentIndex] * jacobian;
                                        splitSum.y += inputEnv.data[environmentIndex + 1] * jacobian;
                                        splitSum.z += inputEnv.data[environmentIndex + 2] * jacobian;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Divide by the total sum of the Jacobians we used to give the final integral.
            // In integrateLambertian(), we normalized exactly with the pixel area, however totalJacobian includes these terms already
            splitSum = splitSum / totalJacobian;

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
    uint32_t cubemapSize = 16;
    uint32_t inputDownscale = 1;
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
        else if (currentArg == "--input-downscale") {
            currentIndex += 1;
            if (currentIndex >= argc) {
                PANIC("missing argument to --input-downscale");
            }

            std::string_view scaleStr = args[currentIndex];
            auto scaleResult = std::from_chars(scaleStr.data(), scaleStr.data() + scaleStr.size(), inputDownscale);
            if (scaleResult.ec == std::errc::invalid_argument || scaleResult.ec == std::errc::result_out_of_range) {
                PANIC("invalid argument to --input-downscale");
            }
            if (inputDownscale == 0 || (inputDownscale & (inputDownscale - 1)) != 0) {
                PANIC("scale factor for --input-downscale should be a power of 2 >= 1");
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

    // Downscale our input environment by averaging if needed
    uint32_t fullWidth = inputEnv.width;
    uint32_t fullHeight = inputEnv.height;
    if (inputDownscale != 1) {
        Cubemap downscaledInputEnv {
            .data = std::vector<float>(textureWidth / inputDownscale * textureHeight / inputDownscale * 6 * 4),
            .width = static_cast<uint32_t>(textureWidth) / inputDownscale,
            .height = static_cast<uint32_t>(textureHeight) / inputDownscale,
        };

        for (uint32_t y = 0; y < downscaledInputEnv.height * 6; y++) {
            for (uint32_t x = 0; x < downscaledInputEnv.width; x++) {
                uint32_t downscaleIndex = (y * downscaledInputEnv.width + x) * 4;
                Vec3<float> avg = Vec3<float>(0.0f);
                for (uint32_t avgY = 0; avgY < inputDownscale; avgY++) {
                    for (uint32_t avgX = 0; avgX < inputDownscale; avgX++) {
                        uint32_t fullX = x * inputDownscale + avgX;
                        uint32_t fullY = y * inputDownscale + avgY;
                        uint32_t fullSizeIndex = (fullY * textureWidth + fullX) * 4;
                        avg.x += inputEnv.data[fullSizeIndex];
                        avg.y += inputEnv.data[fullSizeIndex + 1];
                        avg.z += inputEnv.data[fullSizeIndex + 2];
                    }
                }
                avg = avg / static_cast<float>(inputDownscale * inputDownscale);
                downscaledInputEnv.data[downscaleIndex] = avg.x;
                downscaledInputEnv.data[downscaleIndex + 1] = avg.y;
                downscaledInputEnv.data[downscaleIndex + 2] = avg.z;
            }
        }

        inputEnv = downscaledInputEnv;
    }

    // Free our original image buffer
    stbi_image_free(pixels);

    if (doIntegrateLambertian) {
        integrateLambertian(inputPath, inputEnv, cubemapSize);
    }
    if (doIntegrateGGX) {
        integrateGGX(inputPath, inputEnv, fullWidth, fullHeight, cubemapSize);
    }

    return 0;
}
