#include <vulkan/vulkan.h>

#include <algorithm>
#include <fstream>
#include <cmath>

#include "instance.hpp"
#include "util.hpp"

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
    VkImageViewCreateInfo imageViewCreateInfo {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components = VkComponentMapping {
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange = VkImageSubresourceRange {
            .aspectMask = aspectFlags,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    VkImageView result;
    VK_ERR(vkCreateImageView(device, &imageViewCreateInfo, nullptr, &result), "failed to create image view!");

    return result;
}

// Create a command buffer that will be used for a single time
VkCommandBuffer beginSingleUseCBuffer(RenderInstance const& renderInstance) {
    VkCommandBufferAllocateInfo commandBufferAllocInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = renderInstance.commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VkCommandBuffer commandBuffer;
    VK_ERR(vkAllocateCommandBuffers(renderInstance.device, &commandBufferAllocInfo, &commandBuffer), "failed to allocate command buffer!");

    VkCommandBufferBeginInfo beginInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_ERR(vkBeginCommandBuffer(commandBuffer, &beginInfo), "failed to begin command buffer!");

    return commandBuffer;
}

void endSingleUseCBuffer(RenderInstance const& renderInstance, VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer,
    };
    vkQueueSubmit(renderInstance.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(renderInstance.graphicsQueue);

    vkFreeCommandBuffers(renderInstance.device, renderInstance.commandPool, 1, &commandBuffer);
}

// Helper function to read files into a vector of chars
std::vector<uint8_t> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        PANIC("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<uint8_t> buffer(fileSize);

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

    file.close();

    return buffer;
}

// RGBE conversion functions inspired 15-466 image based lighting code
// NOTE: I'm assuming the destination RGB has an alpha channel, since R32G32B32_SFLOAT seems like it is not commonly supported
// R32G32B32A32 is, so I use that instead. Support stats per https://vulkan.gpuinfo.org/.
void convertRGBEtoRGB(uint8_t* src, float* dst, uint32_t pixelCount) {
    for (uint32_t pixel = 0; pixel < pixelCount; pixel++) {
        uint8_t* srcPixel = src + pixel * 4;
        float* dstPixel = dst + pixel * 4;

        dstPixel[3] = 1.0f; // Alpha value, so set to 1 as RGBE doesn't support alpha
        if (srcPixel[0] == 0 && srcPixel[1] == 0 && srcPixel[2] == 0 && srcPixel[3] == 0) {
            dstPixel[0] = 0.0f;
            dstPixel[1] = 0.0f;
            dstPixel[2] = 0.0f;
        }
        else {
            int exp = static_cast<int>(srcPixel[3]) - 128;
            dstPixel[0] = std::ldexp((srcPixel[0] + 0.5f) / 256.0f, exp);
            dstPixel[1] = std::ldexp((srcPixel[1] + 0.5f) / 256.0f, exp);
            dstPixel[2] = std::ldexp((srcPixel[2] + 0.5f) / 256.0f, exp);
        }
    }
}

void convertRGBtoRGBE(float* src, uint8_t* dst, uint32_t pixelCount) {
    for (uint32_t pixel = 0; pixel < pixelCount; pixel++) {
        float* srcPixel = src + pixel * 4;
        uint8_t* dstPixel = dst + pixel * 4;

        float max = std::max(std::max(srcPixel[0], srcPixel[1]), srcPixel[2]);
        if (max <= 1e-32f) {
            // Our max value is too low, so just specify pure 0
            dstPixel[0] = 0;
            dstPixel[1] = 0;
            dstPixel[2] = 0;
            dstPixel[3] = 0;
            continue;
        }

        int exp;
        float fracMult = 256.0f * std::frexp(max, &exp) / max;

        if (exp > 127) {
            // Exponent is greater than what RGBE can represent, so clamp to max value
            dstPixel[0] = 255;
            dstPixel[1] = 255;
            dstPixel[2] = 255;
            dstPixel[3] = 255;
            continue;
        }

        dstPixel[0] = std::clamp(int32_t(srcPixel[0] * fracMult), 0, 255);
        dstPixel[1] = std::clamp(int32_t(srcPixel[1] * fracMult), 0, 255);
        dstPixel[2] = std::clamp(int32_t(srcPixel[2] * fracMult), 0, 255);
        dstPixel[3] = exp + 128;
    }
}
