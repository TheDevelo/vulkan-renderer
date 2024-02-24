#include <vulkan/vulkan.h>

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
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        PANIC("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

// NOTE: I'm assuming the destination RGB has an alpha channel, since R32G32B32_SFLOAT seems like it is not commonly supported
void convertRGBEtoRGB(uint8_t* src, float* dst, uint32_t pixelCount) {
    for (uint32_t pixel = 0; pixel < pixelCount; pixel++) {
        uint8_t* srcPixel = src + pixel * 4;
        float* dstPixel = dst + pixel * 4;

        dstPixel[3] = 1.0f;
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
