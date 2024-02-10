#include <vulkan/vulkan.h>

#include "buffer.hpp"
#include "instance.hpp"
#include "linear.hpp"
#include "util.hpp"

CombinedBuffer::CombinedBuffer(std::shared_ptr<RenderInstance>& renderInstanceIn, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps) : renderInstance(renderInstanceIn) {
    VkBufferCreateInfo bufferInfo {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VK_ERR(vkCreateBuffer(renderInstance->device, &bufferInfo, nullptr, &buffer), "failed to create buffer!");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(renderInstance->device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = findMemoryType(*renderInstance, memRequirements.memoryTypeBits, memProps),
    };

    VK_ERR(vkAllocateMemory(renderInstance->device, &allocInfo, nullptr, &bufferMemory), "failed to allocate buffer memory!");

    vkBindBufferMemory(renderInstance->device, buffer, bufferMemory, 0);
}

CombinedBuffer::~CombinedBuffer() {
    vkDestroyBuffer(renderInstance->device, buffer, nullptr);
    vkFreeMemory(renderInstance->device, bufferMemory, nullptr);
}

CombinedImage::CombinedImage(std::shared_ptr<RenderInstance>& renderInstanceIn, uint32_t width, uint32_t height,
                             VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memProps) : renderInstance(renderInstanceIn) {
    createImage(*renderInstance, width, height, format, tiling, usage, memProps, image, imageMemory);
}

void createImage(RenderInstance const& renderInstance, uint32_t width, uint32_t height, VkFormat format,
                 VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memProps, VkImage& image, VkDeviceMemory& imageMemory) {
    VkImageCreateInfo imageInfo {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = { width, height, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = tiling,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VK_ERR(vkCreateImage(renderInstance.device, &imageInfo, nullptr, &image), "failed to create image!");

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(renderInstance.device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = findMemoryType(renderInstance, memRequirements.memoryTypeBits, memProps),
    };

    VK_ERR(vkAllocateMemory(renderInstance.device, &allocInfo, nullptr, &imageMemory), "failed to allocate image memory!");
    vkBindImageMemory(renderInstance.device, image, imageMemory, 0);
}

CombinedImage::~CombinedImage() {
    vkDestroyImage(renderInstance->device, image, nullptr);
    vkFreeMemory(renderInstance->device, imageMemory, nullptr);
}

void copyBuffers(RenderInstance const& renderInstance, BufferCopy* bufferCopyInfos, uint32_t bufferCopyCount) {
    VkCommandBuffer commandBuffer = beginSingleUseCBuffer(renderInstance);

    for (uint32_t i = 0; i < bufferCopyCount; i++) {
        BufferCopy& copyInfo = bufferCopyInfos[i];

        VkBufferCopy copyCmd {
            .srcOffset = copyInfo.srcOffset,
            .dstOffset = copyInfo.dstOffset,
            .size = copyInfo.size,
        };
        vkCmdCopyBuffer(commandBuffer, copyInfo.srcBuffer, copyInfo.dstBuffer, 1, &copyCmd);
    }

    endSingleUseCBuffer(renderInstance, commandBuffer);
}

// Helper function that gets the memory type we need for allocating a buffer
uint32_t findMemoryType(RenderInstance const& renderInstance, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(renderInstance.physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    PANIC("failed to find suitable memory type!");
}
