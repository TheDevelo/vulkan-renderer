#include <vulkan/vulkan.h>

#include "buffer.hpp"
#include "instance.hpp"
#include "linear.hpp"
#include "util.hpp"

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

CombinedBuffer::CombinedBuffer(std::shared_ptr<RenderInstance>& renderInstanceIn, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps) : renderInstance(renderInstanceIn) {
    createBuffer(*renderInstance, size, usage, memProps, buffer, bufferMemory);
}

void createBuffer(RenderInstance const& renderInstance, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VK_ERR(vkCreateBuffer(renderInstance.device, &bufferInfo, nullptr, &buffer), "failed to create buffer!");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(renderInstance.device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = findMemoryType(renderInstance, memRequirements.memoryTypeBits, memProps),
    };

    VK_ERR(vkAllocateMemory(renderInstance.device, &allocInfo, nullptr, &bufferMemory), "failed to allocate buffer memory!");

    vkBindBufferMemory(renderInstance.device, buffer, bufferMemory, 0);
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

// Buffer/Image manipulation helpers
void copyBuffers(VkCommandBuffer commandBuffer, BufferCopy* bufferCopyInfos, uint32_t bufferCopyCount) {
    for (uint32_t i = 0; i < bufferCopyCount; i++) {
        BufferCopy& copyInfo = bufferCopyInfos[i];

        VkBufferCopy copyCmd {
            .srcOffset = copyInfo.srcOffset,
            .dstOffset = copyInfo.dstOffset,
            .size = copyInfo.size,
        };
        vkCmdCopyBuffer(commandBuffer, copyInfo.srcBuffer, copyInfo.dstBuffer, 1, &copyCmd);
    }
}

void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkImageMemoryBarrier barrier {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = VkImageSubresourceRange {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    // Determine the source and destination stages/access masks. Currently hardcoded for our old/new layout pairs, maybe there is a more dynamic way to do this?
    VkPipelineStageFlags srcStage;
    VkPipelineStageFlags dstStage;
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkBufferImageCopy region {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = VkImageSubresourceLayers {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = { 0, 0, 0 },
        .imageExtent = { width, height, 1 },
    };

    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

void copyImageToBuffer(VkCommandBuffer commandBuffer, VkImage image, VkBuffer buffer, uint32_t width, uint32_t height) {
    VkBufferImageCopy region {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = VkImageSubresourceLayers {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = { 0, 0, 0 },
        .imageExtent = { width, height, 1 },
    };

    vkCmdCopyImageToBuffer(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
}
