#include <vulkan/vulkan.h>

#include "buffer.hpp"
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
