// Header file for both buffers and images that free themselves. Why images in the file called buffer.hpp? Sue me.
#pragma once
#include <vulkan/vulkan.h>

#include <memory>

#include "instance.hpp"

class CombinedBuffer {
public:
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;

    CombinedBuffer(std::shared_ptr<RenderInstance>& renderInstanceIn, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps);
    ~CombinedBuffer();

private:
    // Need to keep a copy of renderInstance for the destructor
    std::shared_ptr<RenderInstance> renderInstance;
};

class CombinedImage {
public:
    VkImage image;
    VkDeviceMemory imageMemory;
    VkImageView imageView;

    CombinedImage(std::shared_ptr<RenderInstance>& renderInstanceIn, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memProps, VkImageAspectFlags aspectFlags);
    ~CombinedImage();

private:
    // Need to keep a copy of renderInstance for the destructor
    std::shared_ptr<RenderInstance> renderInstance;
};

// Similar to a combined image, but for a cubemap instead
class CombinedCubemap {
public:
    VkImage image;
    VkDeviceMemory imageMemory;
    VkImageView imageView;

    CombinedCubemap(std::shared_ptr<RenderInstance>& renderInstanceIn, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memProps, VkImageAspectFlags aspectFlags);
    ~CombinedCubemap();

private:
    // Need to keep a copy of renderInstance for the destructor
    std::shared_ptr<RenderInstance> renderInstance;
};

// Container struct for buffer copy regions
struct BufferCopy {
    VkBuffer srcBuffer;
    VkDeviceSize srcOffset;
    VkBuffer dstBuffer;
    VkDeviceSize dstOffset;
    VkDeviceSize size;
};

// Buffer/image copying and transitioning helpers
void copyBuffers(VkCommandBuffer commandBuffer, BufferCopy* bufferCopyInfos, uint32_t bufferCopyCount);
void copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t layers);
void copyImageToBuffer(VkCommandBuffer commandBuffer, VkImage image, VkBuffer buffer, uint32_t width, uint32_t height);
void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);

// Raw buffer/image creation methods (for headless, as RenderInstance can't use CombinedImage/CombinedBuffers)
void createBuffer(RenderInstance const& renderInstance, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
void createImage(RenderInstance const& renderInstance, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
                 VkImageUsageFlags usage, VkMemoryPropertyFlags memProps, VkImage& image, VkDeviceMemory& imageMemory);

// Texture/Cubemap loading helpers
std::unique_ptr<CombinedImage> loadImage(std::shared_ptr<RenderInstance>& renderInstance, std::string const& path);
std::unique_ptr<CombinedCubemap> loadCubemap(std::shared_ptr<RenderInstance>& renderInstance, std::string const& path);
