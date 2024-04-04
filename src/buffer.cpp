#include <vulkan/vulkan.h>
#include "stb_image.h"

#include <cstring>

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

// =============================================================================
// CombinedBuffer
// =============================================================================
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
    if (renderInstance != nullptr) {
        vkDestroyBuffer(renderInstance->device, buffer, nullptr);
        vkFreeMemory(renderInstance->device, bufferMemory, nullptr);
    }
}

CombinedBuffer::CombinedBuffer(CombinedBuffer&& src) {
    buffer = src.buffer;
    bufferMemory = src.bufferMemory;
    renderInstance = src.renderInstance;
    src.renderInstance = nullptr; // Set renderInstance to nullptr to indicate the original has been deactivated
}

// =============================================================================
// CombinedImage
// =============================================================================
CombinedImage::CombinedImage(std::shared_ptr<RenderInstance>& renderInstanceIn, uint32_t width, uint32_t height, VkFormat format,
                             VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memProps, VkImageAspectFlags aspectFlags) : renderInstance(renderInstanceIn) {
    createImage(*renderInstance, width, height, format, tiling, usage, memProps, image, imageMemory);
    imageView = createImageView(renderInstance->device, image, format, aspectFlags);
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
    if (renderInstance != nullptr) {
        vkDestroyImageView(renderInstance->device, imageView, nullptr);
        vkDestroyImage(renderInstance->device, image, nullptr);
        vkFreeMemory(renderInstance->device, imageMemory, nullptr);
    }
}

CombinedImage::CombinedImage(CombinedImage&& src) {
    image = src.image;
    imageMemory = src.imageMemory;
    imageView = src.imageView;
    renderInstance = src.renderInstance;
    src.renderInstance = nullptr; // Set renderInstance to nullptr to indicate the original has been deactivated
}

// =============================================================================
// CombinedCubemap
// =============================================================================
CombinedCubemap::CombinedCubemap(std::shared_ptr<RenderInstance>& renderInstanceIn, uint32_t width, uint32_t height, uint32_t mipLevels,
                                 VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memProps, VkImageAspectFlags aspectFlags) : renderInstance(renderInstanceIn) {
    VkImageCreateInfo imageInfo {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = { width, height, 1 },
        .mipLevels = mipLevels,
        .arrayLayers = 6,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = tiling,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VK_ERR(vkCreateImage(renderInstance->device, &imageInfo, nullptr, &image), "failed to create image!");

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(renderInstance->device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = findMemoryType(*renderInstance, memRequirements.memoryTypeBits, memProps),
    };

    VK_ERR(vkAllocateMemory(renderInstance->device, &allocInfo, nullptr, &imageMemory), "failed to allocate image memory!");
    vkBindImageMemory(renderInstance->device, image, imageMemory, 0);

    // Create the image view
    VkImageViewCreateInfo imageViewCreateInfo {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_CUBE,
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
            .levelCount = mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 6,
        },
    };

    VK_ERR(vkCreateImageView(renderInstance->device, &imageViewCreateInfo, nullptr, &imageView), "failed to create image view!");
}

CombinedCubemap::~CombinedCubemap() {
    if (renderInstance != nullptr) {
        vkDestroyImageView(renderInstance->device, imageView, nullptr);
        vkDestroyImage(renderInstance->device, image, nullptr);
        vkFreeMemory(renderInstance->device, imageMemory, nullptr);
    }
}

CombinedCubemap::CombinedCubemap(CombinedCubemap&& src) {
    image = src.image;
    imageMemory = src.imageMemory;
    imageView = src.imageView;
    renderInstance = src.renderInstance;
    src.renderInstance = nullptr; // Set renderInstance to nullptr to indicate the original has been deactivated
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

void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageSubresourceRange imageRange) {
    VkImageMemoryBarrier barrier {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = imageRange,
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
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
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
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, VkImageSubresourceLayers imageLayers) {
    VkBufferImageCopy region {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = imageLayers,
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

// Image load helpers
// NOTE: format should be 4 bytes wide, to match the 4 bytes per pixel of the original image
std::unique_ptr<CombinedImage> loadImage(std::shared_ptr<RenderInstance>& renderInstance, std::string const& path, VkFormat format) {
    // Load the texture
    int textureWidth, textureHeight, textureChannels;
    stbi_uc* pixels = stbi_load(path.c_str(), &textureWidth, &textureHeight, &textureChannels, STBI_rgb_alpha);
    if (!pixels) {
        PANIC(string_format("failed to load texture image! - %s", path.c_str()));
    }

    // Create a staging buffer for our image
    VkDeviceSize imageSize = textureWidth * textureHeight * 4;
    CombinedBuffer stagingBuffer(renderInstance, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data;
    vkMapMemory(renderInstance->device, stagingBuffer.bufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(renderInstance->device, stagingBuffer.bufferMemory);

    // Free our CPU-side loaded texture
    stbi_image_free(pixels);

    // Create the GPU-side image
    std::unique_ptr<CombinedImage> image = std::make_unique<CombinedImage>(renderInstance, static_cast<uint32_t>(textureWidth), static_cast<uint32_t>(textureHeight), format,
                                                                           VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    // Copy staging buffer to our image and prepare it for shader reads
    VkCommandBuffer commandBuffer = beginSingleUseCBuffer(*renderInstance);

    transitionImageLayout(commandBuffer, image->image, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(commandBuffer, stagingBuffer.buffer, image->image, static_cast<uint32_t>(textureWidth), static_cast<uint32_t>(textureHeight));
    transitionImageLayout(commandBuffer, image->image, format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    endSingleUseCBuffer(*renderInstance, commandBuffer);

    return image;
}

// NOTE: mipLevels controls how many mipmap levels get created, but doesn't fill/transition them
std::unique_ptr<CombinedCubemap> loadCubemap(std::shared_ptr<RenderInstance>& renderInstance, std::string const& path, uint32_t mipLevels) {
    // Load the texture
    int textureWidth, textureHeight, textureChannels;
    stbi_uc* pixels = stbi_load(path.c_str(), &textureWidth, &textureHeight, &textureChannels, STBI_rgb_alpha);
    if (!pixels) {
        PANIC(string_format("failed to load texture image! - %s", path.c_str()));
    }
    textureHeight /= 6;

    // Convert image from RGBE format
    std::vector<float> rgbFloats;
    rgbFloats.resize(textureWidth * textureHeight * 6 * 4);
    convertRGBEtoRGB(pixels, rgbFloats.data(), textureWidth * textureHeight * 6);

    // Create a staging buffer for our image
    VkDeviceSize imageSize = rgbFloats.size() * sizeof(float);
    CombinedBuffer stagingBuffer(renderInstance, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data;
    vkMapMemory(renderInstance->device, stagingBuffer.bufferMemory, 0, imageSize, 0, &data);
    memcpy(data, rgbFloats.data(), static_cast<size_t>(imageSize));
    vkUnmapMemory(renderInstance->device, stagingBuffer.bufferMemory);

    // Free our CPU-side loaded texture
    stbi_image_free(pixels);

    // Create the GPU-side image
    std::unique_ptr<CombinedCubemap> cubemap = std::make_unique<CombinedCubemap>(renderInstance, static_cast<uint32_t>(textureWidth), static_cast<uint32_t>(textureHeight), mipLevels, VK_FORMAT_R32G32B32A32_SFLOAT,
                                                                                 VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    // Copy staging buffer to our image and prepare it for shader reads
    VkCommandBuffer commandBuffer = beginSingleUseCBuffer(*renderInstance);

    transitionImageLayout(commandBuffer, cubemap->image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VkImageSubresourceRange {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 6,
            });
    copyBufferToImage(commandBuffer, stagingBuffer.buffer, cubemap->image, static_cast<uint32_t>(textureWidth), static_cast<uint32_t>(textureHeight),
            VkImageSubresourceLayers {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 6,
            });
    transitionImageLayout(commandBuffer, cubemap->image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VkImageSubresourceRange {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 6,
            });

    endSingleUseCBuffer(*renderInstance, commandBuffer);

    return cubemap;
}

void loadMipmapIntoCubemap(std::shared_ptr<RenderInstance>& renderInstance, CombinedCubemap& cubemap, std::string const& path, uint32_t mipLevel) {
    // Load the texture
    int textureWidth, textureHeight, textureChannels;
    stbi_uc* pixels = stbi_load(path.c_str(), &textureWidth, &textureHeight, &textureChannels, STBI_rgb_alpha);
    if (!pixels) {
        PANIC(string_format("failed to load texture image! - %s", path.c_str()));
    }
    textureHeight /= 6;

    // Convert image from RGBE format
    std::vector<float> rgbFloats;
    rgbFloats.resize(textureWidth * textureHeight * 6 * 4);
    convertRGBEtoRGB(pixels, rgbFloats.data(), textureWidth * textureHeight * 6);

    // Create a staging buffer for our image
    VkDeviceSize imageSize = rgbFloats.size() * sizeof(float);
    CombinedBuffer stagingBuffer(renderInstance, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data;
    vkMapMemory(renderInstance->device, stagingBuffer.bufferMemory, 0, imageSize, 0, &data);
    memcpy(data, rgbFloats.data(), static_cast<size_t>(imageSize));
    vkUnmapMemory(renderInstance->device, stagingBuffer.bufferMemory);

    // Free our CPU-side loaded texture
    stbi_image_free(pixels);

    // Copy staging buffer to the desired mipmap leveli in the cubemap and prepare it for shader reads
    VkCommandBuffer commandBuffer = beginSingleUseCBuffer(*renderInstance);
    transitionImageLayout(commandBuffer, cubemap.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VkImageSubresourceRange {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = mipLevel,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 6,
            });
    copyBufferToImage(commandBuffer, stagingBuffer.buffer, cubemap.image, static_cast<uint32_t>(textureWidth), static_cast<uint32_t>(textureHeight),
            VkImageSubresourceLayers {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = mipLevel,
                .baseArrayLayer = 0,
                .layerCount = 6,
            });
    transitionImageLayout(commandBuffer, cubemap.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VkImageSubresourceRange {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = mipLevel,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 6,
            });
    endSingleUseCBuffer(*renderInstance, commandBuffer);
}
