#pragma once
#include <vulkan/vulkan.h>

#include <stdexcept>

// Error handling macro
#define VK_ERR(res, msg) if (res != VK_SUCCESS) { throw std::runtime_error(msg); }

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format);
