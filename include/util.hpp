#pragma once
#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>

#include <memory>
#include <string>
#include <stdexcept>

#include "instance.hpp"

// Error handling macros
#define VK_ERR(fn, msg) { VkResult res = fn; if (res != VK_SUCCESS) { throw std::runtime_error(string_format("%s - %s", msg, string_VkResult(res))); } }
#define PANIC(msg) throw std::runtime_error(msg)

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);

// Single use command buffer helpers
VkCommandBuffer beginSingleUseCBuffer(RenderInstance const& renderInstance);
void endSingleUseCBuffer(RenderInstance const& renderInstance, VkCommandBuffer commandBuffer);

// Reading file to buffer helper
std::vector<uint8_t> readFile(const std::string& filename);

// String formatting function, since apparently my gcc doesn't support the c++20 string formatting library???
template<typename ...Args>
inline std::string string_format(const std::string& format, Args... args) {
    // Determine the size needed to format our string.
    int sizeI = std::snprintf( nullptr, 0, format.c_str(), args...);
    if (sizeI < 0) {
        throw std::runtime_error("Error during formatting.");
    }
    // Add one extra for '\0'
    sizeI += 1;

    // Allocate a string buffer for the formatting
    size_t size = static_cast<size_t>(sizeI);
    std::unique_ptr<char[]> buf(new char[size]);

    // Format the string
    std::snprintf(buf.get(), size, format.c_str(), args...);
    // Remove last char as it is '\0'
    return std::string(buf.get(), buf.get() + size - 1);
}

void convertRGBEtoRGB(uint8_t* src, float* dst, uint32_t pixelCount);
void convertRGBtoRGBE(float* src, uint8_t* dst, uint32_t pixelCount);
