#include <iostream>
#include <fstream>

#include "instance.hpp"
#include "options.hpp"
#include "buffer.hpp"
#include "util.hpp"

const int MAX_HEADLESS_RENDER_IMAGES = 3;

void RenderInstance::initHeadless() {
    renderImageFormat = VK_FORMAT_R8G8B8A8_SRGB;
    renderImageExtent = VkExtent2D {
        .width = options::getWindowWidth(),
        .height = options::getWindowHeight(),
    };

    // Create the render targets we want to use for headless mode
    for (int i = 0; i < MAX_HEADLESS_RENDER_IMAGES; i++) {
        VkImage image;
        VkDeviceMemory imageMemory;
        createImage(*this, renderImageExtent.width, renderImageExtent.height,
                    renderImageFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    image, imageMemory);

        headlessRenderImages.push_back(image);
        headlessRenderImagesMemory.push_back(imageMemory);
        renderImageViews.push_back(createImageView(device, image, renderImageFormat, VK_IMAGE_ASPECT_COLOR_BIT));

        renderingSemaphores.push_back(VK_NULL_HANDLE);
    }

    // Create the copy destination buffer for when we want to save a rendered image
    VkDeviceSize imageCopyBufferSize = renderImageExtent.width * renderImageExtent.height * 4;
    createBuffer(*this, imageCopyBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, imageCopyBuffer, imageCopyBufferMemory);
    vkMapMemory(device, imageCopyBufferMemory, 0, imageCopyBufferSize, 0, &imageCopyBufferMap);

    lastUsedImage = 0;
    renderingSemaphoreValues.resize(MAX_HEADLESS_RENDER_IMAGES);
};

void RenderInstance::cleanupHeadless() {
    for (int i = 0; i < MAX_HEADLESS_RENDER_IMAGES; i++) {
        vkDestroyImage(device, headlessRenderImages[i], nullptr);
        vkFreeMemory(device, headlessRenderImagesMemory[i], nullptr);
    }
    vkDestroyBuffer(device, imageCopyBuffer, nullptr);
    vkFreeMemory(device, imageCopyBufferMemory, nullptr);
}

bool RenderInstance::shouldCloseHeadless() {
    static int x = 0;
    x += 1;
    return x > 10000;
}

void RenderInstance::processEventsHeadless() {
    static int x = 0;
    x += 1;
    if (x % 100 == 0) {
        // Wait for the most recently released frame to finish rendering
        int lastReleasedImage = lastUsedImage - 1;
        if (lastReleasedImage == -1) {
            lastReleasedImage = MAX_HEADLESS_RENDER_IMAGES - 1;
        }
        VkSemaphoreWaitInfo waitInfo {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores = &renderingSemaphores[lastReleasedImage],
            .pValues = &renderingSemaphoreValues[lastReleasedImage],
        };
        vkWaitSemaphores(device, &waitInfo, UINT64_MAX);

        // Copy the image to the copy buffer
        transitionImageLayout(*this, headlessRenderImages[lastReleasedImage], renderImageFormat, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        copyImageToBuffer(*this, headlessRenderImages[lastReleasedImage], imageCopyBuffer, renderImageExtent.width, renderImageExtent.height);
        transitionImageLayout(*this, headlessRenderImages[lastReleasedImage], renderImageFormat, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        // Output the copied image to file using the ppm file format
        std::ofstream outputImage(string_format("output%d.ppm", x/100), std::ios::out | std::ios::binary);
        std::string header = string_format("P6 %d %d 255\n", renderImageExtent.width, renderImageExtent.height);
        outputImage.write(header.c_str(), header.size());
        for (uint32_t p = 0; p < renderImageExtent.width * renderImageExtent.height; p++) {
            // Need to copy pixel by pixel since PPM doesn't support alpha
            outputImage.write(reinterpret_cast<const char*>(imageCopyBufferMap) + p * 4, 3);
        }
    }
}

RenderInstanceImageStatus RenderInstance::acquireImageHeadless(VkSemaphore availableSemaphore, uint64_t semaphoreCurVal, uint32_t& dstImageIndex) {
    // Wait for the last used image to finish rendering
    if (renderingSemaphores[lastUsedImage] != VK_NULL_HANDLE) {
        VkSemaphoreWaitInfo waitInfo {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores = &renderingSemaphores[lastUsedImage],
            .pValues = &renderingSemaphoreValues[lastUsedImage],
        };
        vkWaitSemaphores(device, &waitInfo, UINT64_MAX);
    }

    // Tell the client the image to use
    dstImageIndex = lastUsedImage;

    // Immediately signal the image is available (because we waited for it to be ready)
    VkSemaphoreSignalInfo signalInfo {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .semaphore = availableSemaphore,
        .value = semaphoreCurVal + 1,
    };
    vkSignalSemaphore(device, &signalInfo);

    // Set the next image as longest used ago
    lastUsedImage = (lastUsedImage + 1) % MAX_HEADLESS_RENDER_IMAGES;
    return RI_TARGET_OK;
}

RenderInstanceImageStatus RenderInstance::presentImageHeadless(VkSemaphore renderFinishedSemaphore, uint64_t semaphoreCurVal, uint32_t imageIndex) {
    renderingSemaphores[imageIndex] = renderFinishedSemaphore;
    renderingSemaphoreValues[imageIndex] = semaphoreCurVal + 1;
    return RI_TARGET_OK;
}
