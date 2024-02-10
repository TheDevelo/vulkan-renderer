#include "instance.hpp"
#include "options.hpp"
#include "buffer.hpp"
#include "util.hpp"

const int MAX_HEADLESS_RENDER_IMAGES = 3;

void RenderInstance::initHeadless() {
    renderImageFormat = VK_FORMAT_R8G8B8A8_UNORM;
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

    lastUsedImage = 0;
};

void RenderInstance::cleanupHeadless() {
    for (int i = 0; i < MAX_HEADLESS_RENDER_IMAGES; i++) {
        vkDestroyImage(device, headlessRenderImages[i], nullptr);
        vkFreeMemory(device, headlessRenderImagesMemory[i], nullptr);
    }
}

bool RenderInstance::shouldCloseHeadless() {
    return false;
}

void RenderInstance::processEventsHeadless() {
}

RenderInstanceImageStatus RenderInstance::acquireImageHeadless(VkSemaphore availableSemaphore, uint32_t& dstImageIndex) {
    dstImageIndex = lastUsedImage;
    return RI_TARGET_OK;
}

RenderInstanceImageStatus RenderInstance::presentImageHeadless(VkSemaphore renderFinishedSemaphore, uint32_t imageIndex) {
    renderingSemaphores[imageIndex] = renderFinishedSemaphore;
    return RI_TARGET_OK;
}
