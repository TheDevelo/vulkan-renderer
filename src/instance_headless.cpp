#include <charconv>
#include <iostream>
#include <fstream>
#include <cstring>

#include "instance.hpp"
#include "options.hpp"
#include "buffer.hpp"
#include "util.hpp"

const int MAX_HEADLESS_RENDER_IMAGES = 3;

void RenderInstance::initHeadless() {
    // Parse the event file into the event list
    std::ifstream eventsFile(options::getHeadlessEventsPath());
    std::string line;
    while (std::getline(eventsFile, line)) {
        RenderInstanceEvent event;

        // Get the timestamp
        std::string timestampStr = line.substr(0, line.find(" "));
        line.erase(0, line.find(" ") + 1);

        auto charconvResult = std::from_chars(timestampStr.data(), timestampStr.data() + timestampStr.size(), event.timestamp);
        if (charconvResult.ec == std::errc::invalid_argument || charconvResult.ec == std::errc::result_out_of_range) {
            PANIC("Headless events parsing error: timestamp invalid");
        }

        // Get the command type
        std::string eventType = line.substr(0, line.find(" "));
        line.erase(0, line.find(" ") + 1);

        // Parse the rest of the command
        if (eventType == "AVAILABLE") {
            event.type = RI_EV_INTERNAL_AVAILABLE;
        }
        else if (eventType == "SAVE") {
            event.type = RI_EV_INTERNAL_SAVE;
            event.data = InternalSaveEvent {
                .outputPath = line,
            };
        }
        else if (eventType == "MARK") {
            event.type = RI_EV_INTERNAL_MARK;
            event.data = InternalMarkEvent {
                .message = line,
            };
        }
        else if (eventType == "PLAY") {
            // Split the arguments into time and rate
            std::string timeStr = line.substr(0, line.find(" "));
            line.erase(0, line.find(" ") + 1);

            // Parse
            float animTime;
            float animRate;
            auto timeResult = std::from_chars(timeStr.data(), timeStr.data() + timeStr.size(), animTime);
            auto rateResult = std::from_chars(line.data(), line.data() + line.size(), animRate);
            if (timeResult.ec == std::errc::invalid_argument || timeResult.ec == std::errc::result_out_of_range ||
                rateResult.ec == std::errc::invalid_argument || rateResult.ec == std::errc::result_out_of_range) {
                PANIC("Headless events parsing error: invalid arguments to PLAY");
            }

            event.type = RI_EV_SET_ANIMATION;
            event.data = SetAnimationEvent {
                .time = animTime,
                .rate = animRate,
            };
        }
        else {
            PANIC("Headless events parsing error: invalid event type");
        }

        headlessEvents.push_back(event);
    }

    // Create our render images
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
    return currentHeadlessEvent == headlessEvents.size();
}

float RenderInstance::processEventsHeadless() {
    // Clear the event queue before processing any events
    eventQueue.clear();

    float startTime;
    if (currentHeadlessEvent == 0) {
        startTime = 0.0f;
    }
    else {
        startTime = headlessEvents[currentHeadlessEvent - 1].timestamp;
    }
    float endTime = startTime;

    while (currentHeadlessEvent < headlessEvents.size()) {
        RenderInstanceEvent const& event = headlessEvents[currentHeadlessEvent];
        // Advance to the next event, in case we hit AVAILABLE and break out
        endTime = event.timestamp;
        currentHeadlessEvent += 1;

        if (event.type == RI_EV_INTERNAL_AVAILABLE) {
            break;
        }
        else if (event.type == RI_EV_INTERNAL_SAVE) {
            // Save last rendered image to output file
            InternalSaveEvent const& data = get<InternalSaveEvent>(event.data);

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
            // We first copy to a secondary buffer because writing to the file itself comparitively very long
            // That way, we can throw it into a thread and have it work in the background
            std::unique_ptr<uint8_t[]> imageCopy(new uint8_t[renderImageExtent.width * renderImageExtent.height * 4]);
            memcpy(imageCopy.get(), imageCopyBufferMap, renderImageExtent.width * renderImageExtent.height * 4);

            std::thread writer([renderImageExtent = renderImageExtent, imageCopy = move(imageCopy), outputPath = data.outputPath] {
                std::ofstream outputImage(outputPath, std::ios::out | std::ios::binary);

                std::string header = string_format("P6 %d %d 255\n", renderImageExtent.width, renderImageExtent.height);
                outputImage.write(header.c_str(), header.size());

                for (uint32_t p = 0; p < renderImageExtent.width * renderImageExtent.height; p++) {
                    // Need to copy pixel by pixel since PPM doesn't support alpha
                    outputImage.write(reinterpret_cast<const char*>(imageCopy.get()) + p * 4, 3);
                }
            });
            imageWriters.push_back(move(writer));
        }
        else if (event.type == RI_EV_INTERNAL_MARK) {
            InternalMarkEvent const& data = get<InternalMarkEvent>(event.data);
            std::cout << "MARK " << data.message << std::endl;
        }
        else {
            eventQueue.push_back(event);
        }
    }

    // Divide final time delta by 1000000 since endTime and startTime are in microseconds
    return (endTime - startTime) / 1000000.0f;
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
