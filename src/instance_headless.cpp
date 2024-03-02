#include <charconv>
#include <iostream>
#include <fstream>
#include <cstring>

#include "instance.hpp"
#include "options.hpp"
#include "buffer.hpp"
#include "scene.hpp"
#include "util.hpp"

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
        else if (eventType == "CAMERA") {
            event.type = RI_EV_SWAP_FIXED_CAMERA;
            event.data = SwapFixedCameraEvent {
                .name = line,
            };
        }
        else if (eventType == "CULLING") {
            CullingMode cullingMode;
            if (line == "none") {
                cullingMode = CullingMode::OFF;
            }
            else if (line == "frustum") {
                cullingMode = CullingMode::FRUSTUM;
            }
            else if (line == "bvh") {
                cullingMode = CullingMode::BVH;
            }
            else {
                PANIC("Headless events parsing error: Invalid culling mode given to CULLING");
            }

            event.type = RI_EV_CHANGE_CULLING;
            event.data = ChangeCullingEvent {
                .cullingMode = cullingMode,
            };
        }
        else if (eventType == "EXPOSURE") {
            float exposure;
            auto result = std::from_chars(line.data(), line.data() + line.size(), exposure);
            if (result.ec == std::errc::invalid_argument || result.ec == std::errc::result_out_of_range) {
                PANIC("Headless events parsing error: invalid arguments to EXPOSURE");
            }

            event.type = RI_EV_EXPOSURE;
            event.data = ExposureEvent {
                .setOrMultiply = false,
                .exposure = exposure,
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
    for (uint32_t i = 0; i < options::getHeadlessRenderTargetCount(); i++) {
        HeadlessRenderTarget target {
            .renderingSemaphore = VK_NULL_HANDLE,
        };

        // Create the render image itself
        createImage(*this, renderImageExtent.width, renderImageExtent.height,
                    renderImageFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    target.image, target.imageMemory);

        // Allocate the copying command buffer
        VkCommandBufferAllocateInfo commandBufferAllocInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VK_ERR(vkAllocateCommandBuffers(device, &commandBufferAllocInfo, &target.copyCommandBuffer), "failed to allocate command buffer!");

        // Create the copy destination buffer for when we want to save a rendered image
        VkDeviceSize copyBufferSize = renderImageExtent.width * renderImageExtent.height * 4;
        createBuffer(*this, copyBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, target.copyBuffer, target.copyBufferMemory);
        vkMapMemory(device, target.copyBufferMemory, 0, copyBufferSize, 0, &target.copyBufferMap);

        // Create the render target fence
        VkFenceCreateInfo fenceInfo { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, };
        VK_ERR(vkCreateFence(device, &fenceInfo, nullptr, &target.copyFence), "failed to create fence!");

        headlessRenderTargets.push_back(std::move(target));
        renderImageViews.push_back(createImageView(device, target.image, renderImageFormat, VK_IMAGE_ASPECT_COLOR_BIT));
    }
    lastUsedImage = 0;
};

void RenderInstance::cleanupHeadless() {
    for (uint32_t i = 0; i < options::getHeadlessRenderTargetCount(); i++) {
        HeadlessRenderTarget const& target = headlessRenderTargets[i];
        vkDestroyImage(device, target.image, nullptr);
        vkFreeMemory(device, target.imageMemory, nullptr);

        vkDestroyBuffer(device, target.copyBuffer, nullptr);
        vkFreeMemory(device, target.copyBufferMemory, nullptr);

        vkDestroyFence(device, target.copyFence, nullptr);
    }
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

            // Get the most recently released frame to copy
            int lastReleasedImage = lastUsedImage - 1;
            if (lastReleasedImage == -1) {
                lastReleasedImage = options::getHeadlessRenderTargetCount() - 1;
            }
            HeadlessRenderTarget& target = headlessRenderTargets[lastReleasedImage];
            target.copyLatch = std::make_unique<std::latch>(1);

            // Copy the image to the copy buffer once the frame has finished rendering
            VkCommandBufferBeginInfo beginInfo { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, };
            VK_ERR(vkBeginCommandBuffer(target.copyCommandBuffer, &beginInfo), "failed to begin recording command buffer!");

            transitionImageLayout(target.copyCommandBuffer, target.image, renderImageFormat, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            copyImageToBuffer(target.copyCommandBuffer, target.image, target.copyBuffer, renderImageExtent.width, renderImageExtent.height);
            transitionImageLayout(target.copyCommandBuffer, target.image, renderImageFormat, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

            vkEndCommandBuffer(target.copyCommandBuffer);
            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkTimelineSemaphoreSubmitInfo timelineInfo {
                .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                .waitSemaphoreValueCount = 1,
                .pWaitSemaphoreValues = &target.renderingSemaphoreValue,
            };
            VkSubmitInfo submitInfo {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = &timelineInfo,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &target.renderingSemaphore,
                .pWaitDstStageMask = &waitStage,
                .commandBufferCount = 1,
                .pCommandBuffers = &target.copyCommandBuffer,
            };
            vkQueueSubmit(graphicsQueue, 1, &submitInfo, target.copyFence);

            // Spawn a thread to output the copied image to file using the ppm file format
            std::thread writer([&, renderImageExtent = renderImageExtent, outputPath = data.outputPath] {
                // Allocate a thread-specific buffer that we'll copy our image to. It lets us release the main thread earlier
                // Additionally, compacting the image in the memory region given by target.copyBufferMap is slow (I assume because each write then has to be mirrored to the GPU)
                std::unique_ptr<uint8_t[]> imageCopy(new uint8_t[renderImageExtent.width * renderImageExtent.height * 4]);

                // Wait for the copy to finish
                vkWaitForFences(device, 1, &target.copyFence, VK_TRUE, UINT64_MAX);
                vkResetFences(device, 1, &target.copyFence);

                // Copy the image to our thread-specific buffer
                memcpy(imageCopy.get(), target.copyBufferMap, renderImageExtent.width * renderImageExtent.height * 4);

                // Signal that we are done using the image to the latch
                target.copyLatch.value()->count_down();

                std::ofstream outputImage(outputPath, std::ios::out | std::ios::binary);

                // Write the PPM header
                std::string header = string_format("P6 %d %d 255\n", renderImageExtent.width, renderImageExtent.height);
                outputImage.write(header.c_str(), header.size());

                // Write the image in PPM format
                // Since the image is RGBA, while PPM only takes RGB, we need to first compact the image down
                for (uint32_t p = 0; p < renderImageExtent.width * renderImageExtent.height; p++) {
                    memcpy(imageCopy.get() + p * 3, imageCopy.get() + p * 4, 3);
                }
                outputImage.write(reinterpret_cast<const char*>(imageCopy.get()), renderImageExtent.width * renderImageExtent.height * 3);
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
    HeadlessRenderTarget& target = headlessRenderTargets[lastUsedImage];
    if (target.renderingSemaphore != VK_NULL_HANDLE) {
        VkSemaphoreWaitInfo waitInfo {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores = &target.renderingSemaphore,
            .pValues = &target.renderingSemaphoreValue,
        };
        vkWaitSemaphores(device, &waitInfo, UINT64_MAX);
    }

    // If the image is being copied to disk, wait until the image is free to be rendered to
    if (target.copyLatch.has_value()) {
        target.copyLatch.value()->wait();
        target.copyLatch = std::nullopt; // Reset the latch
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
    lastUsedImage = (lastUsedImage + 1) % options::getHeadlessRenderTargetCount();
    return RI_TARGET_OK;
}

RenderInstanceImageStatus RenderInstance::presentImageHeadless(VkSemaphore renderFinishedSemaphore, uint64_t semaphoreCurVal, uint32_t imageIndex) {
    HeadlessRenderTarget& target = headlessRenderTargets[imageIndex];
    target.renderingSemaphore = renderFinishedSemaphore;
    target.renderingSemaphoreValue = semaphoreCurVal + 1;
    return RI_TARGET_OK;
}
