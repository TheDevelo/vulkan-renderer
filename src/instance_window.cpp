#include <algorithm>
#include <chrono>

#include "instance.hpp"
#include "linear.hpp"
#include "options.hpp"
#include "util.hpp"

// === Event handling code
static void targetResizeCallback(GLFWwindow* window, int width, int height) {
    RenderInstance* instance = reinterpret_cast<RenderInstance*>(glfwGetWindowUserPointer(window));
    instance->targetResized = true;
}

static bool wHeld = false;
static bool aHeld = false;
static bool sHeld = false;
static bool dHeld = false;
static bool mouseCaptured = false;
static double prevXPos = 0.0f;
static double prevYPos = 0.0f;

// Used for calculating time between processEvents calls
static std::chrono::system_clock::time_point prevTime;

static void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    RenderInstance* instance = reinterpret_cast<RenderInstance*>(glfwGetWindowUserPointer(window));
    // Camera selection controls
    if (key == GLFW_KEY_Z && action == GLFW_PRESS) {
        instance->eventQueue.emplace_back(RenderInstanceEvent {
            .type = RI_EV_USE_USER_CAMERA,
        });
    }
    if (key == GLFW_KEY_X && action == GLFW_PRESS) {
        instance->eventQueue.emplace_back(RenderInstanceEvent {
            .type = RI_EV_USE_DEBUG_CAMERA,
        });
    }
    if (key == GLFW_KEY_C && action == GLFW_PRESS) {
        instance->eventQueue.emplace_back(RenderInstanceEvent {
            .type = RI_EV_SWAP_FIXED_CAMERA,
            .data = SwapFixedCameraEvent {},
        });
    }

    // User camera control
    if (key == GLFW_KEY_W && action != GLFW_REPEAT) {
        wHeld = action == GLFW_PRESS;
    }
    if (key == GLFW_KEY_A && action != GLFW_REPEAT) {
        aHeld = action == GLFW_PRESS;
    }
    if (key == GLFW_KEY_S && action != GLFW_REPEAT) {
        sHeld = action == GLFW_PRESS;
    }
    if (key == GLFW_KEY_D && action != GLFW_REPEAT) {
        dHeld = action == GLFW_PRESS;
    }

    // Mouse capture control for user camera
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS && mouseCaptured) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        mouseCaptured = false;
    }

    // Animation toggle
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        instance->eventQueue.emplace_back(RenderInstanceEvent {
            .type = RI_EV_TOGGLE_ANIMATION,
        });
    }

}

static void glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && GLFW_PRESS && !mouseCaptured) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwGetCursorPos(window, &prevXPos, &prevYPos);
        mouseCaptured = true;
    }
}

static void glfwMouseCursorCallback(GLFWwindow* window, double x, double y) {
    constexpr float DPI = DEG2RADF(180.0f) / 500.0f; // 500 pixels per 180
    RenderInstance* instance = reinterpret_cast<RenderInstance*>(glfwGetWindowUserPointer(window));
    if (mouseCaptured) {
        // Calculate the radians rotated based on screen pixels moved and DPI
        float xyRadians = (prevXPos - x) * DPI;
        float zRadians = (prevYPos - y) * DPI;
        instance->eventQueue.emplace_back(RenderInstanceEvent {
            .type = RI_EV_USER_CAMERA_ROTATE,
            .data = UserCameraRotateEvent {
                .xyRadians = xyRadians,
                .zRadians = zRadians,
            }
        });

        prevXPos = x;
        prevYPos = y;
    }
}

void RenderInstance::initRealWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(options::getWindowWidth(), options::getWindowHeight(), "VKRenderer", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, targetResizeCallback);
    glfwSetKeyCallback(window, glfwKeyCallback);
    glfwSetMouseButtonCallback(window, glfwMouseButtonCallback);
    glfwSetCursorPosCallback(window, glfwMouseCursorCallback);

    prevTime = std::chrono::high_resolution_clock::now();
}

bool RenderInstance::shouldCloseReal() {
    return glfwWindowShouldClose(window);
}

float RenderInstance::processEventsReal() {
    // Empty out the event queue before we repopulate it
    eventQueue.clear();

    // Process GLFW events
    glfwPollEvents();

    // Add any events for held actions
    UserCameraMoveEvent cameraMoveData {
        .forwardAmount = 0,
        .sideAmount = 0,
    };
    if (wHeld) {
        cameraMoveData.forwardAmount += 1;
    }
    if (aHeld) {
        cameraMoveData.sideAmount += 1;
    }
    if (sHeld) {
        cameraMoveData.forwardAmount -= 1;
    }
    if (dHeld) {
        cameraMoveData.sideAmount -= 1;
    }
    RenderInstanceEvent cameraMoveEvent {
        .type = RI_EV_USER_CAMERA_MOVE,
        .data = cameraMoveData,
    };
    if (cameraMoveData.forwardAmount != 0 || cameraMoveData.sideAmount != 0) {
        eventQueue.emplace_back(cameraMoveEvent);
    }

    std::chrono::system_clock::time_point curTime = std::chrono::high_resolution_clock::now();
    float processTime = std::chrono::duration<float, std::chrono::seconds::period>(curTime - prevTime).count();
    prevTime = curTime;
    return processTime;
}

// === Real surface/swapchain code
SwapChainSupportDetails RenderInstance::querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;

    // Fill out our swap chain details
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

// Create the swap chain and retrieve its respective images
void RenderInstance::createRealSwapChain() {
    // Query for the swap chain format, present mode, and extent we want
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    renderImageFormat = surfaceFormat.format;
    renderImageExtent = extent;

    // Create the swap chain
    VkSwapchainCreateInfoKHR swapchainCreateInfo {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform = swapChainSupport.capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = presentMode,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE,
    };

    // Set the image sharing mode based on whether the graphics and present queues are different
    QueueFamilyIndices indices = getQueueFamilies();
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };
    if (indices.graphicsFamily != indices.presentFamily) {
        swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchainCreateInfo.queueFamilyIndexCount = 2;
        swapchainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else {
        swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swapchainCreateInfo.queueFamilyIndexCount = 0;
        swapchainCreateInfo.pQueueFamilyIndices = nullptr;
    }

    VK_ERR(vkCreateSwapchainKHR(device, &swapchainCreateInfo, nullptr, &swapChain), "failed to create swap chain!");

    // Retreive the image handles and create their respective image views
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    renderImageViews.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
        renderImageViews[i] = createImageView(device, swapChainImages[i], renderImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

VkSurfaceFormatKHR RenderInstance::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    // Select the ideal format based on SRGB support
    for (const VkSurfaceFormatKHR& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    // Backup case: just return the first format.
    return availableFormats[0];
}

VkPresentModeKHR RenderInstance::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availableModes) {
    // Find if triple buffering is available as a mode.
    for (const VkPresentModeKHR& availableMode : availableModes) {
        if (availableMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availableMode;
        }
    }

    // If not, just use standard VSYNC
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D RenderInstance::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    // Check that we have options for our swap extent. If the width is not the max uint32_t, then we must use capabilities.currentExtent.
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    // Get the window size in pixels, irregardles of the screen scaling
    // Replace with X11 code later...
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D actualExtent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
    };
    actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return actualExtent;
}

// Create the surface for our window
void RenderInstance::initVulkanSurface() {
    /// Will replace with X11 specific code later...
    VK_ERR(glfwCreateWindowSurface(instance, window, nullptr, &surface), "failed to create window surface!");
}

// This is only for real windows. I don't know why we would ever need to recreate our fake swapchain
void RenderInstance::recreateSwapChain() {
    // In case of minimization, wait until we are unminimized.
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    // We wait until all rendering is finished before we switch the swapchain out
    // We could instead recreate the swapchain using the old swapchain to swap on the fly, but its easier this way.
    vkDeviceWaitIdle(device);

    cleanupRealSwapChain();
    createRealSwapChain();
}

void RenderInstance::cleanupRealSwapChain() {
    vkDestroySwapchainKHR(device, swapChain, nullptr);
}

RenderInstanceImageStatus RenderInstance::acquireImageReal(VkSemaphore availableSemaphore, uint32_t& dstImageIndex) {
    VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, availableSemaphore, VK_NULL_HANDLE, &dstImageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return RI_TARGET_REBUILD;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        return RI_TARGET_FAILURE;
    }
    return RI_TARGET_OK;
}

RenderInstanceImageStatus RenderInstance::presentImageReal(VkSemaphore renderFinishedSemaphore, uint32_t imageIndex) {
    VkSwapchainKHR swapChains[] = { swapChain };
    VkPresentInfoKHR presentInfo {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &renderFinishedSemaphore,
        .swapchainCount = 1,
        .pSwapchains = swapChains,
        .pImageIndices = &imageIndex,
    };
    VkResult result = vkQueuePresentKHR(presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || targetResized) {
        targetResized = false;
        recreateSwapChain();
        return RI_TARGET_REBUILD;
    } else if (result != VK_SUCCESS) {
        return RI_TARGET_FAILURE;
    }
    return RI_TARGET_OK;
}
