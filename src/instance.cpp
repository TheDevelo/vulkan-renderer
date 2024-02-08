#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <vector>
#include <cstring>

#include "instance.hpp"
#include "options.hpp"
#include "util.hpp"

// Desired validation layers/device extensions
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

// Vulkan extension wrapper functions
VkResult UVkCreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void UVkDestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// Debug callback for our validation layers
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData
) {
    // The \33[1;xxm stuff sets colors in linux terminals, so that our debug output is more readable
    std::cerr << "\33[1;32m[VULKAN]";
    if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        std::cerr << "\33[1;32m [INFO]\33[0m ";
    }
    else if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "\33[1;33m [WARN]\33[0m ";
    }
    else if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        std::cerr << "\33[1;31m [ERR ]\33[0m ";
    }
    else if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
        std::cerr << "\33[1;34m [VERB]\33[0m ";
    }
    else {
        std::cerr << "\33[0m ";
    }
    std::cerr << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

// Forward declared helper-functions
bool checkValidationLayerSupport();
std::vector<const char*> getRequiredExtensions();
bool checkDeviceExtensionSupport(VkPhysicalDevice device);

// Render instance constructor
RenderInstance::RenderInstance(RenderInstanceOptions const& opts) {
    initRealWindow();

    initVulkanInstance();
    initVulkanSurface();
    initVulkanDevice();

    createRealSwapChain();

    createCommandPool();
};

// Cleanup destructor for our render instance
RenderInstance::~RenderInstance() {
    cleanupRealSwapChain();

    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyDevice(device, nullptr);
    if (options::isValidationEnabled()) {
        UVkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);
    glfwTerminate();
}

// Create our instance along with our debugging layer if enabled
void RenderInstance::initVulkanInstance() {
    VkApplicationInfo appInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "VK Renderer",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "Fuchsian Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0,
    };

    // Need to get extensions required for Vulkan before we can request our instance
    std::vector<const char*> extensions = getRequiredExtensions();

    VkInstanceCreateInfo instanceCreateInfo {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = 0,
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    // Create info for our debug callback. Need it now since we also pass it into our instanceCreateInfo as pNext
    VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
    };

    if (options::isValidationEnabled()) {
        if (!checkValidationLayerSupport()) {
            PANIC("validation layers requested, but not available!");
        }

        instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();
        instanceCreateInfo.pNext = &debugMessengerCreateInfo;
    }

    VK_ERR(vkCreateInstance(&instanceCreateInfo, nullptr, &instance), "failed to create VK instance!");

    // Create our vulkan validation layer callback if enabled
    if (options::isValidationEnabled()) {
        VK_ERR(UVkCreateDebugUtilsMessengerEXT(instance, &debugMessengerCreateInfo, nullptr, &debugMessenger), "failed to set up debug messenger!");
    }
}

std::vector<const char*> getRequiredExtensions() {
    // Use GLFW extension list as a base. Once we replace with X11, then we won't need GLFW extensions.
    uint32_t glfwExtCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtCount);

    if (options::isValidationEnabled()) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    std::set<std::string> requiredLayers(validationLayers.begin(), validationLayers.end());

    for (const VkLayerProperties& layerProps : availableLayers) {
        requiredLayers.erase(layerProps.layerName);
    }

    return requiredLayers.empty();
}

// Create the surface for our window
void RenderInstance::initVulkanSurface() {
    /// Will replace with X11 specific code later...
    VK_ERR(glfwCreateWindowSurface(instance, window, nullptr, &surface), "failed to create window surface!");
}

// Select our physical device, and create our logical device along with any associated queues
void RenderInstance::initVulkanDevice() {
    // Grab the list of possible physical devices
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        PANIC("failed to find GPUs with Vulkan support!");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // List devices if requested
    if (options::listDevices()) {
        VkPhysicalDeviceProperties deviceProps;
        uint32_t n = 1;
        for (VkPhysicalDevice device : devices) {
            vkGetPhysicalDeviceProperties(device, &deviceProps);
            std::cout << "Device " << n << ": " << deviceProps.deviceName << std::endl;
            n += 1;
        }
    }

    // Pick the first suitable device
    for (const VkPhysicalDevice& device : devices) {
        if (isDeviceSuitable(device)) {
            physicalDevice = device;
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        PANIC("failed to find a suitable GPU!");
    }

    // Grab the indices of the queue families we want for our selected physical device
    QueueFamilyIndices indices = getQueueFamilies();

    // Setup the creation info for our queues. Note that we need to make sure we don't double up on creating queues.
    // Our indices aren't guaranteed to be unique, which is why we first filter them through a set to get the unique queue indices.
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queueFamily,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority, // This expects an array, but we only have 1 queue, so 1 float makes a long enough array :)
        };
        queueCreateInfos.push_back(queueCreateInfo);
    }

    // Create our logical device
    VkPhysicalDeviceFeatures deviceFeatures {
        .samplerAnisotropy = VK_TRUE,
    };
    VkDeviceCreateInfo deviceCreateInfo {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledLayerCount = 0,
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
        .pEnabledFeatures = &deviceFeatures,
    };

    // Enable device-specific validation layers. This only applies to older Vulkan implementations.
    if (options::isValidationEnabled()) {
        deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        deviceCreateInfo.ppEnabledLayerNames = validationLayers.data();
    }

    VK_ERR(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device), "failed to create the logical device!");

    // Grab the queue handles from our logical device
    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
}

// Make sure that our device is suitable for our purposes.
bool RenderInstance::isDeviceSuitable(VkPhysicalDevice device) {
    // Grab our device properties
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    // Also grab the queue families available for our device
    QueueFamilyIndices indices = findQueueFamilies(device);

    // Check if we have all our device extensions supported
    if (!checkDeviceExtensionSupport(device)) {
        return false;
    }

    // Make sure we have an adequate swap chain available
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
    if (swapChainSupport.formats.empty() || swapChainSupport.presentModes.empty()) {
        return false;
    }

    // Make sure that we have the queue families that we want for our renderer
    if (!indices.isComplete()) {
        return false;
    }

    // Make sure we have sampler anisotropy available
    if (deviceFeatures.samplerAnisotropy != VK_TRUE) {
        return false;
    }

    if (options::getDevice().has_value()) {
        // Only return true if the device name matches the one specified
        return strcmp(deviceProperties.deviceName, options::getDevice().value().c_str()) == 0;
    }
    else {
        // In the case a device is not specified, only approve of a device if it is a dGPU
        // Is this a good idea in the real world where not everyone has a dGPU? No
        // Am I doing this because my laptop's iGPU comes before my dGPU, and I don't want to type out my dGPU name? Yes
        return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    }
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const VkExtensionProperties& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

// Find the queue families with the desired capabilities
QueueFamilyIndices RenderInstance::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        // Check for a queue that supports graphics
        const VkQueueFamilyProperties& queueFamily = queueFamilies[i];
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }

        // Check for a queue that supports presentation
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (presentSupport) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }
    }

    return indices;
}

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
    // TODO: EXTEND FOR HEADLESS!!!
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

inline QueueFamilyIndices RenderInstance::getQueueFamilies() {
    return findQueueFamilies(physicalDevice);
}

// TODO: Do acquireImage/presentImage for headless mode...
RenderInstanceImageStatus RenderInstance::acquireImage(VkSemaphore availableSemaphore, uint32_t& dstImageIndex) {
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

RenderInstanceImageStatus RenderInstance::presentImage(VkSemaphore renderFinishedSemaphore, uint32_t imageIndex) {
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
    for (VkImageView imageView : renderImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }
    vkDestroySwapchainKHR(device, swapChain, nullptr);
}

void RenderInstance::createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = getQueueFamilies();

    // Create the command pool
    VkCommandPoolCreateInfo poolInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
    };

    VK_ERR(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool), "failed to create command pool!");
}
