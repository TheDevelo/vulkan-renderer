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

std::vector<const char*> deviceExtensions = {
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
std::vector<const char*> getRequiredExtensions(bool lightweight);
bool checkDeviceExtensionSupport(VkPhysicalDevice device);

// Render instance constructor
RenderInstance::RenderInstance(RenderInstanceOptions const& opts) {
    lightweight = opts.lightweight;

    if (!options::isHeadless() && !lightweight) {
        initRealWindow();
    }

    initVulkanInstance();
    if (!options::isHeadless() && !lightweight) {
        initVulkanSurface();
    }
    initVulkanDevice();

    if (!options::isHeadless() && !lightweight) {
        createRealSwapChain();
    }

    createCommandPool();

    if (options::isHeadless() && !lightweight) {
        initHeadless();
    }
};

// Cleanup destructor for our render instance
RenderInstance::~RenderInstance() {
    // Wait for any background tasks to finish before we start destructing resources
    for (std::thread& writer : imageWriters) {
        writer.join();
    }

    if (options::isHeadless() && !lightweight) {
        cleanupHeadless();
    }
    else if (!lightweight) {
        // Clean up the swapchain, surface, and window
        cleanupRealSwapChain();

        vkDestroySurfaceKHR(instance, surface, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    for (VkImageView imageView : renderImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyDevice(device, nullptr);
    if (options::isValidationEnabled()) {
        UVkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
}

// Create our instance along with our debugging layer if enabled
void RenderInstance::initVulkanInstance() {
    VkApplicationInfo appInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "VK Renderer",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "Fuchsian Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_2,
    };

    // Need to get extensions required for Vulkan before we can request our instance
    std::vector<const char*> extensions = getRequiredExtensions(lightweight);

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

std::vector<const char*> getRequiredExtensions(bool lightweight) {
    std::vector<const char*> extensions;

    if (!options::isHeadless() && !lightweight) {
        // Use GLFW extension list as a base. Once we replace with X11, then we won't need GLFW extensions.
        uint32_t glfwExtCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtCount);

        extensions = std::vector<const char*>(glfwExtensions, glfwExtensions + glfwExtCount);
    }

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
    if (!options::isHeadless() && !lightweight) {
        // Add the swapchain extension if not in headless mode
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    VkPhysicalDeviceFeatures deviceFeatures {
        .samplerAnisotropy = VK_TRUE,
    };
    VkPhysicalDeviceVulkan12Features vulkan12Features {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .runtimeDescriptorArray = VK_TRUE,
        .scalarBlockLayout = VK_TRUE,
        .timelineSemaphore = VK_TRUE,
    };
    VkDeviceCreateInfo deviceCreateInfo {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &vulkan12Features,
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

    // Make sure we have an adequate swap chain available if in interactive mode
    if (!options::isHeadless() && !lightweight) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        if (swapChainSupport.formats.empty() || swapChainSupport.presentModes.empty()) {
            return false;
        }
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

        // Check for a queue that supports presentation if in interactive mode.
        if (options::isHeadless() || lightweight) {
            // If in headless/lightweight, just pretend presentQueue = graphicsQueue (fine since we don't need to present anyways)
            indices.presentFamily = indices.graphicsFamily;
        }
        else {
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }
        }

        if (indices.isComplete()) {
            break;
        }
    }

    return indices;
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

// Interaction functions that forward to real/headless
RenderInstanceImageStatus RenderInstance::acquireImage(VkSemaphore availableSemaphore, uint64_t semaphoreCurVal, uint32_t& dstImageIndex) {
    if (options::isHeadless()) {
        return acquireImageHeadless(availableSemaphore, semaphoreCurVal, dstImageIndex);
    }
    else {
        return acquireImageReal(availableSemaphore, dstImageIndex);
    }
}

RenderInstanceImageStatus RenderInstance::presentImage(VkSemaphore renderFinishedSemaphore, uint64_t semaphoreCurVal, uint32_t imageIndex) {
    if (options::isHeadless()) {
        return presentImageHeadless(renderFinishedSemaphore, semaphoreCurVal, imageIndex);
    }
    else {
        return presentImageReal(renderFinishedSemaphore, imageIndex);
    }
}

bool RenderInstance::shouldClose() {
    if (options::isHeadless()) {
        return shouldCloseHeadless();
    }
    else {
        return shouldCloseReal();
    }
}

float RenderInstance::processEvents() {
    if (options::isHeadless()) {
        return processEventsHeadless();
    }
    else {
        return processEventsReal();
    }
}
