#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include <cstdlib>
#include <cstring>

#include "linear.hpp"

// Error handling macro
#define VK_ERR(res, msg) if (res != VK_SUCCESS) { throw std::runtime_error(msg); }

const uint32_t WINDOW_W = 800;
const uint32_t WINDOW_H = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

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

// Container struct for queue family indices
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// Container struct for swap chain information
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

// Container struct for buffer copy regions
struct BufferCopy {
    VkBuffer srcBuffer;
    VkDeviceSize srcOffset;
    VkBuffer dstBuffer;
    VkDeviceSize dstOffset;
    VkDeviceSize size;
};

// Helper function to read files into a vector of chars
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

// Vertex struct for Vulkan
struct Vertex {
    Vec2<float> pos;
    Vec3<float> color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription {
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions {{
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, pos),
            },
            {
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, color),
            },
        }};

        return attributeDescriptions;
    }
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}},
    {{0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
};

const std::vector<uint16_t> indices = {
    0, 1, 2, 0, 2, 3
};

// MVP matrices for the vertex shader
struct MVPMatrices {
    Mat4<float> model;
    Mat4<float> viewProj;
};

class VKRendererApp {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;

    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMaps;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    uint32_t currentFrame = 0;
    bool framebufferResized = false;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WINDOW_W, WINDOW_H, "VKRenderer", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    void initVulkan() {
        initVulkanInstance();
        initVulkanSurface();
        initVulkanDevice();

        createSwapChain();

        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();

        createFramebuffers();

        createCommandBuffers();

        createVertexBuffer();
        createUniformBuffers();
        createDescriptorSets();

        createSyncObjects();
    }

    // Create our instance along with our debugging layer if enabled
    void initVulkanInstance() {
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

        if (enableValidationLayers) {
            if (!checkValidationLayerSupport()) {
                throw std::runtime_error("validation layers requested, but not available!");
            }

            instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();
            instanceCreateInfo.pNext = &debugMessengerCreateInfo;
        }

        VK_ERR(vkCreateInstance(&instanceCreateInfo, nullptr, &instance), "failed to create VK instance!");

        // Create our vulkan validation layer callback if enabled
        if (enableValidationLayers) {
            VK_ERR(UVkCreateDebugUtilsMessengerEXT(instance, &debugMessengerCreateInfo, nullptr, &debugMessenger), "failed to set up debug messenger!");
        }
    }

    std::vector<const char*> getRequiredExtensions() {
        // Use GLFW extension list as a base. Once we replace with X11, then we won't need GLFW extensions.
        uint32_t glfwExtCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtCount);

        if (enableValidationLayers) {
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
    void initVulkanSurface() {
        /// Will replace with X11 specific code later...
        VK_ERR(glfwCreateWindowSurface(instance, window, nullptr, &surface), "failed to create window surface!");
    }

    // Select our physical device, and create our logical device along with any associated queues
    void initVulkanDevice() {
        // Grab the list of possible physical devices
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // Pick the first suitable device
        for (const VkPhysicalDevice& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }

        // Grab the indices of the queue families we want for our selected physical device
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

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
        VkPhysicalDeviceFeatures deviceFeatures {}; // Empty for now, but we can specify desired device features in this struct.
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
        if (enableValidationLayers) {
            deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            deviceCreateInfo.ppEnabledLayerNames = validationLayers.data();
        }

        VK_ERR(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device), "failed to create the logical device!");

        // Grab the queue handles from our logical device
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    // Make sure that our device is suitable for our purposes.
    bool isDeviceSuitable(VkPhysicalDevice device) {
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

        // Lastly, select discrete GPUs only so that we get good performance
        // This is a bad idea for a real world application, as not all computers have a DGPU
        // However, since I have one and the test cluster will have one, its fine to do this to filter out my iGPU
        return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
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
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
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

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
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
    void createSwapChain() {
        // Query for the swap chain format, present mode, and extent we want
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

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
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
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
        swapChainImageViews.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo imageViewCreateInfo {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = swapChainImages[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = swapChainImageFormat,
                .components = VkComponentMapping {
                    .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                },
                .subresourceRange = VkImageSubresourceRange {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };

            VK_ERR(vkCreateImageView(device, &imageViewCreateInfo, nullptr, &swapChainImageViews[i]), "failed to create image views!");
        }
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        // Select the ideal format based on SRGB support
        for (const VkSurfaceFormatKHR& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        // Backup case: just return the first format.
        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availableModes) {
        // Find if triple buffering is available as a mode.
        for (const VkPresentModeKHR& availableMode : availableModes) {
            if (availableMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availableMode;
            }
        }

        // If not, just use standard VSYNC
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
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

    // Creates the render pass for our renderer
    void createRenderPass() {
        // The attachment describes our screen's framebuffer, and how we want the render pass to modify it.
        VkAttachmentDescription colorAttachment {
            .format = swapChainImageFormat,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };
        VkAttachmentReference colorAttachmentRef {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        // Define the subpass for rendering our triangle
        VkSubpassDescription subpassDesc {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef,
        };
        VkSubpassDependency dependency {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        };

        // Create the whole render pass
        VkRenderPassCreateInfo renderPassInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &colorAttachment,
            .subpassCount = 1,
            .pSubpasses = &subpassDesc,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };

        VK_ERR(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass), "failed to create render pass!");
    }

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding mvpLayoutBinding {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        };

        VkDescriptorSetLayoutCreateInfo layoutInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &mvpLayoutBinding,
        };

        VK_ERR(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout), "failed to create descriptor set layout!");
    }

    // Creates the basic graphics pipeline for our renderer
    void createGraphicsPipeline() {
        // Load in the vertex and fragment shaders
        std::vector<char> vertShaderCode = readFile("shaders/vert.spv");
        std::vector<char> fragShaderCode = readFile("shaders/frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        // Create the shader stages for our pipeline
        VkPipelineShaderStageCreateInfo vertShaderStageInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertShaderModule,
            .pName = "main",
        };
        VkPipelineShaderStageCreateInfo fragShaderStageInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragShaderModule,
            .pName = "main",
        };
        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        // Create info used in our render pipeline about the various operations
        // Dynamic states deal with what we can change after pipeline creation.
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
        };
        VkPipelineDynamicStateCreateInfo dynamicStateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data(),
        };
        // Vertex input describes how vertex data will be passed into our shader
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vertexInputInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data(),
        };
        // Input assembly describes what primitives we make from our vertex data
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };
        // Viewport and scissor describe what regions of our scene and screen we should render
        VkPipelineViewportStateCreateInfo viewportStateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount = 1,
        };
        // Rasterizer determines how we convert from primitives to fragments
        VkPipelineRasterizationStateCreateInfo rasterizerStateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f,
        };
        // Multisampling determines anti-aliasing configuration
        VkPipelineMultisampleStateCreateInfo multisamplingInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
        };
        // Color blending determines how we mix the pixels onto the framebuffer
        // The attachment is info per framebuffer, while create info is global
        VkPipelineColorBlendAttachmentState colorBlendAttachment {
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
        VkPipelineColorBlendStateCreateInfo colorBlendInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };
        // Pipeline layout determines which uniforms are available to the shaders
        VkPipelineLayoutCreateInfo pipelineLayoutInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout,
        };

        VK_ERR(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout), "failed to create pipeline layout!");

        // Finally create the render pipeline
        VkGraphicsPipelineCreateInfo renderPipelineInfo {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssemblyInfo,
            .pViewportState = &viewportStateInfo,
            .pRasterizationState = &rasterizerStateInfo,
            .pMultisampleState = &multisamplingInfo,
            .pDepthStencilState = nullptr, // Null for now, will eventually fill in once we add our depth buffer
            .pColorBlendState = &colorBlendInfo,
            .pDynamicState = &dynamicStateInfo,
            .layout = pipelineLayout,
            .renderPass = renderPass,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };

        // This call takes in multiple pipelines to create. I'm guessing real apps need a bunch of pipelines, so they made a call to batch compile them.
        // In fact, this is probably where "shader compilation" and "shader caches" come in for games I play. Neat!
        VK_ERR(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &renderPipelineInfo, nullptr, &pipeline), "failed to create graphics pipeline!");

        // Clean up our shader modules now that we are finished with them (the pipeline keeps its own copy)
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo moduleCreateInfo {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.size(),
            .pCode = reinterpret_cast<const uint32_t*>(code.data()),
        };

        VkShaderModule shaderModule;
        VK_ERR(vkCreateShaderModule(device, &moduleCreateInfo, nullptr, &shaderModule), "failed to create shader module!");

        return shaderModule;
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i],
            };

            VkFramebufferCreateInfo framebufferInfo {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = renderPass,
                .attachmentCount = 1,
                .pAttachments = attachments,
                .width = swapChainExtent.width,
                .height = swapChainExtent.height,
                .layers = 1,
            };

            VK_ERR(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]), "failed to create framebuffer!");
        }
    }

    // Create the command pool and buffer to send rendering commands to the GPU
    void createCommandBuffers() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        // Create the command pool
        VkCommandPoolCreateInfo poolInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
        };

        VK_ERR(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool), "failed to create command pool!");

        // Allocate the command buffers
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo commandBufferAllocInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = static_cast<uint32_t>(commandBuffers.size()),
        };

        VK_ERR(vkAllocateCommandBuffers(device, &commandBufferAllocInfo, commandBuffers.data()), "failed to allocate command buffers!");
    }

    // Create the vertex buffer AND index buffer for our triangle
    void createVertexBuffer() {
        VkDeviceSize vertexSize = sizeof(vertices[0]) * vertices.size();
        createBuffer(vertexSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
        VkDeviceSize indexSize = sizeof(indices[0]) * indices.size();
        createBuffer(indexSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        // Staging buffer will contain both our data for the vertex and index buffer. We'll then copy both simultaneously.
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(vertexSize + indexSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, vertexSize + indexSize, 0, &data);
        memcpy(data, vertices.data(), (size_t) vertexSize);
        // Need to static cast to uint8_t* since we can't do pointer arithmetic on void*. uint8_t* works because vertexSize is in bytes.
        memcpy(static_cast<uint8_t*>(data) + vertexSize, indices.data(), (size_t) vertexSize);
        vkUnmapMemory(device, stagingBufferMemory);

        BufferCopy bufferCopyInfos[] = {
            {
                .srcBuffer = stagingBuffer,
                .srcOffset = 0,
                .dstBuffer = vertexBuffer,
                .dstOffset = 0,
                .size = vertexSize,
            },
            {
                .srcBuffer = stagingBuffer,
                .srcOffset = vertexSize,
                .dstBuffer = indexBuffer,
                .dstOffset = 0,
                .size = indexSize,
            },
        };
        copyBuffers(bufferCopyInfos, 2);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps, VkBuffer& buffer, VkDeviceMemory& memory) {
        VkBufferCreateInfo bufferInfo {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };

        VK_ERR(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer), "failed to create buffer!");

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, memProps),
        };

        VK_ERR(vkAllocateMemory(device, &allocInfo, nullptr, &memory), "failed to allocate buffer memory!");

        vkBindBufferMemory(device, buffer, memory, 0);
    }

    void copyBuffers(BufferCopy* bufferCopyInfos, uint32_t bufferCopyCount) {
        VkCommandBufferAllocateInfo commandBufferAllocInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        VkCommandBuffer commandBuffer;
        VK_ERR(vkAllocateCommandBuffers(device, &commandBufferAllocInfo, &commandBuffer), "failed to allocate command buffer!");

        VkCommandBufferBeginInfo beginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VK_ERR(vkBeginCommandBuffer(commandBuffer, &beginInfo), "failed to begin command buffer!");

        for (uint32_t i = 0; i < bufferCopyCount; i++) {
            BufferCopy& copyInfo = bufferCopyInfos[i];

            VkBufferCopy copyCmd {
                .srcOffset = copyInfo.srcOffset,
                .dstOffset = copyInfo.dstOffset,
                .size = copyInfo.size,
            };
            vkCmdCopyBuffer(commandBuffer, copyInfo.srcBuffer, copyInfo.dstBuffer, 1, &copyCmd);
        }

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
        };
        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    // Helper function that gets the memory type we need for allocating a buffer
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(MVPMatrices);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMaps.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMaps[i]);
        }
    }

    void createDescriptorSets() {
        // Create the descriptor pool
        VkDescriptorPoolSize poolSize {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        };

        VkDescriptorPoolCreateInfo poolInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            .poolSizeCount = 1,
            .pPoolSizes = &poolSize,
        };

        VK_ERR(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool), "failed to create descriptor pool!");

        // Allocate the descriptor sets
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            .pSetLayouts = layouts.data(),
        };

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        VK_ERR(vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()), "failed to allocate descriptor sets!");

        // Point our descriptor sets at the underlying uniform buffers
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo bufferInfo {
                .buffer = uniformBuffers[i],
                .offset = 0,
                .range = sizeof(MVPMatrices),
            };

            VkWriteDescriptorSet descriptorWrite {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &bufferInfo,
            };

            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
        }
    }

    // Create the synchronization objects for rendering to our screen
    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, };
        VkFenceCreateInfo fenceInfo {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VK_ERR(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]), "failed to create semaphores!");
            VK_ERR(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]), "failed to create semaphores!");
            VK_ERR(vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]), "failed to create semaphores!");
        }
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            drawFrame();
        }

        // Wait until our device has finished all operations before quitting
        vkDeviceWaitIdle(device);
    }

    void drawFrame() {
        // Wait for the last frame to be drawn. These are array functions, hence the 1
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // Acquire the next image on the swapchain for us to render to
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire next image from swap chain!");
        }

        // Only reset the fence if we are going to be submitting work
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // Record our render commands
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        updateMVPMatrices(currentFrame);

        // Submit our command buffer for rendering!
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        VkSubmitInfo submitInfo {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = waitSemaphores,
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffers[currentFrame],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = signalSemaphores,
        };

        VK_ERR(vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]), "failed to submit draw command buffer!");

        // Present the frame onto the screen
        VkSwapchainKHR swapChains[] = { swapChain };
        VkPresentInfoKHR presentInfo {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = signalSemaphores,
            .swapchainCount = 1,
            .pSwapchains = swapChains,
            .pImageIndices = &imageIndex,
        };
        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // Record the rendering commands onto commandBuffer for rendering to imageIndex
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        // Start the command buffer
        VkCommandBufferBeginInfo beginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0,
        };

        VK_ERR(vkBeginCommandBuffer(commandBuffer, &beginInfo), "failed to begin recording command buffer!");

        // Start the render pass
        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        VkRenderPassBeginInfo renderPassInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = renderPass,
            .framebuffer = swapChainFramebuffers[imageIndex],
            .renderArea = VkRect2D {
                .offset = {0, 0},
                .extent = swapChainExtent,
            },
            .clearValueCount = 1,
            .pClearValues = &clearColor,
        };

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Draw our render pipeline
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkViewport viewport {
            .x = 0.0f,
            .y = 0.0f,
            .width = (float) swapChainExtent.width,
            .height = (float) swapChainExtent.height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        VkRect2D scissor {
            .offset = {0, 0},
            .extent = swapChainExtent,
        };
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkBuffer vertexBuffers[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        // End our render pass and command buffer
        vkCmdEndRenderPass(commandBuffer);

        VK_ERR(vkEndCommandBuffer(commandBuffer), "failed to record command buffer!");
    }

    void updateMVPMatrices(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float totalTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        Mat4<float> id = Mat4<float>{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, totalTime/10.0f - 1.0f, 1.0f};
        Mat4<float> model = linear::rotate(totalTime, Vec3<float>(0.0f, 0.0f, 1.0f));
        Mat4<float> view = linear::lookAt(Vec3<float>(2.0f), Vec3<float>(0.0f), Vec3<float>(0.0f, 0.0f, 1.0f));
        Mat4<float> proj = linear::perspective(DEG2RADF(60.0), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
        MVPMatrices mvp {
            .model = linear::mmul(id, model),
            .viewProj = linear::mmul(proj, view),
        };

        memcpy(uniformBuffersMaps[currentImage], &mvp, sizeof(mvp));
    }

    void recreateSwapChain() {
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

        cleanupSwapChain();

        createSwapChain();
        createFramebuffers();
    }

    void cleanup() {
        cleanupSwapChain();

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        vkDestroyDevice(device, nullptr);
        if (enableValidationLayers) {
            UVkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void cleanupSwapChain() {
        for (VkFramebuffer framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (VkImageView imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

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

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        VKRendererApp* app = reinterpret_cast<VKRendererApp*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }
};

int main() {
    VKRendererApp app;

    Vec2<float> x(1.0, 2.0);
    Vec2<float> y(3.0, 4.0);

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << (x + y).x << " " << (x + y).y << std::endl;
    std::cout << (x - y).x << " " << (x - y).y << std::endl;
    std::cout << (x * y).x << " " << (x * y).y << std::endl;
    std::cout << (x / y).x << " " << (x / y).y << std::endl;

    Vec3<float> a(x, 5.0);
    Vec3<float> b(y, 6.0);

    std::cout << (a * b).x << " " << (a * b).y << " " << (a * b).z << std::endl;

    return EXIT_SUCCESS;
}
