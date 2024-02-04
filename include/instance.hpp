#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <optional>
#include <vector>

// Options for the render instance constructor
struct RenderInstanceOptions {
    bool headless;
};

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

enum RenderInstanceImageStatus {
    RI_TARGET_OK,
    RI_TARGET_REBUILD,
    RI_TARGET_FAILURE,
};

// RenderInstance provides a Vulkan instance/device to render to, a way to acquire/present render targets, and a way to handle windowing events
class RenderInstance {
public:
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    std::vector<VkImageView> renderImageViews;
    VkFormat renderImageFormat;
    VkExtent2D renderImageExtent;

    bool targetResized = false;

    // Constructor/Destructor
    RenderInstance(RenderInstanceOptions const& opts);
    ~RenderInstance();

    // Acquire/Present a render image
    // acquireImage takes in a semaphore that will be signaled once the dstImageIndex is available
    RenderInstanceImageStatus acquireImage(VkSemaphore availableSemaphore, uint32_t& dstImageIndex);
    // presentImage takes in a semaphore to wait on before we can present the image
    RenderInstanceImageStatus presentImage(VkSemaphore renderFinishedSemaphore, uint32_t imageIndex);

    bool shouldClose();
    void processWindowEvents();

    QueueFamilyIndices getQueueFamilies();

private:
    // Real window state
    GLFWwindow* window;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;

    // Real window init functions
    void initRealWindow();
    void createRealSwapChain();
    void recreateSwapChain();
    void cleanupRealSwapChain();

    // Helper for real window inits
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availableModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    // Vulkan instance init functions
    void initVulkanInstance();
    void initVulkanSurface();
    void initVulkanDevice();

    // Helper init functions
    bool isDeviceSuitable(VkPhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
};
