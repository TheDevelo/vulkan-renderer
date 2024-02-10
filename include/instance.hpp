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

// Event types and data structs
struct UserCameraMoveEvent {
    int forwardAmount; // +1 for forward, -1 for backward
    int sideAmount; // +1 for left, -1 for right
};

struct UserCameraRotateEvent {
    float xyRadians;
    float zRadians;
};

struct SetAnimationEvent {
    float time;
    float rate;
};

enum RenderInstanceEventType {
    RI_EV_SWAP_FIXED_CAMERA, // Also will switch to fixed camera from user/debug camera
    RI_EV_USE_USER_CAMERA,
    RI_EV_USE_DEBUG_CAMERA,
    RI_EV_USER_CAMERA_MOVE,
    RI_EV_USER_CAMERA_ROTATE,
    RI_EV_TOGGLE_ANIMATION,
    RI_EV_SET_ANIMATION,
    RI_EV_INTERNAL_AVAILABLE,
};

struct RenderInstanceEvent {
    RenderInstanceEventType type;
    union {
        UserCameraMoveEvent userCameraMoveData;
        UserCameraRotateEvent userCameraRotateData;
        SetAnimationEvent setAnimationData;
    };
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

    VkCommandPool commandPool;

    std::vector<VkImageView> renderImageViews;
    VkFormat renderImageFormat;
    VkExtent2D renderImageExtent;

    bool targetResized = false;

    // Constructor/Destructor
    RenderInstance(RenderInstanceOptions const& opts);
    ~RenderInstance();

    // Acquire/Present a render image
    // acquireImage takes in a semaphore that will be signaled once the dstImageIndex is available
    RenderInstanceImageStatus acquireImage(VkSemaphore availableSemaphore, uint64_t semaphoreCurVal, uint32_t& dstImageIndex);
    // presentImage takes in a semaphore to wait on before we can present the image
    RenderInstanceImageStatus presentImage(VkSemaphore renderFinishedSemaphore, uint64_t semaphoreCurVal, uint32_t imageIndex);

    // Windowing events
    bool shouldClose();
    void processEvents();
    std::vector<RenderInstanceEvent> eventQueue;

    inline QueueFamilyIndices getQueueFamilies() {
        return findQueueFamilies(physicalDevice);
    };

private:
    // Real window state
    GLFWwindow* window;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;

    // Real window functions
    void initRealWindow();
    void createRealSwapChain();
    void recreateSwapChain();
    void cleanupRealSwapChain();

    bool shouldCloseReal();
    void processEventsReal();
    RenderInstanceImageStatus acquireImageReal(VkSemaphore availableSemaphore, uint32_t& dstImageIndex);
    RenderInstanceImageStatus presentImageReal(VkSemaphore renderFinishedSemaphore, uint32_t imageIndex);

    // Helper for real window inits
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availableModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    // Fake window state
    // Would have liked to use CombinedImage for these, but it requires a shared_ptr to RenderInstance. So manual management it is.
    std::vector<VkImage> headlessRenderImages;
    std::vector<VkDeviceMemory> headlessRenderImagesMemory;

    std::vector<VkSemaphore> renderingSemaphores;
    std::vector<uint64_t> renderingSemaphoreValues; // The timeline semaphore value we are waiting for

    uint32_t lastUsedImage;

    VkBuffer imageCopyBuffer;
    VkDeviceMemory imageCopyBufferMemory;
    void* imageCopyBufferMap;

    // Fake window (headless) functions
    void initHeadless();
    void cleanupHeadless();
    bool shouldCloseHeadless();
    void processEventsHeadless();
    RenderInstanceImageStatus acquireImageHeadless(VkSemaphore availableSemaphore, uint64_t semaphoreCurVal, uint32_t& dstImageIndex);
    RenderInstanceImageStatus presentImageHeadless(VkSemaphore renderFinishedSemaphore, uint64_t semaphoreCurVal, uint32_t imageIndex);

    // Vulkan instance init functions
    void initVulkanInstance();
    void initVulkanSurface();
    void initVulkanDevice();
    void createCommandPool();

    // Helper init functions
    bool isDeviceSuitable(VkPhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
};
