#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <latch>
#include <optional>
#include <thread>
#include <variant>
#include <vector>

// Options for the render instance constructor
struct RenderInstanceOptions {
    bool lightweight;
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
struct SwapFixedCameraEvent {
    std::optional<std::string> name;
};

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

enum class CullingMode; // Forward declaration
struct ChangeCullingEvent {
    CullingMode cullingMode;
};

struct ExposureEvent {
    bool setOrMultiply; // true = multiply, false = set
    float exposure;
};

struct InternalSaveEvent {
    std::string outputPath;
};

struct InternalMarkEvent {
    std::string message;
};

enum RenderInstanceEventType {
    RI_EV_SWAP_FIXED_CAMERA, // Also will switch to fixed camera from user/debug camera
    RI_EV_USE_USER_CAMERA,
    RI_EV_USE_DEBUG_CAMERA,
    RI_EV_USER_CAMERA_MOVE,
    RI_EV_USER_CAMERA_ROTATE,
    RI_EV_TOGGLE_ANIMATION,
    RI_EV_SET_ANIMATION,
    RI_EV_CHANGE_CULLING,
    RI_EV_EXPOSURE,
    RI_EV_INTERNAL_AVAILABLE,
    RI_EV_INTERNAL_SAVE,
    RI_EV_INTERNAL_MARK,
};

struct RenderInstanceEvent {
    RenderInstanceEventType type;
    float timestamp; // The timestamp of when the event happened, for headless mode. In microseconds
    std::variant<
        SwapFixedCameraEvent,
        UserCameraMoveEvent,
        UserCameraRotateEvent,
        SetAnimationEvent,
        ChangeCullingEvent,
        ExposureEvent,
        InternalSaveEvent,
        InternalMarkEvent
    > data;
};

// Container for a headless render target, as well as the needed machinery to copy the image to disk in parallel
struct HeadlessRenderTarget {
    // The render target itself
    VkImage image;
    VkDeviceMemory imageMemory;

    // Buffer and command buffer used to copy the image to CPU
    VkCommandBuffer copyCommandBuffer;
    VkBuffer copyBuffer;
    VkDeviceMemory copyBufferMemory;
    void* copyBufferMap;

    // Synchronization
    VkFence copyFence;
    VkSemaphore renderingSemaphore;
    uint64_t renderingSemaphoreValue; // The timeline semaphore value we are waiting for
    std::optional<std::unique_ptr<std::latch>> copyLatch;
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
    float processEvents(); // Return value is time since last call to processEvents
    std::vector<RenderInstanceEvent> eventQueue;

    // Determines whether we have a lightweight instance (one without a windowing system, whether real or headless)
    bool lightweight;

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
    float processEventsReal();
    RenderInstanceImageStatus acquireImageReal(VkSemaphore availableSemaphore, uint32_t& dstImageIndex);
    RenderInstanceImageStatus presentImageReal(VkSemaphore renderFinishedSemaphore, uint32_t imageIndex);

    // Helper for real window inits
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availableModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    // Fake window state
    // Would have liked to use CombinedImage for these, but it requires a shared_ptr to RenderInstance. So manual management it is.
    std::vector<RenderInstanceEvent> headlessEvents;
    uint32_t currentHeadlessEvent;

    std::vector<HeadlessRenderTarget> headlessRenderTargets;
    uint32_t lastUsedImage;
    std::vector<std::thread> imageWriters;

    // Fake window (headless) functions
    void initHeadless();
    void cleanupHeadless();
    bool shouldCloseHeadless();
    float processEventsHeadless();
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
