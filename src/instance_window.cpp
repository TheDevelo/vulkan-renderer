#include "instance.hpp"
#include "linear.hpp"
#include "options.hpp"

#include <iostream>

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

static void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    RenderInstance* instance = reinterpret_cast<RenderInstance*>(glfwGetWindowUserPointer(window));
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
        });
    }

    // WASD for user camera control
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

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS && mouseCaptured) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        mouseCaptured = false;
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
            .userCameraRotateData = UserCameraRotateEvent {
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
}

bool RenderInstance::shouldClose() {
    return glfwWindowShouldClose(window);
}

void RenderInstance::processWindowEvents() {
    // Empty out the event queue before we repopulate it
    eventQueue.clear();

    // Process GLFW events
    glfwPollEvents();

    // Add any events for held actions
    RenderInstanceEvent cameraMoveEvent {
        .type = RI_EV_USER_CAMERA_MOVE,
        .userCameraMoveData = UserCameraMoveEvent {
            .forwardAmount = 0,
            .sideAmount = 0,
        },
    };
    if (wHeld) {
        cameraMoveEvent.userCameraMoveData.forwardAmount += 1;
    }
    if (aHeld) {
        cameraMoveEvent.userCameraMoveData.sideAmount += 1;
    }
    if (sHeld) {
        cameraMoveEvent.userCameraMoveData.forwardAmount -= 1;
    }
    if (dHeld) {
        cameraMoveEvent.userCameraMoveData.sideAmount -= 1;
    }
    if (cameraMoveEvent.userCameraMoveData.forwardAmount != 0 || cameraMoveEvent.userCameraMoveData.sideAmount != 0) {
        eventQueue.emplace_back(cameraMoveEvent);
    }
}
