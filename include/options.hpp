#pragma once
#include <optional>
#include <string>

#include "scene.hpp"

namespace options {
    void parse(int argc, char** argv);

    std::string getScenePath();
    std::optional<std::string> getDefaultCamera();
    std::optional<std::string> getDevice();
    bool listDevices();
    bool isValidationEnabled();
    uint32_t getWindowWidth();
    uint32_t getWindowHeight();
    bool logFrameTimes();
    bool isHeadless();
    std::string getHeadlessEventsPath();
    CullingMode getDefaultCullingMode();
    uint32_t getHeadlessRenderTargetCount();
}
