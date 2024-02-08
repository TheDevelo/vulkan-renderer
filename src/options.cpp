#include <charconv>
#include <vector>

#include "options.hpp"
#include "util.hpp"

// Command option variables
static std::string scenePath;
static std::optional<std::string> cameraName;
static std::optional<std::string> deviceName;
static bool listDevicesBool = false;
static bool enableValidation = false;
static uint32_t windowWidth = 1280;
static uint32_t windowHeight = 960;

namespace options {
    void parse(int argc, char** argv) {
        int currentIndex = 1;
        const std::vector<std::string_view> args(argv + 0, argv + argc);

        // Variables to keep track of required arguments
        bool setScenePath = false;

        while (currentIndex < argc) {
            std::string_view currentArg = args[currentIndex];
            if (currentArg == "--scene") {
                currentIndex += 1;
                if (currentIndex >= argc) {
                    PANIC("missing argument to --scene");
                }

                scenePath = args[currentIndex];
                setScenePath = true;
            }
            else if (currentArg == "--camera") {
                currentIndex += 1;
                if (currentIndex >= argc) {
                    PANIC("missing argument to --camera");
                }

                cameraName = args[currentIndex];
            }
            else if (currentArg == "--physical-device") {
                currentIndex += 1;
                if (currentIndex >= argc) {
                    PANIC("missing argument to --physical-device");
                }

                deviceName = args[currentIndex];
            }
            else if (currentArg == "--list-physical-devices") {
                listDevicesBool = true;
            }
            else if (currentArg == "--enable-validation") {
                enableValidation = true;
            }
            else if (currentArg == "--window-size") {
                currentIndex += 2;
                if (currentIndex >= argc) {
                    PANIC("missing argument to --window-size");
                }

                std::string_view xStr = args[currentIndex - 1];
                std::string_view yStr = args[currentIndex];
                auto xResult = std::from_chars(xStr.data(), xStr.data() + xStr.size(), windowWidth);
                auto yResult = std::from_chars(yStr.data(), yStr.data() + yStr.size(), windowHeight);
                if (xResult.ec == std::errc::invalid_argument || xResult.ec == std::errc::result_out_of_range ||
                    yResult.ec == std::errc::invalid_argument || yResult.ec == std::errc::result_out_of_range) {
                    PANIC("invalid argument to --window-size");
                }
            }
            else {
                PANIC("invalid command line argument: " + std::string(currentArg));
            }

            currentIndex += 1;
        }

        // Check if we had required arguments
        // Note that --list-physical-devices supercedes all required arguments. If its there, the rest don't matter.
        if (!listDevicesBool) {
            if (!setScenePath) {
                PANIC("missing --scene argument");
            }
        }
    }

    std::string getScenePath() {
        return scenePath;
    }

    std::optional<std::string> getDefaultCamera() {
        return cameraName;
    }

    std::optional<std::string> getDevice() {
        return deviceName;
    }

    bool listDevices() {
        return listDevicesBool;
    };

    bool isValidationEnabled() {
        return enableValidation;
    };

    uint32_t getWindowWidth() {
        return windowWidth;
    }

    uint32_t getWindowHeight() {
        return windowHeight;
    }
}
