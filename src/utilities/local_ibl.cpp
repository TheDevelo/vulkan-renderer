#include "stb_image.h"
#include "stb_image_write.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include <cstdlib>
#include <cstring>

#include "buffer.hpp"
#include "descriptor.hpp"
#include "instance.hpp"
#include "linear.hpp"
#include "materials.hpp"
#include "options.hpp"
#include "scene.hpp"
#include "util.hpp"

const int MAX_FRAMES_IN_FLIGHT = 2;

class LocalIBLUtility {
public:
    void run(std::string scenePathIn, uint32_t cubeSizeIn, float exposure) {
        scenePath = scenePathIn;
        cubeSize = cubeSizeIn;
        // Init our render instance
        initRenderInstance();

        // Load our scene
        scene = Scene(renderInstance, scenePath);
        scene.cameraInfo.exposure = exposure;

        // Init the rest of our Vulkan primitives
        initVulkan();

        mainLoop();
    }

private:
    std::shared_ptr<RenderInstance> renderInstance;
    Scene scene;
    std::unique_ptr<MaterialPipelines> materialPipelines;
    std::unique_ptr<Descriptors> descriptors;

    std::vector<CombinedImage> renderTargetImages;
    std::unique_ptr<CombinedImage> depthImage;
    std::vector<VkFramebuffer> renderTargetFramebuffers;
    std::vector<VkFramebuffer> shadowMapFramebuffers;

    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkFence> inFlightFences;

    uint32_t currentFrame = 0;
    std::string scenePath;
    uint32_t cubeSize = 0;

    void initVulkan() {
        materialPipelines = std::make_unique<MaterialPipelines>(renderInstance, scene, VK_FORMAT_R32G32B32A32_SFLOAT);

        createCommandBuffers();

        createImages();
        createFramebuffers();

        createSyncObjects();

        descriptors = std::make_unique<Descriptors>(renderInstance, scene, *materialPipelines, MAX_FRAMES_IN_FLIGHT);
    }

    void initRenderInstance() {
        renderInstance = std::make_shared<RenderInstance>(RenderInstanceOptions {
            .lightweight = true,
        });
    }

    void createFramebuffers() {
        // Create render target framebuffers
        renderTargetFramebuffers.resize(renderTargetImages.size());
        for (size_t i = 0; i < renderTargetImages.size(); i++) {
            std::array<VkImageView, 2> attachments = {
                renderTargetImages[i].imageView,
                depthImage->imageView,
            };

            VkFramebufferCreateInfo framebufferInfo {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = materialPipelines->solidRenderPass,
                .attachmentCount = static_cast<uint32_t>(attachments.size()),
                .pAttachments = attachments.data(),
                .width = cubeSize,
                .height = cubeSize * 6,
                .layers = 1,
            };

            VK_ERR(vkCreateFramebuffer(renderInstance->device, &framebufferInfo, nullptr, &renderTargetFramebuffers[i]), "failed to create framebuffer!");
        }

        // Create shadow map framebuffers
        shadowMapFramebuffers.resize(scene.shadowMaps.size());
        // Need to loop over the lights instead of the shadow map targets since we also need the size of the shadow maps
        for (size_t i = 0; i < scene.lights.size(); i++) {
            if (!scene.lights[i].info.useShadowMap) {
                continue;
            }
            size_t shadowMapIndex = scene.lights[i].info.shadowMapIndex;

            std::array<VkImageView, 1> attachments = {
                scene.shadowMaps[shadowMapIndex].imageView,
            };

            VkFramebufferCreateInfo framebufferInfo {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = materialPipelines->shadowRenderPass,
                .attachmentCount = static_cast<uint32_t>(attachments.size()),
                .pAttachments = attachments.data(),
                .width = scene.lights[i].shadowMapSize,
                .height = scene.lights[i].shadowMapSize,
                .layers = 1,
            };

            VK_ERR(vkCreateFramebuffer(renderInstance->device, &framebufferInfo, nullptr, &shadowMapFramebuffers[shadowMapIndex]), "failed to create framebuffer!");
        }
    }

    // Create the command pool and buffer to send rendering commands to the GPU
    void createCommandBuffers() {
        // Allocate the command buffers
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo commandBufferAllocInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = renderInstance->commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = static_cast<uint32_t>(commandBuffers.size()),
        };

        VK_ERR(vkAllocateCommandBuffers(renderInstance->device, &commandBufferAllocInfo, commandBuffers.data()), "failed to allocate command buffers!");
    }

    void createImages() {
        // We create the images with a height of 6 * cubeSize so that we can directly render into our final stacked format
        // We'll then set the appropriate render viewport to render each face to the appropriate region
        depthImage = std::make_unique<CombinedImage>(renderInstance, cubeSize, cubeSize * 6, VK_FORMAT_D32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                                                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);

        renderTargetImages.reserve(scene.environments.size());
        for (size_t i = 0; i < scene.environments.size(); i++) {
            renderTargetImages.emplace_back(renderInstance, cubeSize, cubeSize * 6, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    // Create the synchronization objects for rendering to our screen
    void createSyncObjects() {
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkFenceCreateInfo fenceInfo {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VK_ERR(vkCreateFence(renderInstance->device, &fenceInfo, nullptr, &inFlightFences[i]), "failed to create fences!");
        }
    }

    void mainLoop() {
        // Inject the cameras used to render the cubemaps
        scene.useUserCamera = false;
        scene.useDebugCamera = false;
        scene.selectedCamera = scene.cameras.size();
        Camera& captureCamera = scene.cameras.emplace_back(Camera {
            .name = "LocalIBL Camera",
            .aspectRatio = 1.0f,
            .vFov = DEG2RADF(90.0f),
            .nearZ = 0.1f,
        });

        uint32_t baseNode = scene.nodes.size();
        scene.nodes.emplace_back(Node {
            .name = "+X",
            .translation = Vec3<float>(0.0f),
            .rotation = Vec4<float>(-0.7071068, 0, 0.7071068, 0),
            .scale = Vec3<float>(1.0f),
        });
        scene.nodes.emplace_back(Node {
            .name = "-X",
            .translation = Vec3<float>(0.0f),
            .rotation = Vec4<float>(0.7071068, 0, 0.7071068, 0),
            .scale = Vec3<float>(1.0f),
        });
        scene.nodes.emplace_back(Node {
            .name = "+Y",
            .translation = Vec3<float>(0.0f),
            .rotation = Vec4<float>(0.7071068, 0, 0, 0.7071068),
            .scale = Vec3<float>(1.0f),
        });
        scene.nodes.emplace_back(Node {
            .name = "-Y",
            .translation = Vec3<float>(0.0f),
            .rotation = Vec4<float>(-0.7071068, 0, 0, 0.7071068),
            .scale = Vec3<float>(1.0f),
        });
        scene.nodes.emplace_back(Node {
            .name = "+Z",
            .translation = Vec3<float>(0.0f),
            .rotation = Vec4<float>(-1, 0, 0, 0),
            .scale = Vec3<float>(1.0f),
        });
        scene.nodes.emplace_back(Node {
            .name = "-Z",
            .translation = Vec3<float>(0.0f),
            .rotation = Vec4<float>(0, 0, 1, 0),
            .scale = Vec3<float>(1.0f),
        });
        for (int f = 0; f < 6; f++) {
            scene.nodes[baseNode + f].calculateTransforms();
        }

        // Set the environment and light transforms
        scene.updateEnvironmentTransforms();
        scene.updateLightTransforms();
        for (size_t i = 0; i < scene.environments.size(); i++) {
            uint8_t* dstEnvMap = reinterpret_cast<uint8_t*>(descriptors->environmentUniformMaps[i]);
            memcpy(dstEnvMap, &scene.environments[i].info, sizeof(EnvironmentInfo));
        }
        for (size_t i = 0; i < scene.lights.size(); i++) {
            uint8_t* dstLightMap = reinterpret_cast<uint8_t*>(descriptors->lightStorageMap) + i * sizeof(LightInfo);
            memcpy(dstLightMap, &scene.lights[i].info, sizeof(LightInfo));
        }

        // Prerender our scene's shadow maps
        drawShadowMaps();

        for (size_t i = 0; i < scene.environments.size(); i++) {
            // Skip any global environments
            if (scene.environments[i].info.type == 0) {
                continue;
            }

            for (uint32_t f = 0; f < 6; f++) {
                // Update the camera ancestory to point to our current environment
                captureCamera.ancestors = scene.environments[i].ancestors;
                captureCamera.ancestors.push_back(baseNode + f);
                drawCubeFace(i, f);
            }
        }

        // Wait until our device has finished all operations
        vkDeviceWaitIdle(renderInstance->device);

        // Allocate the copy buffer
        VkDeviceSize copyBufferSize = cubeSize * cubeSize * 6 * 4 * 4;
        CombinedBuffer copyBuffer(renderInstance, copyBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        float* copyBufferMem;
        vkMapMemory(renderInstance->device, copyBuffer.bufferMemory, 0, copyBufferSize, 0, (void**) &copyBufferMem);
        std::vector<float> flippedCube(cubeSize * cubeSize * 6 * 4);

        for (size_t i = 0; i < scene.environments.size(); i++) {
            // Skip any global environments
            if (scene.environments[i].info.type == 0) {
                continue;
            }

            // Copy cubemap to the copy buffer
            VkCommandBuffer commandBuffer = beginSingleUseCBuffer(*renderInstance);
            transitionImageLayout(commandBuffer, renderTargetImages[i].image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            copyImageToBuffer(commandBuffer, renderTargetImages[i].image, copyBuffer.buffer, cubeSize, cubeSize * 6);
            endSingleUseCBuffer(*renderInstance, commandBuffer);

            // Flip each face of the cubemap
            for (uint32_t f = 0; f < 6; f++) {
                uint32_t faceOffset = f * cubeSize * cubeSize * 4;
                for (uint32_t row = 0; row < cubeSize; row++) {
                    uint32_t flippedRow = cubeSize - row - 1;
                    memcpy(flippedCube.data() + faceOffset + flippedRow * cubeSize * 4, copyBufferMem + faceOffset + row * cubeSize * 4, cubeSize * 4 * 4);
                }
            }

            // Save cubemap to file
            // Convert our lambertian cubemap to RGBE format
            std::vector<uint8_t> cubemapOut(cubeSize * cubeSize * 6 * 4);
            convertRGBtoRGBE(flippedCube.data(), cubemapOut.data(), cubeSize * cubeSize * 6);
            std::filesystem::path outputPath = scenePath;
            outputPath.replace_extension(string_format(".localIBL_%d.png", i));
            stbi_write_png(outputPath.c_str(), cubeSize, cubeSize * 6, 4, cubemapOut.data(), cubeSize * 4);
        }
    }

    void drawShadowMaps() {
        VkCommandBuffer commandBuffer = beginSingleUseCBuffer(*renderInstance);

        for (uint32_t i = 0; i < scene.lights.size(); i++) {
            if (!scene.lights[i].info.useShadowMap) {
                continue;
            }
            size_t shadowMapIndex = scene.lights[i].info.shadowMapIndex;
            VkExtent2D shadowMapExtent {
                .width = scene.lights[i].shadowMapSize,
                .height = scene.lights[i].shadowMapSize,
            };

            // Start the render pass
            VkClearValue clearColor {
                .depthStencil = { 1.0f, 0 },
            };
            VkRenderPassBeginInfo renderPassInfo {
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = materialPipelines->shadowRenderPass,
                .framebuffer = shadowMapFramebuffers[shadowMapIndex],
                .renderArea = VkRect2D {
                    .offset = {0, 0},
                    .extent = shadowMapExtent,
                },
                .clearValueCount = 1,
                .pClearValues = &clearColor,
            };
            vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            // Set the camera viewport and scissor
            VkViewport viewport {
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(shadowMapExtent.width),
                .height = static_cast<float>(shadowMapExtent.height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
            VkRect2D scissor {
                .offset = {0, 0},
                .extent = shadowMapExtent,
            };
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            // Draw the scene
            SceneRenderInfo sceneRenderInfo {
                .commandBuffer = commandBuffer,
                .pipelines = *materialPipelines,
                .type = SceneRenderType::SHADOW,

                .cameraDescriptorOffset = currentFrame * static_cast<uint32_t>(sizeof(CameraInfo)),
                .environmentDescriptorOffset = currentFrame * static_cast<uint32_t>(sizeof(EnvironmentInfo)),
                .lightDescriptorOffset = currentFrame * static_cast<uint32_t>(sizeof(LightInfo) * scene.lights.size()),

                .lightIndex = i,
            };
            scene.renderScene(sceneRenderInfo);

            // End our render pass
            vkCmdEndRenderPass(commandBuffer);

            // Transition the shadow map into a format that can be read by the shaders
            transitionImageLayout(commandBuffer, scene.shadowMaps[shadowMapIndex].image, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VkImageSubresourceRange {
                    .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                });
        }

        endSingleUseCBuffer(*renderInstance, commandBuffer);
    }

    void drawCubeFace(uint32_t cubemapIndex, uint32_t face) {
        // Wait for the frame in the currentFrame slot to finish drawing. These are array functions, hence the 1
        vkWaitForFences(renderInstance->device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        vkResetFences(renderInstance->device, 1, &inFlightFences[currentFrame]);

        // Update our camera buffer
        scene.updateCameraTransform(*renderInstance);
        scene.cameraInfo.tonemap = false;
        uint8_t* dstCameraMap = reinterpret_cast<uint8_t*>(descriptors->cameraUniformMap) + currentFrame * sizeof(CameraInfo);
        memcpy(dstCameraMap, &scene.cameraInfo, sizeof(CameraInfo));

        // Record our render commands
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], cubemapIndex, face);

        // Submit our command buffer for rendering!
        VkSubmitInfo submitInfo {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffers[currentFrame],
        };

        VK_ERR(vkQueueSubmit(renderInstance->graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]), "failed to submit draw command buffer!");

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // Record the rendering commands onto commandBuffer for rendering to imageIndex
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t cubemapIndex, uint32_t face) {
        // Start the command buffer
        VkCommandBufferBeginInfo beginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0,
        };

        VK_ERR(vkBeginCommandBuffer(commandBuffer, &beginInfo), "failed to begin recording command buffer!");

        // Start the solid render pass
        std::array<VkClearValue, 2> clearColors;
        clearColors[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearColors[1].depthStencil = {1.0f, 0};
        VkExtent2D renderExtent {
            .width = cubeSize,
            .height = cubeSize,
        };
        VkRenderPassBeginInfo renderPassInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = materialPipelines->solidRenderPass,
            .framebuffer = renderTargetFramebuffers[cubemapIndex],
            .renderArea = VkRect2D {
                .offset = {0, static_cast<int32_t>(cubeSize * face)},
                .extent = renderExtent,
            },
            .clearValueCount = static_cast<uint32_t>(clearColors.size()),
            .pClearValues = clearColors.data(),
        };

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Set the viewport to be the correct face of the cubemap
        // We invert height to flip Y, since we rotated our cameras to be 180 of expected
        VkViewport viewport {
            .x = 0.0f,
            .y = static_cast<float>(cubeSize * face),
            .width = static_cast<float>(cubeSize),
            .height = static_cast<float>(cubeSize),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor {
            .offset = {0, static_cast<int32_t>(cubeSize * face)},
            .extent = renderExtent,
        };
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // Draw the scene
        SceneRenderInfo sceneRenderInfo {
            .commandBuffer = commandBuffer,
            .pipelines = *materialPipelines,
            .type = SceneRenderType::SOLID,

            .cameraDescriptorOffset = currentFrame * static_cast<uint32_t>(sizeof(CameraInfo)),
            .environmentDescriptorOffset = 0,
            .lightDescriptorOffset = 0,
        };
        scene.renderScene(sceneRenderInfo);

        // End our render pass and command buffer
        vkCmdEndRenderPass(commandBuffer);

        VK_ERR(vkEndCommandBuffer(commandBuffer), "failed to record command buffer!");
    }

public:
    // Run all of our Vulkan cleanup within the destructor
    // That way, an exception will still properly wind down the Vulkan resources
    ~LocalIBLUtility() {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyFence(renderInstance->device, inFlightFences[i], nullptr);
        }

        for (VkFramebuffer framebuffer : renderTargetFramebuffers) {
            vkDestroyFramebuffer(renderInstance->device, framebuffer, nullptr);
        }
        for (VkFramebuffer framebuffer : shadowMapFramebuffers) {
            vkDestroyFramebuffer(renderInstance->device, framebuffer, nullptr);
        }
    }
};

int main(int argc, char* argv[]) {
    uint32_t cubemapSize = 256;
    std::string scenePath = "";

    int currentIndex = 1;
    float exposure = 1.0;
    const std::vector<std::string_view> args(argv + 0, argv + argc);
    while (currentIndex < argc) {
        std::string_view currentArg = args[currentIndex];
        if (currentArg == "--scene") {
            currentIndex += 1;
            if (currentIndex >= argc) {
                PANIC("missing argument to --cubemap-size");
            }

            scenePath = args[currentIndex];
        }
        else if (currentArg == "--cubemap-size") {
            currentIndex += 1;
            if (currentIndex >= argc) {
                PANIC("missing argument to --cubemap-size");
            }

            std::string_view sizeStr = args[currentIndex];
            auto sizeResult = std::from_chars(sizeStr.data(), sizeStr.data() + sizeStr.size(), cubemapSize);
            if (sizeResult.ec == std::errc::invalid_argument || sizeResult.ec == std::errc::result_out_of_range) {
                PANIC("invalid argument to --cubemap-size");
            }
        }
        else if (currentArg == "--exposure") {
            currentIndex += 1;
            if (currentIndex >= argc) {
                PANIC("missing argument to --exposure");
            }

            std::string_view expStr = args[currentIndex];
            auto expResult = std::from_chars(expStr.data(), expStr.data() + expStr.size(), exposure);
            if (expResult.ec == std::errc::invalid_argument || expResult.ec == std::errc::result_out_of_range) {
                PANIC("invalid argument to --exposure");
            }
        }
        else {
            PANIC("invalid command line argument: " + std::string(currentArg));
        }

        currentIndex += 1;
    }
    if (scenePath == "") {
        PANIC("Scene path must be specified with --scene!");
    }

    LocalIBLUtility iblUtil;

    try {
        iblUtil.run(scenePath, cubemapSize, exposure);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
