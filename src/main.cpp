#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <map>
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

class VKRendererApp {
public:
    void run() {
        // Init our render instance
        initRenderInstance();

        // Load our scene
        scene = Scene(renderInstance, options::getScenePath());

        // Init the rest of our Vulkan primitives
        initVulkan();

        mainLoop();
    }

private:
    std::shared_ptr<RenderInstance> renderInstance;
    Scene scene;
    std::unique_ptr<MaterialPipelines> materialPipelines;
    std::unique_ptr<Descriptors> descriptors;

    std::unique_ptr<CombinedImage> depthImage;
    std::vector<VkFramebuffer> renderTargetFramebuffers;
    std::vector<VkFramebuffer> shadowMapFramebuffers;

    std::vector<VkFramebuffer> envFramebuffers;
    std::vector<VkImageView> envImageViews;
    std::map<uint32_t, uint32_t> envFramebufferStartIndex;

    std::vector<VkCommandBuffer> commandBuffers;

    // NOTE: the semaphore values encode the expected value once all in flight frames are finished rendering
    // Thus when the next frame starts rendering, we wait on values past the previous frames
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<uint64_t> imageAvailableSemaphoreValues;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<uint64_t> renderFinishedSemaphoreValues;
    std::vector<VkFence> inFlightFences;

    uint32_t currentFrame = 0;

    void initVulkan() {
        materialPipelines = std::make_unique<MaterialPipelines>(renderInstance, scene, renderInstance->renderImageFormat);

        createCommandBuffers();

        createDepthImage();
        createFramebuffers(false);

        createSyncObjects();

        descriptors = std::make_unique<Descriptors>(renderInstance, scene, *materialPipelines, MAX_FRAMES_IN_FLIGHT);
    }

    void initRenderInstance() {
        renderInstance = std::make_shared<RenderInstance>(RenderInstanceOptions {
            .lightweight = false,
        });
    }

    void createFramebuffers(bool recreate) {
        // Create render target framebuffers
        renderTargetFramebuffers.resize(renderInstance->renderImageViews.size());
        for (size_t i = 0; i < renderInstance->renderImageViews.size(); i++) {
            std::array<VkImageView, 2> attachments = {
                renderInstance->renderImageViews[i],
                depthImage->imageView,
            };

            VkFramebufferCreateInfo framebufferInfo {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = materialPipelines->solidRenderPass,
                .attachmentCount = static_cast<uint32_t>(attachments.size()),
                .pAttachments = attachments.data(),
                .width = renderInstance->renderImageExtent.width,
                .height = renderInstance->renderImageExtent.height,
                .layers = 1,
            };

            VK_ERR(vkCreateFramebuffer(renderInstance->device, &framebufferInfo, nullptr, &renderTargetFramebuffers[i]), "failed to create framebuffer!");
        }

        if (!recreate) {
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

            // Create the framebuffers and image views for our parallaxed GGX stacks with mirror local environments
            uint32_t envFramebuffersNeeded = 0;
            for (uint32_t i = 0; i < scene.environments.size(); i++) {
                if (scene.environments[i].info.type == 2) {
                    envFramebufferStartIndex.insert_or_assign(i, envFramebuffersNeeded);
                    envFramebuffersNeeded += scene.environments[i].info.ggxMipLevels * 6;
                }
            }

            envFramebuffers.resize(envFramebuffersNeeded);
            envImageViews.resize(envFramebuffersNeeded);
            for (uint32_t i = 0; i < scene.environments.size(); i++) {
                if (scene.environments[i].info.empty || scene.environments[i].info.type != 2) {
                    continue;
                }

                uint32_t startIndex = envFramebufferStartIndex[i];
                uint32_t width = scene.environments[i].ggx->width;
                uint32_t height = scene.environments[i].ggx->height;
                for (uint32_t mipLevel = 0; mipLevel < scene.environments[i].info.ggxMipLevels; mipLevel++) {
                    for (uint32_t f = 0; f < 6; f++) {
                        uint32_t index = startIndex + mipLevel * 6 + f;

                        // Create the image view to the specific face and mipmap
                        VkImageViewCreateInfo imageViewCreateInfo {
                            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                            .image = scene.environments[i].parallaxedGGX->image,
                            .viewType = VK_IMAGE_VIEW_TYPE_2D,
                            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                            .components = VkComponentMapping {
                                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                            },
                            .subresourceRange = VkImageSubresourceRange {
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = mipLevel,
                                .levelCount = 1,
                                .baseArrayLayer = f,
                                .layerCount = 1,
                            },
                        };
                        VK_ERR(vkCreateImageView(renderInstance->device, &imageViewCreateInfo, nullptr, &envImageViews[index]), "failed to create image view!");

                        // Create the associated framebuffer
                        std::array<VkImageView, 1> attachments = {
                            envImageViews[index],
                        };

                        VkFramebufferCreateInfo framebufferInfo {
                            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                            .renderPass = materialPipelines->mirrorLocalRenderPass,
                            .attachmentCount = static_cast<uint32_t>(attachments.size()),
                            .pAttachments = attachments.data(),
                            .width = width,
                            .height = height,
                            .layers = 1,
                        };

                        VK_ERR(vkCreateFramebuffer(renderInstance->device, &framebufferInfo, nullptr, &envFramebuffers[index]), "failed to create framebuffer!");
                    }
                    width /= 2;
                    height /= 2;
                }
            }
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

    void createDepthImage() {
        depthImage = std::make_unique<CombinedImage>(renderInstance, renderInstance->renderImageExtent.width, renderInstance->renderImageExtent.height, VK_FORMAT_D32_SFLOAT,
                                                     VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // Create the synchronization objects for rendering to our screen
    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        imageAvailableSemaphoreValues.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphoreValues.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        // Timeline semaphore struct is out here so that it doesn't drop out of scope by the time we create our semaphores
        VkSemaphoreTypeCreateInfo timelineSemaphoreInfo {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .pNext = NULL,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue = 0,
        };
        VkSemaphoreCreateInfo semaphoreInfo {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        VkFenceCreateInfo fenceInfo {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };

        // Only use timeline semaphores on headless mode, because the swapchain doesn't support them :(
        if (options::isHeadless()) {
            semaphoreInfo.pNext = &timelineSemaphoreInfo;
        }

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VK_ERR(vkCreateSemaphore(renderInstance->device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]), "failed to create semaphores!");
            VK_ERR(vkCreateSemaphore(renderInstance->device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]), "failed to create semaphores!");
            VK_ERR(vkCreateFence(renderInstance->device, &fenceInfo, nullptr, &inFlightFences[i]), "failed to create fences!");
            imageAvailableSemaphoreValues[i] = 0;
            renderFinishedSemaphoreValues[i] = 0;
        }
    }

    void mainLoop() {
        // frameTime is the real world time to send a frame through, animationTime is the total time used for animations, and processTime is the simulated time between frames
        // frameTime and processTime will roughly match for real rendering, but will be very different for headless mode
        float frameTime = 0.0f;
        float animationRate = 1.0f;
        float animationTime = scene.minAnimTime;

        std::array<float, 6> standardAnimationRates = {{ 1.0f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.0f }};
        uint32_t standardAnimationRateIndex = 0;

        while (!renderInstance->shouldClose()) {
            std::chrono::system_clock::time_point startTime = std::chrono::high_resolution_clock::now();
            float processTime = renderInstance->processEvents();

            // Update the animation time
            animationTime += processTime * animationRate;
            if (animationTime > scene.maxAnimTime) {
                animationTime -= scene.maxAnimTime - scene.minAnimTime;
            }

            // Process events
            for (RenderInstanceEvent event : renderInstance->eventQueue) {
                if (event.type == RI_EV_SWAP_FIXED_CAMERA) {
                    SwapFixedCameraEvent data = get<SwapFixedCameraEvent>(event.data);
                    if (data.name.has_value()) {
                        scene.switchCameraByName(data.name.value());
                        scene.useUserCamera = false;
                        scene.useDebugCamera = false;
                    }
                    else {
                        if (scene.useUserCamera && scene.cameras.size() != 0) {
                            scene.useUserCamera = false;
                            scene.useDebugCamera = false;
                        }
                        else if (scene.cameras.size() != 0) {
                            scene.selectedCamera = (scene.selectedCamera + 1) % scene.cameras.size();
                        }
                    }
                }
                else if (event.type == RI_EV_USE_USER_CAMERA) {
                    scene.useUserCamera = true;
                    scene.useDebugCamera = false;
                }
                else if (event.type == RI_EV_USE_DEBUG_CAMERA) {
                    scene.useUserCamera = true;
                    scene.useDebugCamera = true;
                }
                else if (event.type == RI_EV_USER_CAMERA_MOVE && scene.useUserCamera) {
                    scene.moveUserCamera(get<UserCameraMoveEvent>(event.data), processTime);
                }
                else if (event.type == RI_EV_USER_CAMERA_ROTATE && scene.useUserCamera) {
                    scene.rotateUserCamera(get<UserCameraRotateEvent>(event.data));
                }
                else if (event.type == RI_EV_TOGGLE_ANIMATION) {
                    standardAnimationRateIndex = (standardAnimationRateIndex + 1) % standardAnimationRates.size();
                    animationRate = standardAnimationRates[standardAnimationRateIndex];
                }
                else if (event.type == RI_EV_SET_ANIMATION) {
                    SetAnimationEvent data = get<SetAnimationEvent>(event.data);
                    animationTime = data.time;
                    animationRate = data.rate;
                }
                else if (event.type == RI_EV_CHANGE_CULLING) {
                    ChangeCullingEvent data = get<ChangeCullingEvent>(event.data);
                    scene.cullingMode = data.cullingMode;
                }
                else if (event.type == RI_EV_EXPOSURE) {
                    ExposureEvent data = get<ExposureEvent>(event.data);
                    if (data.setOrMultiply) {
                        scene.cameraInfo.exposure *= data.exposure;
                    }
                    else {
                        scene.cameraInfo.exposure = data.exposure;
                    }
                }
            }

            // Update scene transforms based on animations
            if (animationRate != 0.0f) {
                scene.updateAnimation(animationTime);
            }

            // Draw the frame
            drawFrame();

            // Update frameTime
            std::chrono::system_clock::time_point endTime = std::chrono::high_resolution_clock::now();
            frameTime = std::chrono::duration<float, std::chrono::seconds::period>(endTime - startTime).count();

            if (options::logFrameTimes()) {
                std::cout << "REPORT frame-time " << frameTime * 1000.0f << "ms" << std::endl;
            }
        }

        // Wait until our device has finished all operations before quitting
        vkDeviceWaitIdle(renderInstance->device);
    }

    void drawFrame() {
        // Wait for the frame in the currentFrame slot to finish drawing. These are array functions, hence the 1
        vkWaitForFences(renderInstance->device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // Acquire the next image on the swapchain for us to render to
        uint32_t imageIndex;
        RenderInstanceImageStatus result = renderInstance->acquireImage(imageAvailableSemaphores[currentFrame], imageAvailableSemaphoreValues[currentFrame], imageIndex);

        if (result == RI_TARGET_REBUILD) {
            recreateFramebuffers();
            return;
        }
        else if (result == RI_TARGET_FAILURE) {
            PANIC("failed to acquire next image from swap chain!");
        }

        // Only reset the fence if we are going to be submitting work
        vkResetFences(renderInstance->device, 1, &inFlightFences[currentFrame]);

        // Update our uniform buffers
        scene.updateCameraTransform(*renderInstance);
        scene.updateEnvironmentTransforms();
        scene.updateLightTransforms();
        uint8_t* dstCameraMap = reinterpret_cast<uint8_t*>(descriptors->cameraUniformMap) + currentFrame * sizeof(CameraInfo);
        memcpy(dstCameraMap, &scene.cameraInfo, sizeof(CameraInfo));
        for (size_t i = 0; i < scene.environments.size(); i++) {
            uint8_t* dstEnvMap = reinterpret_cast<uint8_t*>(descriptors->environmentUniformMaps[i]) + currentFrame * sizeof(EnvironmentInfo);
            memcpy(dstEnvMap, &scene.environments[i].info, sizeof(EnvironmentInfo));
        }
        for (size_t i = 0; i < scene.lights.size(); i++) {
            uint8_t* dstLightMap = reinterpret_cast<uint8_t*>(descriptors->lightStorageMap) + (i + currentFrame * scene.lights.size()) * sizeof(LightInfo);
            memcpy(dstLightMap, &scene.lights[i].info, sizeof(LightInfo));
        }

        // Record our render commands
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        // Submit our command buffer for rendering!
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        uint64_t waitValues[] = { imageAvailableSemaphoreValues[currentFrame] + 1 };
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        uint64_t signalValues[] = { renderFinishedSemaphoreValues[currentFrame] + 1 };
        VkTimelineSemaphoreSubmitInfo timelineInfo {
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = waitValues,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = signalValues,
        };
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
        if (options::isHeadless()) {
            // Only enable timeline semaphores on headless
            submitInfo.pNext = &timelineInfo;
        }

        VK_ERR(vkQueueSubmit(renderInstance->graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]), "failed to submit draw command buffer!");

        // Present the frame onto the screen
        result = renderInstance->presentImage(renderFinishedSemaphores[currentFrame], renderFinishedSemaphoreValues[currentFrame], imageIndex);

        if (result == RI_TARGET_REBUILD) {
            recreateFramebuffers();
        } else if (result == RI_TARGET_FAILURE) {
            PANIC("failed to present swap chain image!");
        }

        imageAvailableSemaphoreValues[currentFrame] += 1;
        renderFinishedSemaphoreValues[currentFrame] += 1;
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

        // Start by rendering the shadow maps
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
                .width = (float) shadowMapExtent.width,
                .height = (float) shadowMapExtent.height,
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

        // Render the parallax-correct GGX cubemaps for each mirror local environment
        for (uint32_t i = 0; i < scene.environments.size(); i++) {
            if (scene.environments[i].info.empty || scene.environments[i].info.type != 2) {
                continue;
            }
            uint32_t startIndex = envFramebufferStartIndex[i];
            uint32_t width = scene.environments[i].ggx->width;
            uint32_t height = scene.environments[i].ggx->height;
            for (uint32_t mipLevel = 0; mipLevel < scene.environments[i].info.ggxMipLevels; mipLevel++) {
                for (uint32_t f = 0; f < 6; f++) {
                    uint32_t index = startIndex + mipLevel * 6 + f;

                    VkExtent2D extent {
                        .width = width,
                        .height = width,
                    };

                    // Start the render pass
                    VkClearValue clearColor {
                        .color = {{0.0f, 0.0f, 0.0f, 1.0f}},
                    };
                    VkRenderPassBeginInfo renderPassInfo {
                        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                        .renderPass = materialPipelines->mirrorLocalRenderPass,
                        .framebuffer = envFramebuffers[index],
                        .renderArea = VkRect2D {
                            .offset = {0, 0},
                            .extent = extent,
                        },
                        .clearValueCount = 1,
                        .pClearValues = &clearColor,
                    };
                    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

                    // Set the camera viewport and scissor
                    // We negate height to flip the render AND render backfaces
                    VkViewport viewport {
                        .x = 0.0f,
                        .y = (float) height,
                        .width = (float) width,
                        .height = -((float) height),
                        .minDepth = 0.0f,
                        .maxDepth = 1.0f,
                    };
                    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
                    VkRect2D scissor {
                        .offset = {0, 0},
                        .extent = extent,
                    };
                    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

                    // Draw the scene
                    SceneRenderInfo sceneRenderInfo {
                        .commandBuffer = commandBuffer,
                        .pipelines = *materialPipelines,
                        .type = SceneRenderType::MIRROR_LOCAL,

                        .cameraDescriptorOffset = currentFrame * static_cast<uint32_t>(sizeof(CameraInfo)),
                        .environmentDescriptorOffset = currentFrame * static_cast<uint32_t>(sizeof(EnvironmentInfo)),

                        .face = f,
                        .mipLevel = mipLevel,
                    };
                    scene.renderEnvBoundingMesh(sceneRenderInfo, i);

                    // End our render pass
                    vkCmdEndRenderPass(commandBuffer);
                }
                width /= 2;
                height /= 2;
            }

            // Transition the cubemap into a format that can be read by the shaders
            transitionImageLayout(commandBuffer, scene.environments[i].parallaxedGGX->image, VK_FORMAT_R32G32B32A32_SFLOAT,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VkImageSubresourceRange {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = scene.environments[i].info.ggxMipLevels,
                    .baseArrayLayer = 0,
                    .layerCount = 6,
                });
        }

        // Start the solid render pass
        std::array<VkClearValue, 2> clearColors;
        clearColors[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearColors[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo renderPassInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = materialPipelines->solidRenderPass,
            .framebuffer = renderTargetFramebuffers[imageIndex],
            .renderArea = VkRect2D {
                .offset = {0, 0},
                .extent = renderInstance->renderImageExtent,
            },
            .clearValueCount = static_cast<uint32_t>(clearColors.size()),
            .pClearValues = clearColors.data(),
        };

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Calculate our viewport to add letter/pillarboxing, so that the aspect ratio of the rendered region matches our camera's
        float cameraAspect;
        if (scene.useUserCamera) {
            cameraAspect = renderInstance->renderImageExtent.width / (float) renderInstance->renderImageExtent.height;
        }
        else {
            cameraAspect = scene.cameras[scene.selectedCamera].aspectRatio;
        }
        VkViewport viewport {
            .x = 0.0f,
            .y = 0.0f,
            .width = (float) renderInstance->renderImageExtent.width,
            .height = (float) renderInstance->renderImageExtent.height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        if (viewport.width / viewport.height > cameraAspect) {
            viewport.width = cameraAspect * viewport.height;
            viewport.x = (renderInstance->renderImageExtent.width - viewport.width) * 0.5f;
        }
        else if (viewport.width / viewport.height < cameraAspect) {
            viewport.height = viewport.width / cameraAspect;
            viewport.y = (renderInstance->renderImageExtent.height - viewport.height) * 0.5f;
        }
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor {
            .offset = {0, 0},
            .extent = renderInstance->renderImageExtent,
        };
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // Draw the scene
        SceneRenderInfo sceneRenderInfo {
            .commandBuffer = commandBuffer,
            .pipelines = *materialPipelines,
            .type = SceneRenderType::SOLID,

            .cameraDescriptorOffset = currentFrame * static_cast<uint32_t>(sizeof(CameraInfo)),
            .environmentDescriptorOffset = currentFrame * static_cast<uint32_t>(sizeof(EnvironmentInfo)),
            .lightDescriptorOffset = currentFrame * static_cast<uint32_t>(sizeof(LightInfo) * scene.lights.size()),
        };
        scene.renderScene(sceneRenderInfo);

        // End our render pass and command buffer
        vkCmdEndRenderPass(commandBuffer);

        VK_ERR(vkEndCommandBuffer(commandBuffer), "failed to record command buffer!");
    }

    void recreateFramebuffers() {
        cleanupFramebuffers();

        createDepthImage();
        createFramebuffers(true);
    }

public:
    // Run all of our Vulkan cleanup within the destructor
    // That way, an exception will still properly wind down the Vulkan resources
    ~VKRendererApp() {
        cleanupFramebuffers();

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(renderInstance->device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(renderInstance->device, renderFinishedSemaphores[i], nullptr);
            vkDestroyFence(renderInstance->device, inFlightFences[i], nullptr);
        }

        for (VkFramebuffer framebuffer : shadowMapFramebuffers) {
            vkDestroyFramebuffer(renderInstance->device, framebuffer, nullptr);
        }

        for (VkFramebuffer framebuffer : envFramebuffers) {
            vkDestroyFramebuffer(renderInstance->device, framebuffer, nullptr);
        }
        for (VkImageView view : envImageViews) {
            vkDestroyImageView(renderInstance->device, view, nullptr);
        }
    }

private:
    void cleanupFramebuffers() {
        for (VkFramebuffer framebuffer : renderTargetFramebuffers) {
            vkDestroyFramebuffer(renderInstance->device, framebuffer, nullptr);
        }
    }
};

int main(int argc, char* argv[]) {
    options::parse(argc, argv);
    // List all physical devices if requested, and then immediately return
    if (options::listDevices()) {
        // Init a basic Vulkan instance
        VkInstance instance;
        VkInstanceCreateInfo instanceCreateInfo {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .enabledLayerCount = 0,
            .enabledExtensionCount = 0,
        };
        VK_ERR(vkCreateInstance(&instanceCreateInfo, nullptr, &instance), "failed to create VK instance!");

        // Grab the list of possible physical devices
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // List each device out
        VkPhysicalDeviceProperties deviceProps;
        uint32_t n = 1;
        for (VkPhysicalDevice device : devices) {
            vkGetPhysicalDeviceProperties(device, &deviceProps);
            std::cout << "Device " << n << ": " << deviceProps.deviceName << std::endl;
            n += 1;
        }

        vkDestroyInstance(instance, nullptr);
        return EXIT_SUCCESS;
    }

    VKRendererApp app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
