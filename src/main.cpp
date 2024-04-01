#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include <cstdlib>
#include <cstring>

#include "buffer.hpp"
#include "instance.hpp"
#include "linear.hpp"
#include "materials.hpp"
#include "options.hpp"
#include "scene.hpp"
#include "util.hpp"

const int MAX_FRAMES_IN_FLIGHT = 2;

const float pbrBRDFArray[] =
#include "ggx_lut.inl"
;

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

    std::vector<VkFramebuffer> renderTargetFramebuffers;
    std::vector<VkFramebuffer> shadowMapFramebuffers;

    VkSampler repeatingSampler;
    VkSampler clampedSampler;
    VkSampler shadowMapSampler;

    std::unique_ptr<CombinedImage> depthImage;

    std::unique_ptr<CombinedBuffer> cameraUniformBuffer;
    void* cameraUniformMap;
    std::vector<CombinedBuffer> environmentUniformBuffers;
    std::vector<void*> environmentUniformMaps;
    std::unique_ptr<CombinedImage> pbrBRDFImage;
    std::unique_ptr<CombinedBuffer> lightStorageBuffer;
    void* lightStorageMap;

    VkDescriptorPool descriptorPool;
    std::unique_ptr<CombinedImage> defaultDescriptorImage;

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
        materialPipelines = std::make_unique<MaterialPipelines>(renderInstance, scene);

        createCommandBuffers();

        createDepthImage();
        createFramebuffers();

        createTextureSampler();

        createUniformBuffers();
        createDescriptorSets();

        createSyncObjects();
    }

    void initRenderInstance() {
        RenderInstanceOptions options {
            .headless = false,
        };

        renderInstance = std::shared_ptr<RenderInstance>(new RenderInstance(options));
    }

    void createFramebuffers() {
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

    void createDepthImage() {
        depthImage = std::make_unique<CombinedImage>(renderInstance, renderInstance->renderImageExtent.width, renderInstance->renderImageExtent.height, VK_FORMAT_D32_SFLOAT,
                                                     VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // Create a texture sampler (not per image)
    void createTextureSampler() {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(renderInstance->physicalDevice, &properties);
        VkSamplerCreateInfo repeatingSamplerInfo {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_ALWAYS,
            .minLod = 0.0f,
            .maxLod = VK_LOD_CLAMP_NONE,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };
        VkSamplerCreateInfo clampedSamplerInfo {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_ALWAYS,
            .minLod = 0.0f,
            .maxLod = VK_LOD_CLAMP_NONE,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };
        VkSamplerCreateInfo shadowMapSamplerInfo {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = VK_TRUE,
            .compareOp = VK_COMPARE_OP_LESS,
            .minLod = 0.0f,
            .maxLod = VK_LOD_CLAMP_NONE,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };

        VK_ERR(vkCreateSampler(renderInstance->device, &repeatingSamplerInfo, nullptr, &repeatingSampler), "failed to create texture sampler!");
        VK_ERR(vkCreateSampler(renderInstance->device, &clampedSamplerInfo, nullptr, &clampedSampler), "failed to create texture sampler!");
        VK_ERR(vkCreateSampler(renderInstance->device, &shadowMapSamplerInfo, nullptr, &shadowMapSampler), "failed to create texture sampler!");
    }

    // Also creates storage buffers too :)
    void createUniformBuffers() {
        // Create camera uniform buffers
        VkDeviceSize cameraBufferSize = sizeof(CameraInfo) * MAX_FRAMES_IN_FLIGHT;
        cameraUniformBuffer = std::make_unique<CombinedBuffer>(renderInstance, cameraBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(renderInstance->device, cameraUniformBuffer->bufferMemory, 0, cameraBufferSize, 0, &cameraUniformMap);

        // Create environment uniform buffers
        VkDeviceSize environmentBufferSize = sizeof(EnvironmentInfo) * MAX_FRAMES_IN_FLIGHT;
        environmentUniformBuffers.reserve(scene.environments.size());
        environmentUniformMaps.resize(scene.environments.size());

        for (size_t i = 0; i < scene.environments.size(); i++) {
            environmentUniformBuffers.emplace_back(renderInstance, environmentBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkMapMemory(renderInstance->device, environmentUniformBuffers[i].bufferMemory, 0, environmentBufferSize, 0, &environmentUniformMaps[i]);
        }

        // Load the PBR BRDF image
        VkDeviceSize pbrBRDFSize = 256 * 256 * 8;
        CombinedBuffer stagingBuffer(renderInstance, pbrBRDFSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        pbrBRDFImage = std::make_unique<CombinedImage>(renderInstance, 256, 256, VK_FORMAT_R32G32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

        void* data;
        vkMapMemory(renderInstance->device, stagingBuffer.bufferMemory, 0, pbrBRDFSize, 0, &data);
        memcpy(data, pbrBRDFArray, sizeof(pbrBRDFArray));
        vkUnmapMemory(renderInstance->device, stagingBuffer.bufferMemory);

        // Copy staging buffer to our PBR BRDF image and prepare it for shader reads
        VkCommandBuffer commandBuffer = beginSingleUseCBuffer(*renderInstance);
        transitionImageLayout(commandBuffer, pbrBRDFImage->image, VK_FORMAT_R32G32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(commandBuffer, stagingBuffer.buffer, pbrBRDFImage->image, 256, 256);
        transitionImageLayout(commandBuffer, pbrBRDFImage->image, VK_FORMAT_R32G32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        endSingleUseCBuffer(*renderInstance, commandBuffer);

        // Create the light storage buffer
        VkDeviceSize lightSSBOSize = sizeof(LightInfo) * scene.lights.size() * MAX_FRAMES_IN_FLIGHT;
        lightStorageBuffer = std::make_unique<CombinedBuffer>(renderInstance, lightSSBOSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(renderInstance->device, lightStorageBuffer->bufferMemory, 0, lightSSBOSize, 0, &lightStorageMap);
    }

    void createDescriptorSets() {
        // Create an image that will be used for the image view descriptors that don't have a valid image
        defaultDescriptorImage = std::make_unique<CombinedImage>(renderInstance, 1, 1, VK_FORMAT_R32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        VkCommandBuffer commandBuffer = beginSingleUseCBuffer(*renderInstance);
        transitionImageLayout(commandBuffer, defaultDescriptorImage->image, VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        endSingleUseCBuffer(*renderInstance, commandBuffer);

        // Pre-calculate counts for descriptors
        uint32_t cameraDescs = 1;
        uint32_t environmentDescs = scene.environments.size();
        uint32_t simpleEnvMirrorDescs = scene.materialCounts.simple + scene.materialCounts.environment + scene.materialCounts.mirror;
        uint32_t lambertianDescs = scene.materialCounts.lambertian;
        uint32_t pbrDescs = scene.materialCounts.pbr;
        uint32_t lightDescs = 1;
        uint32_t shadowMapDescs = scene.shadowMaps.size();

        uint32_t uniformDescs = simpleEnvMirrorDescs + lambertianDescs + pbrDescs;
        uint32_t dynamicUniformDescs = cameraDescs + environmentDescs;
        uint32_t dynamicStorageDescs = lightDescs;
        uint32_t combinedImageSamplerDescs = cameraDescs + 3 * environmentDescs + 2 * simpleEnvMirrorDescs + 3 * lambertianDescs + 5 * pbrDescs + shadowMapDescs;

        // Create the descriptor pool
        // NOTE: Each type needs at least 1 descriptor to allocate, or else we get an error
        std::array<VkDescriptorPoolSize, 4> poolSizes {{
            {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = uniformDescs,
            },
            {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .descriptorCount = dynamicUniformDescs,
            },
            {
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
                .descriptorCount = dynamicStorageDescs,
            },
            {
                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = combinedImageSamplerDescs,
            }
        }};

        VkDescriptorPoolCreateInfo poolInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = cameraDescs + environmentDescs + simpleEnvMirrorDescs + lambertianDescs + pbrDescs + lightDescs,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data(),
        };

        VK_ERR(vkCreateDescriptorPool(renderInstance->device, &poolInfo, nullptr, &descriptorPool), "failed to create descriptor pool!");

        // Allocate the camera descriptor set
        VkDescriptorSetAllocateInfo cameraAllocInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &materialPipelines->cameraInfoLayout,
        };

        VK_ERR(vkAllocateDescriptorSets(renderInstance->device, &cameraAllocInfo, &scene.cameraDescriptorSet), "failed to allocate descriptor sets!");

        // Allocate the environment descriptor sets
        std::vector<VkDescriptorSetLayout> envLayouts(environmentDescs, materialPipelines->environmentLayout);
        VkDescriptorSetAllocateInfo envAllocInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = environmentDescs,
            .pSetLayouts = envLayouts.data(),
        };

        std::vector<VkDescriptorSet> envDescriptorSets(environmentDescs);
        VK_ERR(vkAllocateDescriptorSets(renderInstance->device, &envAllocInfo, envDescriptorSets.data()), "failed to allocate descriptor sets!");

        // Allocate the Simple/Environment/Mirror material descriptor sets
        std::vector<VkDescriptorSetLayout> simpleEnvMirrorLayouts(simpleEnvMirrorDescs, materialPipelines->simpleEnvMirrorLayout);
        VkDescriptorSetAllocateInfo simpleEnvMirrorAllocInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = simpleEnvMirrorDescs,
            .pSetLayouts = simpleEnvMirrorLayouts.data(),
        };

        std::vector<VkDescriptorSet> simpleEnvMirrorDescriptorSets(simpleEnvMirrorDescs);
        VK_ERR(vkAllocateDescriptorSets(renderInstance->device, &simpleEnvMirrorAllocInfo, simpleEnvMirrorDescriptorSets.data()), "failed to allocate descriptor sets!");

        // Allocate the Lambertian material descriptor sets
        std::vector<VkDescriptorSetLayout> lambertianLayouts(lambertianDescs, materialPipelines->lambertianLayout);
        VkDescriptorSetAllocateInfo lambertianAllocInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = lambertianDescs,
            .pSetLayouts = lambertianLayouts.data(),
        };

        std::vector<VkDescriptorSet> lambertianDescriptorSets(lambertianDescs);
        VK_ERR(vkAllocateDescriptorSets(renderInstance->device, &lambertianAllocInfo, lambertianDescriptorSets.data()), "failed to allocate descriptor sets!");

        // Allocate the PBR material descriptor sets
        std::vector<VkDescriptorSetLayout> pbrLayouts(pbrDescs, materialPipelines->pbrLayout);
        VkDescriptorSetAllocateInfo pbrAllocInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = pbrDescs,
            .pSetLayouts = pbrLayouts.data(),
        };

        std::vector<VkDescriptorSet> pbrDescriptorSets(pbrDescs);
        VK_ERR(vkAllocateDescriptorSets(renderInstance->device, &pbrAllocInfo, pbrDescriptorSets.data()), "failed to allocate descriptor sets!");

        // Allocate the light descriptor set
        VkDescriptorSetAllocateInfo lightAllocInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &materialPipelines->lightLayout,
        };

        VK_ERR(vkAllocateDescriptorSets(renderInstance->device, &lightAllocInfo, &scene.lightDescriptorSet), "failed to allocate descriptor sets!");

        // Point our descriptor sets at the underlying resources
        std::vector<VkWriteDescriptorSet> descriptorWrites;
        std::vector<VkDescriptorBufferInfo> bufferWrites;
        std::vector<VkDescriptorImageInfo> imageWrites;
        // Need to reserve enough space for bufferWrites and imageWrites so that they don't move around in memory
        bufferWrites.reserve(uniformDescs + dynamicUniformDescs + dynamicStorageDescs);
        imageWrites.reserve(combinedImageSamplerDescs);

        // Camera descriptors
        VkDescriptorBufferInfo& cameraBufferInfo = bufferWrites.emplace_back(VkDescriptorBufferInfo {
            .buffer = cameraUniformBuffer->buffer,
            .offset = 0,
            .range = sizeof(CameraInfo),
        });
        VkDescriptorImageInfo& pbrBRDFInfo = imageWrites.emplace_back(VkDescriptorImageInfo {
            .sampler = clampedSampler,
            .imageView = pbrBRDFImage->imageView,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        });
        descriptorWrites.emplace_back(VkWriteDescriptorSet {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = scene.cameraDescriptorSet,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .pBufferInfo = &cameraBufferInfo,
        });
        descriptorWrites.emplace_back(VkWriteDescriptorSet {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = scene.cameraDescriptorSet,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &pbrBRDFInfo,
        });

        // Environment descriptors
        for (size_t i = 0; i < environmentDescs; i++) {
            VkDescriptorBufferInfo& envBufferInfo = bufferWrites.emplace_back(VkDescriptorBufferInfo {
                .buffer = environmentUniformBuffers[i].buffer,
                .offset = 0,
                .range = sizeof(EnvironmentInfo),
            });
            // Set to radiance initially for empty cubemaps, and set to proper value below if non-empty
            VkDescriptorImageInfo& radianceInfo = imageWrites.emplace_back(VkDescriptorImageInfo {
                .sampler = repeatingSampler,
                .imageView = scene.environments[i].radiance->imageView,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            });
            VkDescriptorImageInfo& lambertianInfo = imageWrites.emplace_back(VkDescriptorImageInfo {
                .sampler = repeatingSampler,
                .imageView = scene.environments[i].radiance->imageView,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            });
            VkDescriptorImageInfo& ggxInfo = imageWrites.emplace_back(VkDescriptorImageInfo {
                .sampler = repeatingSampler,
                .imageView = scene.environments[i].radiance->imageView,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            });
            if (!scene.environments[i].info.empty) {
                lambertianInfo.imageView = scene.environments[i].lambertian->imageView;
                ggxInfo.imageView = scene.environments[i].ggx->imageView;
            }

            descriptorWrites.emplace_back(VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = envDescriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .pBufferInfo = &envBufferInfo,
            });
            descriptorWrites.emplace_back(VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = envDescriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &radianceInfo,
            });
            descriptorWrites.emplace_back(VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = envDescriptorSets[i],
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &lambertianInfo,
            });
            descriptorWrites.emplace_back(VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = envDescriptorSets[i],
                .dstBinding = 3,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &ggxInfo,
            });

            scene.environments[i].descriptorSet = envDescriptorSets[i];
        }

        // Material descriptors
        VkBuffer materialConstantsBuffer = scene.getMaterialConstantsBuffer().buffer;
        size_t simpleEnvMirrorIndex = 0;
        size_t lambertianIndex = 0;
        size_t pbrIndex = 0;
        for (size_t i = 0; i < scene.materials.size(); i++) {
            Material& material = scene.materials[i];
            if (material.type == MaterialType::SIMPLE || material.type == MaterialType::ENVIRONMENT || material.type == MaterialType::MIRROR) {
                material.descriptorSet = simpleEnvMirrorDescriptorSets[simpleEnvMirrorIndex];
                simpleEnvMirrorIndex += 1;
            }
            else if (material.type == MaterialType::LAMBERTIAN) {
                material.descriptorSet = lambertianDescriptorSets[lambertianIndex];
                lambertianIndex += 1;
            }
            else if (material.type == MaterialType::PBR) {
                material.descriptorSet = pbrDescriptorSets[pbrIndex];
                pbrIndex += 1;
            }

            // Add the MaterialConstants binding
            VkDescriptorBufferInfo& materialConstantsInfo = bufferWrites.emplace_back(VkDescriptorBufferInfo {
                .buffer = materialConstantsBuffer,
                .offset = sizeof(MaterialConstants) * i,
                .range = sizeof(MaterialConstants),
            });
            descriptorWrites.emplace_back(VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = material.descriptorSet,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &materialConstantsInfo,
            });

            // Add normal map if we have one
            VkDescriptorImageInfo& normalInfo = imageWrites.emplace_back(VkDescriptorImageInfo {
                .sampler = repeatingSampler,
                .imageView = defaultDescriptorImage->imageView,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            });
            if (material.normalMap != nullptr) {
                normalInfo.imageView = material.normalMap->imageView;
            }
            descriptorWrites.emplace_back(VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = material.descriptorSet,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &normalInfo,
            });

            // Add displacement map if we have one
            VkDescriptorImageInfo& displacementInfo = imageWrites.emplace_back(VkDescriptorImageInfo {
                .sampler = repeatingSampler,
                .imageView = defaultDescriptorImage->imageView,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            });
            if (material.displacementMap != nullptr) {
                displacementInfo.imageView = material.displacementMap->imageView;
            }
            descriptorWrites.emplace_back(VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = material.descriptorSet,
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &displacementInfo,
            });

            if (material.type == MaterialType::LAMBERTIAN || material.type == MaterialType::PBR) {
                // Add albedo map if we have one
                VkDescriptorImageInfo& albedoInfo = imageWrites.emplace_back(VkDescriptorImageInfo {
                    .sampler = repeatingSampler,
                    .imageView = defaultDescriptorImage->imageView,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                });
                if (holds_alternative<std::unique_ptr<CombinedImage>>(material.albedoMap)) {
                    albedoInfo.imageView = get<std::unique_ptr<CombinedImage>>(material.albedoMap)->imageView;
                }
                descriptorWrites.emplace_back(VkWriteDescriptorSet {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = material.descriptorSet,
                    .dstBinding = 3,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &albedoInfo,
                });
            }

            if (material.type == MaterialType::PBR) {
                // Add roughness map if we have one
                VkDescriptorImageInfo& roughnessInfo = imageWrites.emplace_back(VkDescriptorImageInfo {
                    .sampler = repeatingSampler,
                    .imageView = defaultDescriptorImage->imageView,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                });
                if (holds_alternative<std::unique_ptr<CombinedImage>>(material.roughnessMap)) {
                    roughnessInfo.imageView = get<std::unique_ptr<CombinedImage>>(material.roughnessMap)->imageView;
                }
                descriptorWrites.emplace_back(VkWriteDescriptorSet {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = material.descriptorSet,
                    .dstBinding = 4,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &roughnessInfo,
                });

                // Add metalness map if we have one
                VkDescriptorImageInfo& metalnessInfo = imageWrites.emplace_back(VkDescriptorImageInfo {
                    .sampler = repeatingSampler,
                    .imageView = defaultDescriptorImage->imageView,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                });
                if (holds_alternative<std::unique_ptr<CombinedImage>>(material.metalnessMap)) {
                    metalnessInfo.imageView = get<std::unique_ptr<CombinedImage>>(material.metalnessMap)->imageView;
                }
                descriptorWrites.emplace_back(VkWriteDescriptorSet {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = material.descriptorSet,
                    .dstBinding = 5,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &metalnessInfo,
                });
            }
        }

        // Light descriptor set writes
        VkDescriptorBufferInfo& lightBufferInfo = bufferWrites.emplace_back(VkDescriptorBufferInfo {
            .buffer = lightStorageBuffer->buffer,
            .offset = 0,
            .range = sizeof(LightInfo) * scene.lights.size(),
        });
        descriptorWrites.emplace_back(VkWriteDescriptorSet {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = scene.lightDescriptorSet,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
            .pBufferInfo = &lightBufferInfo,
        });

        for (uint32_t i = 0; i < scene.shadowMaps.size(); i++) {
            VkDescriptorImageInfo& displacementInfo = imageWrites.emplace_back(VkDescriptorImageInfo {
                .sampler = shadowMapSampler,
                .imageView = scene.shadowMaps[i].imageView,
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            });
            descriptorWrites.emplace_back(VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = scene.lightDescriptorSet,
                .dstBinding = 1,
                .dstArrayElement = i,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &displacementInfo,
            });
        }

        // Commit all the writes
        vkUpdateDescriptorSets(renderInstance->device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
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
        uint8_t* dstCameraMap = reinterpret_cast<uint8_t*>(cameraUniformMap) + currentFrame * sizeof(CameraInfo);
        memcpy(dstCameraMap, &scene.cameraInfo, sizeof(CameraInfo));
        for (size_t i = 0; i < scene.environments.size(); i++) {
            uint8_t* dstEnvMap = reinterpret_cast<uint8_t*>(environmentUniformMaps[i]) + currentFrame * sizeof(EnvironmentInfo);
            memcpy(dstEnvMap, &scene.environments[i].info, sizeof(EnvironmentInfo));
        }
        for (size_t i = 0; i < scene.lights.size(); i++) {
            uint8_t* dstLightMap = reinterpret_cast<uint8_t*>(lightStorageMap) + (i + currentFrame * scene.lights.size()) * sizeof(LightInfo);
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
        createFramebuffers();
    }

public:
    // Run all of our Vulkan cleanup within the destructor
    // That way, an exception will still properly wind down the Vulkan resources
    ~VKRendererApp() {
        cleanupFramebuffers();

        vkDestroyDescriptorPool(renderInstance->device, descriptorPool, nullptr);

        vkDestroySampler(renderInstance->device, repeatingSampler, nullptr);
        vkDestroySampler(renderInstance->device, clampedSampler, nullptr);
        vkDestroySampler(renderInstance->device, shadowMapSampler, nullptr);

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(renderInstance->device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(renderInstance->device, renderFinishedSemaphores[i], nullptr);
            vkDestroyFence(renderInstance->device, inFlightFences[i], nullptr);
        }

        for (VkFramebuffer framebuffer : shadowMapFramebuffers) {
            vkDestroyFramebuffer(renderInstance->device, framebuffer, nullptr);
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
