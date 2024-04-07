#include <vulkan/vulkan.h>

#include <cstring>

#include "descriptor.hpp"
#include "util.hpp"

const float pbrBRDFArray[] =
#include "ggx_lut.inl"
;

Descriptors::Descriptors(std::shared_ptr<RenderInstance> renderInstanceIn, Scene& scene, MaterialPipelines const& materialPipelines, uint32_t frameCopies) : renderInstance(renderInstanceIn) {
    createTextureSampler();
    createUniformBuffers(scene, frameCopies);
    createDescriptorSets(scene, materialPipelines);
}

// Create a texture sampler (not per image)
void Descriptors::createTextureSampler() {
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
void Descriptors::createUniformBuffers(Scene const& scene, uint32_t frameCopies) {
    // Create camera uniform buffers
    VkDeviceSize cameraBufferSize = sizeof(CameraInfo) * frameCopies;
    cameraUniformBuffer = std::make_unique<CombinedBuffer>(renderInstance, cameraBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkMapMemory(renderInstance->device, cameraUniformBuffer->bufferMemory, 0, cameraBufferSize, 0, &cameraUniformMap);

    // Create environment uniform buffers
    VkDeviceSize environmentBufferSize = sizeof(EnvironmentInfo) * frameCopies;
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
    VkDeviceSize lightSSBOSize = sizeof(LightInfo) * scene.lights.size() * frameCopies;
    lightStorageBuffer = std::make_unique<CombinedBuffer>(renderInstance, lightSSBOSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkMapMemory(renderInstance->device, lightStorageBuffer->bufferMemory, 0, lightSSBOSize, 0, &lightStorageMap);
}

void Descriptors::createDescriptorSets(Scene& scene, MaterialPipelines const& materialPipelines) {
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
        .pSetLayouts = &materialPipelines.cameraInfoLayout,
    };

    VK_ERR(vkAllocateDescriptorSets(renderInstance->device, &cameraAllocInfo, &scene.cameraDescriptorSet), "failed to allocate descriptor sets!");

    // Allocate the environment descriptor sets
    std::vector<VkDescriptorSetLayout> envLayouts(environmentDescs, materialPipelines.environmentLayout);
    VkDescriptorSetAllocateInfo envAllocInfo {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = environmentDescs,
        .pSetLayouts = envLayouts.data(),
    };

    std::vector<VkDescriptorSet> envDescriptorSets(environmentDescs);
    VK_ERR(vkAllocateDescriptorSets(renderInstance->device, &envAllocInfo, envDescriptorSets.data()), "failed to allocate descriptor sets!");

    // Allocate the Simple/Environment/Mirror material descriptor sets
    std::vector<VkDescriptorSetLayout> simpleEnvMirrorLayouts(simpleEnvMirrorDescs, materialPipelines.simpleEnvMirrorLayout);
    VkDescriptorSetAllocateInfo simpleEnvMirrorAllocInfo {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = simpleEnvMirrorDescs,
        .pSetLayouts = simpleEnvMirrorLayouts.data(),
    };

    std::vector<VkDescriptorSet> simpleEnvMirrorDescriptorSets(simpleEnvMirrorDescs);
    VK_ERR(vkAllocateDescriptorSets(renderInstance->device, &simpleEnvMirrorAllocInfo, simpleEnvMirrorDescriptorSets.data()), "failed to allocate descriptor sets!");

    // Allocate the Lambertian material descriptor sets
    std::vector<VkDescriptorSetLayout> lambertianLayouts(lambertianDescs, materialPipelines.lambertianLayout);
    VkDescriptorSetAllocateInfo lambertianAllocInfo {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = lambertianDescs,
        .pSetLayouts = lambertianLayouts.data(),
    };

    std::vector<VkDescriptorSet> lambertianDescriptorSets(lambertianDescs);
    VK_ERR(vkAllocateDescriptorSets(renderInstance->device, &lambertianAllocInfo, lambertianDescriptorSets.data()), "failed to allocate descriptor sets!");

    // Allocate the PBR material descriptor sets
    std::vector<VkDescriptorSetLayout> pbrLayouts(pbrDescs, materialPipelines.pbrLayout);
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
        .pSetLayouts = &materialPipelines.lightLayout,
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

Descriptors::~Descriptors() {
    vkDestroyDescriptorPool(renderInstance->device, descriptorPool, nullptr);

    vkDestroySampler(renderInstance->device, repeatingSampler, nullptr);
    vkDestroySampler(renderInstance->device, clampedSampler, nullptr);
    vkDestroySampler(renderInstance->device, shadowMapSampler, nullptr);
}
