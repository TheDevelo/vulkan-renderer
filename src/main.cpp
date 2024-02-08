#include <vulkan/vulkan.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
#include "options.hpp"
#include "scene.hpp"
#include "util.hpp"

const int MAX_FRAMES_IN_FLIGHT = 2;

// Vertex struct for Vulkan
struct Vertex {
    Vec3<float> pos;
    Vec3<float> normal;
    Vec4<uint8_t> color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription {
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions {{
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, pos),
            },
            {
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, normal),
            },
            {
                .location = 2,
                .binding = 0,
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .offset = offsetof(Vertex, color),
            }
        }};

        return attributeDescriptions;
    }
};

class VKRendererApp {
public:
    void run() {
        // Init our render instance
        initRenderInstance();
        initVulkan();

        // Load our scene
        scene = Scene(renderInstance, options::getScenePath());

        mainLoop();
    }

private:
    std::shared_ptr<RenderInstance> renderInstance;
    Scene scene;

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;

    std::vector<VkFramebuffer> renderTargetFramebuffers;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    std::vector<CombinedBuffer> uniformBuffers;
    std::vector<void*> uniformBuffersMaps;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    uint32_t currentFrame = 0;

    void initVulkan() {
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();

        createCommandBuffers();

        createDepthImage();
        createFramebuffers();

        createTextureImage();
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

    // Creates the render pass for our renderer
    void createRenderPass() {
        // The attachment describes our screen's framebuffer, and how we want the render pass to modify it.
        VkAttachmentDescription colorAttachment {
            .format = renderInstance->renderImageFormat,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };
        VkAttachmentReference colorAttachmentRef {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        // This attachment describes the depth buffer
        VkAttachmentDescription depthAttachment {
            .format = VK_FORMAT_D32_SFLOAT,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        VkAttachmentReference depthAttachmentRef {
            .attachment = 1,
            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        // Define the subpass for rendering our triangle
        VkSubpassDescription subpassDesc {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef,
            .pDepthStencilAttachment = &depthAttachmentRef
        };
        VkSubpassDependency dependency {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        };

        // Create the whole render pass
        std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
        VkRenderPassCreateInfo renderPassInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = static_cast<uint32_t>(attachments.size()),
            .pAttachments = attachments.data(),
            .subpassCount = 1,
            .pSubpasses = &subpassDesc,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };

        VK_ERR(vkCreateRenderPass(renderInstance->device, &renderPassInfo, nullptr, &renderPass), "failed to create render pass!");
    }

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding mvpLayoutBinding {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        };
        VkDescriptorSetLayoutBinding samplerLayoutBinding {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        };

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = { mvpLayoutBinding, samplerLayoutBinding };
        VkDescriptorSetLayoutCreateInfo layoutInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        };

        VK_ERR(vkCreateDescriptorSetLayout(renderInstance->device, &layoutInfo, nullptr, &descriptorSetLayout), "failed to create descriptor set layout!");
    }

    // Creates the basic graphics pipeline for our renderer
    void createGraphicsPipeline() {
        // Load in the vertex and fragment shaders
        std::vector<char> vertShaderCode = readFile("shaders/vert.spv");
        std::vector<char> fragShaderCode = readFile("shaders/frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        // Create the shader stages for our pipeline
        VkPipelineShaderStageCreateInfo vertShaderStageInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertShaderModule,
            .pName = "main",
        };
        VkPipelineShaderStageCreateInfo fragShaderStageInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragShaderModule,
            .pName = "main",
        };
        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        // Create info used in our render pipeline about the various operations
        // Dynamic states deal with what we can change after pipeline creation.
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
        };
        VkPipelineDynamicStateCreateInfo dynamicStateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data(),
        };
        // Vertex input describes how vertex data will be passed into our shader
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vertexInputInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data(),
        };
        // Input assembly describes what primitives we make from our vertex data
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };
        // Viewport and scissor describe what regions of our scene and screen we should render
        VkPipelineViewportStateCreateInfo viewportStateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount = 1,
        };
        // Rasterizer determines how we convert from primitives to fragments
        VkPipelineRasterizationStateCreateInfo rasterizerStateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f,
        };
        // Multisampling determines anti-aliasing configuration
        VkPipelineMultisampleStateCreateInfo multisamplingInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
        };
        // Color blending determines how we mix the pixels onto the framebuffer
        // The attachment is info per framebuffer, while create info is global
        VkPipelineColorBlendAttachmentState colorBlendAttachment {
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
        VkPipelineColorBlendStateCreateInfo colorBlendInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };
        // Depth stencil determines how we use the depth buffer and stencil buffer
        VkPipelineDepthStencilStateCreateInfo depthStencilInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
        };
        // Push constants are small amounts of data we can directly upload through the command buffer
        // They will be used to upload model transforms for each object
        VkPushConstantRange pushConstantInfo {
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = sizeof(Mat4<float>),
        };
        // Pipeline layout determines which uniforms are available to the shaders
        VkPipelineLayoutCreateInfo pipelineLayoutInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantInfo,
        };

        VK_ERR(vkCreatePipelineLayout(renderInstance->device, &pipelineLayoutInfo, nullptr, &pipelineLayout), "failed to create pipeline layout!");

        // Finally create the render pipeline
        VkGraphicsPipelineCreateInfo renderPipelineInfo {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssemblyInfo,
            .pViewportState = &viewportStateInfo,
            .pRasterizationState = &rasterizerStateInfo,
            .pMultisampleState = &multisamplingInfo,
            .pDepthStencilState = &depthStencilInfo, // Null for now, will eventually fill in once we add our depth buffer
            .pColorBlendState = &colorBlendInfo,
            .pDynamicState = &dynamicStateInfo,
            .layout = pipelineLayout,
            .renderPass = renderPass,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };

        // This call takes in multiple pipelines to create. I'm guessing real apps need a bunch of pipelines, so they made a call to batch compile them.
        // In fact, this is probably where "shader compilation" and "shader caches" come in for games I play. Neat!
        VK_ERR(vkCreateGraphicsPipelines(renderInstance->device, VK_NULL_HANDLE, 1, &renderPipelineInfo, nullptr, &pipeline), "failed to create graphics pipeline!");

        // Clean up our shader modules now that we are finished with them (the pipeline keeps its own copy)
        vkDestroyShaderModule(renderInstance->device, vertShaderModule, nullptr);
        vkDestroyShaderModule(renderInstance->device, fragShaderModule, nullptr);
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo moduleCreateInfo {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.size(),
            .pCode = reinterpret_cast<const uint32_t*>(code.data()),
        };

        VkShaderModule shaderModule;
        VK_ERR(vkCreateShaderModule(renderInstance->device, &moduleCreateInfo, nullptr, &shaderModule), "failed to create shader module!");

        return shaderModule;
    }

    void createFramebuffers() {
        renderTargetFramebuffers.resize(renderInstance->renderImageViews.size());
        for (size_t i = 0; i < renderInstance->renderImageViews.size(); i++) {
            std::array<VkImageView, 2> attachments = {
                renderInstance->renderImageViews[i],
                depthImageView,
            };

            VkFramebufferCreateInfo framebufferInfo {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = renderPass,
                .attachmentCount = static_cast<uint32_t>(attachments.size()),
                .pAttachments = attachments.data(),
                .width = renderInstance->renderImageExtent.width,
                .height = renderInstance->renderImageExtent.height,
                .layers = 1,
            };

            VK_ERR(vkCreateFramebuffer(renderInstance->device, &framebufferInfo, nullptr, &renderTargetFramebuffers[i]), "failed to create framebuffer!");
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
        createImage(renderInstance->renderImageExtent.width, renderInstance->renderImageExtent.height, VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    depthImage, depthImageMemory);
        depthImageView = createImageView(renderInstance->device, depthImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // Create a texture and its associated image view
    void createTextureImage() {
        // Load the texture
        int textureWidth, textureHeight, textureChannels;
        stbi_uc* pixels = stbi_load("textures/marble.png", &textureWidth, &textureHeight, &textureChannels, STBI_rgb_alpha);
        if (!pixels) {
            PANIC("failed to load texture image!");
        }

        // Create a staging buffer for our image
        VkDeviceSize imageSize = textureWidth * textureHeight * 4;
        CombinedBuffer stagingBuffer(renderInstance, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        void* data;
        vkMapMemory(renderInstance->device, stagingBuffer.bufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(renderInstance->device, stagingBuffer.bufferMemory);

        // Free our CPU-side loaded texture
        stbi_image_free(pixels);

        // Create the GPU-side image
        createImage(static_cast<uint32_t>(textureWidth), static_cast<uint32_t>(textureHeight), VK_FORMAT_R8G8B8A8_SRGB,
                    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    textureImage, textureImageMemory);

        // Copy staging buffer to our image and prepare it for shader reads
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer.buffer, textureImage, static_cast<uint32_t>(textureWidth), static_cast<uint32_t>(textureHeight));
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // Create the texture image view
        textureImageView = createImageView(renderInstance->device, textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // Create a texture sampler (not per image)
    void createTextureSampler() {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(renderInstance->physicalDevice, &properties);
        VkSamplerCreateInfo samplerInfo {
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
            .maxLod = 0.0f,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };

        VK_ERR(vkCreateSampler(renderInstance->device, &samplerInfo, nullptr, &textureSampler), "failed to create texture sampler!");
    }

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memProps, VkImage& image, VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = { width, height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };

        VK_ERR(vkCreateImage(renderInstance->device, &imageInfo, nullptr, &image), "failed to create image!");

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(renderInstance->device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(*renderInstance, memRequirements.memoryTypeBits, memProps),
        };

        VK_ERR(vkAllocateMemory(renderInstance->device, &allocInfo, nullptr, &imageMemory), "failed to allocate image memory!");
        vkBindImageMemory(renderInstance->device, image, imageMemory, 0);
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = beginSingleUseCBuffer(*renderInstance);

        VkImageMemoryBarrier barrier {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = VkImageSubresourceRange {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        // Determine the source and destination stages/access masks. Currently hardcoded for our old/new layout pairs, maybe there is a more dynamic way to do this?
        VkPipelineStageFlags srcStage;
        VkPipelineStageFlags dstStage;
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        endSingleUseCBuffer(*renderInstance, commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
        VkCommandBuffer commandBuffer = beginSingleUseCBuffer(*renderInstance);

        VkBufferImageCopy region {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = VkImageSubresourceLayers {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = { 0, 0, 0 },
            .imageExtent = { width, height, 1 },
        };

        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        endSingleUseCBuffer(*renderInstance, commandBuffer);
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(ViewProjMatrices);

        uniformBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMaps.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            uniformBuffers.emplace_back(renderInstance, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkMapMemory(renderInstance->device, uniformBuffers[i].bufferMemory, 0, bufferSize, 0, &uniformBuffersMaps[i]);
        }
    }

    void createDescriptorSets() {
        // Create the descriptor pool
        std::array<VkDescriptorPoolSize, 2> poolSizes {{
            {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            },
            {
                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            }
        }};

        VkDescriptorPoolCreateInfo poolInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data(),
        };

        VK_ERR(vkCreateDescriptorPool(renderInstance->device, &poolInfo, nullptr, &descriptorPool), "failed to create descriptor pool!");

        // Allocate the descriptor sets
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            .pSetLayouts = layouts.data(),
        };

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        VK_ERR(vkAllocateDescriptorSets(renderInstance->device, &allocInfo, descriptorSets.data()), "failed to allocate descriptor sets!");

        // Point our descriptor sets at the underlying uniform buffers
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo bufferInfo {
                .buffer = uniformBuffers[i].buffer,
                .offset = 0,
                .range = sizeof(ViewProjMatrices),
            };

            VkDescriptorImageInfo imageInfo {
                .sampler = textureSampler,
                .imageView = textureImageView,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            std::array<VkWriteDescriptorSet, 2> descriptorWrites {{
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &bufferInfo,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptorSets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &imageInfo,
                }
            }};

            vkUpdateDescriptorSets(renderInstance->device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    // Create the synchronization objects for rendering to our screen
    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, };
        VkFenceCreateInfo fenceInfo {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VK_ERR(vkCreateSemaphore(renderInstance->device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]), "failed to create semaphores!");
            VK_ERR(vkCreateSemaphore(renderInstance->device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]), "failed to create semaphores!");
            VK_ERR(vkCreateFence(renderInstance->device, &fenceInfo, nullptr, &inFlightFences[i]), "failed to create semaphores!");
        }
    }

    void mainLoop() {
        float frameTime = 0.01; // Initialize to a frame time at 100 FPS as a first guess
        while (!renderInstance->shouldClose()) {
            auto startTime = std::chrono::high_resolution_clock::now();
            renderInstance->processWindowEvents();
            for (RenderInstanceEvent event : renderInstance->eventQueue) {
                if (event.type == RI_EV_SWAP_FIXED_CAMERA) {
                    if (scene.useUserCamera) {
                        scene.useUserCamera = false;
                        scene.useDebugCamera = false;
                    }
                    else {
                        scene.selectedCamera = (scene.selectedCamera + 1) % scene.cameras.size();
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
                    scene.moveUserCamera(event.userCameraMoveData, frameTime);
                }
                else if (event.type == RI_EV_USER_CAMERA_ROTATE && scene.useUserCamera) {
                    scene.rotateUserCamera(event.userCameraRotateData);
                }
            }

            drawFrame();

            // Update frameTime for use in next frame
            auto endTime = std::chrono::high_resolution_clock::now();
            frameTime = std::chrono::duration<float, std::chrono::seconds::period>(endTime - startTime).count();
        }

        // Wait until our device has finished all operations before quitting
        vkDeviceWaitIdle(renderInstance->device);
    }

    void drawFrame() {
        // Wait for the last frame to be drawn. These are array functions, hence the 1
        vkWaitForFences(renderInstance->device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // Acquire the next image on the swapchain for us to render to
        uint32_t imageIndex;
        RenderInstanceImageStatus result = renderInstance->acquireImage(imageAvailableSemaphores[currentFrame], imageIndex);

        if (result == RI_TARGET_REBUILD) {
            recreateFramebuffers();
            return;
        }
        else if (result == RI_TARGET_FAILURE) {
            PANIC("failed to acquire next image from swap chain!");
        }

        // Only reset the fence if we are going to be submitting work
        vkResetFences(renderInstance->device, 1, &inFlightFences[currentFrame]);

        // Update our viewProj matrices from our camera
        scene.updateCameraTransform(*renderInstance);
        memcpy(uniformBuffersMaps[currentFrame], &scene.viewProj, sizeof(scene.viewProj));

        // Record our render commands
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        // Submit our command buffer for rendering!
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
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

        VK_ERR(vkQueueSubmit(renderInstance->graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]), "failed to submit draw command buffer!");

        // Present the frame onto the screen
        result = renderInstance->presentImage(renderFinishedSemaphores[currentFrame], imageIndex);

        if (result == RI_TARGET_REBUILD) {
            recreateFramebuffers();
        } else if (result == RI_TARGET_FAILURE) {
            PANIC("failed to present swap chain image!");
        }

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

        // Start the render pass
        std::array<VkClearValue, 2> clearColors;
        clearColors[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearColors[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo renderPassInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = renderPass,
            .framebuffer = renderTargetFramebuffers[imageIndex],
            .renderArea = VkRect2D {
                .offset = {0, 0},
                .extent = renderInstance->renderImageExtent,
            },
            .clearValueCount = static_cast<uint32_t>(clearColors.size()),
            .pClearValues = clearColors.data(),
        };

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Draw our render pipeline
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

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
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        // Draw the scene
        SceneRenderInfo sceneRenderInfo {
            .commandBuffer = commandBuffer,
            .pipelineLayout = pipelineLayout,
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

        vkDestroySampler(renderInstance->device, textureSampler, nullptr);
        vkDestroyImageView(renderInstance->device, textureImageView, nullptr);
        vkDestroyImage(renderInstance->device, textureImage, nullptr);
        vkFreeMemory(renderInstance->device, textureImageMemory, nullptr);

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(renderInstance->device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(renderInstance->device, renderFinishedSemaphores[i], nullptr);
            vkDestroyFence(renderInstance->device, inFlightFences[i], nullptr);
        }

        vkDestroyPipeline(renderInstance->device, pipeline, nullptr);
        vkDestroyPipelineLayout(renderInstance->device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(renderInstance->device, descriptorSetLayout, nullptr);
        vkDestroyRenderPass(renderInstance->device, renderPass, nullptr);
    }

private:
    void cleanupFramebuffers() {
        vkDestroyImageView(renderInstance->device, depthImageView, nullptr);
        vkDestroyImage(renderInstance->device, depthImage, nullptr);
        vkFreeMemory(renderInstance->device, depthImageMemory, nullptr);

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
