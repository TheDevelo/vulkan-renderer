#include <vulkan/vulkan.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include <cstdlib>
#include <cstring>

#include "linear.hpp"
#include "instance.hpp"
#include "util.hpp"

const int MAX_FRAMES_IN_FLIGHT = 2;

// Container struct for buffer copy regions
struct BufferCopy {
    VkBuffer srcBuffer;
    VkDeviceSize srcOffset;
    VkBuffer dstBuffer;
    VkDeviceSize dstOffset;
    VkDeviceSize size;
};

// Helper function to read files into a vector of chars
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

// Vertex struct for Vulkan
struct Vertex {
    Vec2<float> pos;
    Vec3<float> color;
    Vec2<float> texUV;

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
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, pos),
            },
            {
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, color),
            },
            {
                .location = 2,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, texUV),
            }
        }};

        return attributeDescriptions;
    }
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
};

const std::vector<uint16_t> indices = {
    0, 1, 2, 0, 2, 3
};

// MVP matrices for the vertex shader
struct MVPMatrices {
    Mat4<float> model;
    Mat4<float> viewProj;
};

class VKRendererApp {
public:
    void run() {
        initRenderInstance();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    std::unique_ptr<RenderInstance> renderInstance;

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;

    std::vector<VkFramebuffer> renderTargetFramebuffers;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMaps;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    uint32_t currentFrame = 0;

    void initVulkan() {
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();

        createFramebuffers();

        createCommandBuffers();

        createTextureImage();
        createTextureSampler();

        createVertexBuffer();
        createUniformBuffers();
        createDescriptorSets();

        createSyncObjects();
    }

    void initRenderInstance() {
        RenderInstanceOptions options {
            .headless = false,
        };

        renderInstance = std::unique_ptr<RenderInstance>(new RenderInstance(options));
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

        // Define the subpass for rendering our triangle
        VkSubpassDescription subpassDesc {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef,
        };
        VkSubpassDependency dependency {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        };

        // Create the whole render pass
        VkRenderPassCreateInfo renderPassInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &colorAttachment,
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
        // Pipeline layout determines which uniforms are available to the shaders
        VkPipelineLayoutCreateInfo pipelineLayoutInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout,
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
            .pDepthStencilState = nullptr, // Null for now, will eventually fill in once we add our depth buffer
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
            VkImageView attachments[] = {
                renderInstance->renderImageViews[i],
            };

            VkFramebufferCreateInfo framebufferInfo {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = renderPass,
                .attachmentCount = 1,
                .pAttachments = attachments,
                .width = renderInstance->renderImageExtent.width,
                .height = renderInstance->renderImageExtent.height,
                .layers = 1,
            };

            VK_ERR(vkCreateFramebuffer(renderInstance->device, &framebufferInfo, nullptr, &renderTargetFramebuffers[i]), "failed to create framebuffer!");
        }
    }

    // Create the command pool and buffer to send rendering commands to the GPU
    void createCommandBuffers() {
        QueueFamilyIndices queueFamilyIndices = renderInstance->getQueueFamilies();

        // Create the command pool
        VkCommandPoolCreateInfo poolInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
        };

        VK_ERR(vkCreateCommandPool(renderInstance->device, &poolInfo, nullptr, &commandPool), "failed to create command pool!");

        // Allocate the command buffers
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo commandBufferAllocInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = static_cast<uint32_t>(commandBuffers.size()),
        };

        VK_ERR(vkAllocateCommandBuffers(renderInstance->device, &commandBufferAllocInfo, commandBuffers.data()), "failed to allocate command buffers!");
    }

    // Create a texture and its associated image view
    void createTextureImage() {
        // Load the texture
        int textureWidth, textureHeight, textureChannels;
        stbi_uc* pixels = stbi_load("textures/marble.png", &textureWidth, &textureHeight, &textureChannels, STBI_rgb_alpha);
        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        // Create a staging buffer for our image
        VkDeviceSize imageSize = textureWidth * textureHeight * 4;
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(renderInstance->device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(renderInstance->device, stagingBufferMemory);

        // Free our CPU-side loaded texture
        stbi_image_free(pixels);

        // Create the GPU-side image
        createImage(static_cast<uint32_t>(textureWidth), static_cast<uint32_t>(textureHeight), VK_FORMAT_R8G8B8A8_SRGB,
                    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    textureImage, textureImageMemory);

        // Copy staging buffer to our image and prepare it for shader reads
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(textureWidth), static_cast<uint32_t>(textureHeight));
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // Free our staging buffer
        vkDestroyBuffer(renderInstance->device, stagingBuffer, nullptr);
        vkFreeMemory(renderInstance->device, stagingBufferMemory, nullptr);

        // Create the texture image view
        textureImageView = createImageView(renderInstance->device, textureImage, VK_FORMAT_R8G8B8A8_SRGB);
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
        vkGetImageMemoryRequirements(renderInstance->device, textureImage, &memRequirements);

        VkMemoryAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, memProps),
        };

        VK_ERR(vkAllocateMemory(renderInstance->device, &allocInfo, nullptr, &imageMemory), "failed to allocate image memory!");
        vkBindImageMemory(renderInstance->device, textureImage, textureImageMemory, 0);
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = beginSingleUseCBuffer();

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

        endSingleUseCBuffer(commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
        VkCommandBuffer commandBuffer = beginSingleUseCBuffer();

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

        endSingleUseCBuffer(commandBuffer);
    }

    // Create the vertex buffer AND index buffer for our triangle
    void createVertexBuffer() {
        VkDeviceSize vertexSize = sizeof(vertices[0]) * vertices.size();
        createBuffer(vertexSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
        VkDeviceSize indexSize = sizeof(indices[0]) * indices.size();
        createBuffer(indexSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        // Staging buffer will contain both our data for the vertex and index buffer. We'll then copy both simultaneously.
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(vertexSize + indexSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(renderInstance->device, stagingBufferMemory, 0, vertexSize + indexSize, 0, &data);
        memcpy(data, vertices.data(), (size_t) vertexSize);
        // Need to static cast to uint8_t* since we can't do pointer arithmetic on void*. uint8_t* works because vertexSize is in bytes.
        memcpy(static_cast<uint8_t*>(data) + vertexSize, indices.data(), (size_t) vertexSize);
        vkUnmapMemory(renderInstance->device, stagingBufferMemory);

        BufferCopy bufferCopyInfos[] = {
            {
                .srcBuffer = stagingBuffer,
                .srcOffset = 0,
                .dstBuffer = vertexBuffer,
                .dstOffset = 0,
                .size = vertexSize,
            },
            {
                .srcBuffer = stagingBuffer,
                .srcOffset = vertexSize,
                .dstBuffer = indexBuffer,
                .dstOffset = 0,
                .size = indexSize,
            },
        };
        copyBuffers(bufferCopyInfos, 2);

        vkDestroyBuffer(renderInstance->device, stagingBuffer, nullptr);
        vkFreeMemory(renderInstance->device, stagingBufferMemory, nullptr);
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps, VkBuffer& buffer, VkDeviceMemory& memory) {
        VkBufferCreateInfo bufferInfo {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };

        VK_ERR(vkCreateBuffer(renderInstance->device, &bufferInfo, nullptr, &buffer), "failed to create buffer!");

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(renderInstance->device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, memProps),
        };

        VK_ERR(vkAllocateMemory(renderInstance->device, &allocInfo, nullptr, &memory), "failed to allocate buffer memory!");

        vkBindBufferMemory(renderInstance->device, buffer, memory, 0);
    }

    void copyBuffers(BufferCopy* bufferCopyInfos, uint32_t bufferCopyCount) {
        VkCommandBuffer commandBuffer = beginSingleUseCBuffer();

        for (uint32_t i = 0; i < bufferCopyCount; i++) {
            BufferCopy& copyInfo = bufferCopyInfos[i];

            VkBufferCopy copyCmd {
                .srcOffset = copyInfo.srcOffset,
                .dstOffset = copyInfo.dstOffset,
                .size = copyInfo.size,
            };
            vkCmdCopyBuffer(commandBuffer, copyInfo.srcBuffer, copyInfo.dstBuffer, 1, &copyCmd);
        }

        endSingleUseCBuffer(commandBuffer);
    }

    // Create a command buffer that will be used for a single time
    VkCommandBuffer beginSingleUseCBuffer() {
        VkCommandBufferAllocateInfo commandBufferAllocInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        VkCommandBuffer commandBuffer;
        VK_ERR(vkAllocateCommandBuffers(renderInstance->device, &commandBufferAllocInfo, &commandBuffer), "failed to allocate command buffer!");

        VkCommandBufferBeginInfo beginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VK_ERR(vkBeginCommandBuffer(commandBuffer, &beginInfo), "failed to begin command buffer!");

        return commandBuffer;
    }

    void endSingleUseCBuffer(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
        };
        vkQueueSubmit(renderInstance->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(renderInstance->graphicsQueue);

        vkFreeCommandBuffers(renderInstance->device, commandPool, 1, &commandBuffer);
    }

    // Helper function that gets the memory type we need for allocating a buffer
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(renderInstance->physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(MVPMatrices);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMaps.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

            vkMapMemory(renderInstance->device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMaps[i]);
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
                .buffer = uniformBuffers[i],
                .offset = 0,
                .range = sizeof(MVPMatrices),
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
        while (!renderInstance->shouldClose()) {
            renderInstance->processWindowEvents();

            drawFrame();
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
            throw std::runtime_error("failed to acquire next image from swap chain!");
        }

        // Only reset the fence if we are going to be submitting work
        vkResetFences(renderInstance->device, 1, &inFlightFences[currentFrame]);

        // Record our render commands
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        updateMVPMatrices(currentFrame);

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
            throw std::runtime_error("failed to present swap chain image!");
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
        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        VkRenderPassBeginInfo renderPassInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = renderPass,
            .framebuffer = renderTargetFramebuffers[imageIndex],
            .renderArea = VkRect2D {
                .offset = {0, 0},
                .extent = renderInstance->renderImageExtent,
            },
            .clearValueCount = 1,
            .pClearValues = &clearColor,
        };

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Draw our render pipeline
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkViewport viewport {
            .x = 0.0f,
            .y = 0.0f,
            .width = (float) renderInstance->renderImageExtent.width,
            .height = (float) renderInstance->renderImageExtent.height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        VkRect2D scissor {
            .offset = {0, 0},
            .extent = renderInstance->renderImageExtent,
        };
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkBuffer vertexBuffers[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        // End our render pass and command buffer
        vkCmdEndRenderPass(commandBuffer);

        VK_ERR(vkEndCommandBuffer(commandBuffer), "failed to record command buffer!");
    }

    void updateMVPMatrices(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float totalTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        Mat4<float> model = linear::rotate(totalTime, Vec3<float>(0.0f, 0.0f, 1.0f));
        Mat4<float> view = linear::lookAt(Vec3<float>(2.0f), Vec3<float>(0.0f), Vec3<float>(0.0f, 0.0f, 1.0f));
        Mat4<float> proj = linear::perspective(DEG2RADF(60.0), renderInstance->renderImageExtent.width / (float) renderInstance->renderImageExtent.height, 0.1f, 10.0f);
        MVPMatrices mvp {
            .model = model,
            .viewProj = linear::mmul(proj, view),
        };

        memcpy(uniformBuffersMaps[currentImage], &mvp, sizeof(mvp));
    }

    void recreateFramebuffers() {
        cleanupFramebuffers();
        createFramebuffers();
    }

    void cleanup() {
        cleanupFramebuffers();

        vkDestroyDescriptorPool(renderInstance->device, descriptorPool, nullptr);

        vkDestroySampler(renderInstance->device, textureSampler, nullptr);
        vkDestroyImageView(renderInstance->device, textureImageView, nullptr);
        vkDestroyImage(renderInstance->device, textureImage, nullptr);
        vkFreeMemory(renderInstance->device, textureImageMemory, nullptr);

        vkDestroyBuffer(renderInstance->device, vertexBuffer, nullptr);
        vkFreeMemory(renderInstance->device, vertexBufferMemory, nullptr);
        vkDestroyBuffer(renderInstance->device, indexBuffer, nullptr);
        vkFreeMemory(renderInstance->device, indexBufferMemory, nullptr);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(renderInstance->device, uniformBuffers[i], nullptr);
            vkFreeMemory(renderInstance->device, uniformBuffersMemory[i], nullptr);
        }

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(renderInstance->device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(renderInstance->device, renderFinishedSemaphores[i], nullptr);
            vkDestroyFence(renderInstance->device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(renderInstance->device, commandPool, nullptr);

        vkDestroyPipeline(renderInstance->device, pipeline, nullptr);
        vkDestroyPipelineLayout(renderInstance->device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(renderInstance->device, descriptorSetLayout, nullptr);
        vkDestroyRenderPass(renderInstance->device, renderPass, nullptr);
    }

    void cleanupFramebuffers() {
        for (VkFramebuffer framebuffer : renderTargetFramebuffers) {
            vkDestroyFramebuffer(renderInstance->device, framebuffer, nullptr);
        }
    }
};

int main() {
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
