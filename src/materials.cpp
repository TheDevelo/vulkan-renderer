#include <vulkan/vulkan.h>

#include <array>

#include "options.hpp"
#include "instance.hpp"
#include "materials.hpp"
#include "util.hpp"

// Shader arrays
const uint32_t vertShaderArray[] =
#include "shaders/shader.vert.inl"
;
const uint32_t simpleFragShaderArray[] =
#include "shaders/simple.frag.inl"
;
const uint32_t environmentFragShaderArray[] =
#include "shaders/environment.frag.inl"
;
const uint32_t mirrorFragShaderArray[] =
#include "shaders/mirror.frag.inl"
;
const uint32_t lambertianFragShaderArray[] =
#include "shaders/lambertian.frag.inl"
;
const uint32_t pbrFragShaderArray[] =
#include "shaders/pbr.frag.inl"
;
const uint32_t shadowVertShaderArray[] =
#include "shaders/shadow.vert.inl"
;

MaterialPipelines::MaterialPipelines(std::shared_ptr<RenderInstance> renderInstanceIn) : renderInstance(renderInstanceIn) {
    createDescriptorSetLayouts();
    createRenderPasses();
    createPipelines();
}

MaterialPipelines::~MaterialPipelines() {
    vkDestroyPipeline(renderInstance->device, simplePipeline, nullptr);
    vkDestroyPipeline(renderInstance->device, environmentPipeline, nullptr);
    vkDestroyPipeline(renderInstance->device, mirrorPipeline, nullptr);
    vkDestroyPipeline(renderInstance->device, lambertianPipeline, nullptr);
    vkDestroyPipeline(renderInstance->device, pbrPipeline, nullptr);
    vkDestroyPipeline(renderInstance->device, shadowPipeline, nullptr);

    vkDestroyPipelineLayout(renderInstance->device, simplePipelineLayout, nullptr);
    vkDestroyPipelineLayout(renderInstance->device, envMirrorPipelineLayout, nullptr);
    vkDestroyPipelineLayout(renderInstance->device, lambertianPipelineLayout, nullptr);
    vkDestroyPipelineLayout(renderInstance->device, pbrPipelineLayout, nullptr);
    vkDestroyPipelineLayout(renderInstance->device, shadowPipelineLayout, nullptr);

    vkDestroyDescriptorSetLayout(renderInstance->device, cameraInfoLayout, nullptr);
    vkDestroyDescriptorSetLayout(renderInstance->device, environmentLayout, nullptr);
    vkDestroyDescriptorSetLayout(renderInstance->device, simpleEnvMirrorLayout, nullptr);
    vkDestroyDescriptorSetLayout(renderInstance->device, lambertianLayout, nullptr);
    vkDestroyDescriptorSetLayout(renderInstance->device, pbrLayout, nullptr);
    vkDestroyDescriptorSetLayout(renderInstance->device, lightLayout, nullptr);

    vkDestroyRenderPass(renderInstance->device, solidRenderPass, nullptr);
    vkDestroyRenderPass(renderInstance->device, shadowRenderPass, nullptr);
}

void MaterialPipelines::createDescriptorSetLayouts() {
    // Camera Info Descriptor Layout
    // Second binding is for the PBR BRDF image.
    std::array<VkDescriptorSetLayoutBinding, 2> cameraInfoBindings = {{
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        }
    }};

    VkDescriptorSetLayoutCreateInfo cameraInfoLayoutInfo {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(cameraInfoBindings.size()),
        .pBindings = cameraInfoBindings.data(),
    };
    VK_ERR(vkCreateDescriptorSetLayout(renderInstance->device, &cameraInfoLayoutInfo, nullptr, &cameraInfoLayout), "failed to create descriptor set layout!");

    // Environment Descriptor Layout
    std::array<VkDescriptorSetLayoutBinding, 4> environmentBindings = {{
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        }
    }};

    VkDescriptorSetLayoutCreateInfo environmentLayoutInfo {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(environmentBindings.size()),
        .pBindings = environmentBindings.data(),
    };
    VK_ERR(vkCreateDescriptorSetLayout(renderInstance->device, &environmentLayoutInfo, nullptr, &environmentLayout), "failed to create descriptor set layout!");

    // Material Descriptor Layout for Simple/Environment/Mirror
    std::array<VkDescriptorSetLayoutBinding, 3> simpleEnvMirrorBindings = {{
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        }
    }};

    VkDescriptorSetLayoutCreateInfo simpleEnvMirrorLayoutInfo {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(simpleEnvMirrorBindings.size()),
        .pBindings = simpleEnvMirrorBindings.data(),
    };
    VK_ERR(vkCreateDescriptorSetLayout(renderInstance->device, &simpleEnvMirrorLayoutInfo, nullptr, &simpleEnvMirrorLayout), "failed to create descriptor set layout!");

    // Material Descriptor Layout for Lambertian
    std::array<VkDescriptorSetLayoutBinding, 4> lambertianBindings = {{
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        }
    }};

    VkDescriptorSetLayoutCreateInfo lambertianLayoutInfo {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(lambertianBindings.size()),
        .pBindings = lambertianBindings.data(),
    };
    VK_ERR(vkCreateDescriptorSetLayout(renderInstance->device, &lambertianLayoutInfo, nullptr, &lambertianLayout), "failed to create descriptor set layout!");

    // Material Descriptor Layout for PBR
    std::array<VkDescriptorSetLayoutBinding, 6> pbrBindings = {{
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 4,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 5,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        }
    }};

    VkDescriptorSetLayoutCreateInfo pbrLayoutInfo {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(pbrBindings.size()),
        .pBindings = pbrBindings.data(),
    };
    VK_ERR(vkCreateDescriptorSetLayout(renderInstance->device, &pbrLayoutInfo, nullptr, &pbrLayout), "failed to create descriptor set layout!");

    // Light Storage Buffer Descriptor Layout
    std::array<VkDescriptorSetLayoutBinding, 1> lightBindings = {{
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        }
    }};

    VkDescriptorSetLayoutCreateInfo lightLayoutInfo {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(lightBindings.size()),
        .pBindings = lightBindings.data(),
    };
    VK_ERR(vkCreateDescriptorSetLayout(renderInstance->device, &lightLayoutInfo, nullptr, &lightLayout), "failed to create descriptor set layout!");
}

// Creates the renderes pass for both solid rendering and shadow map rendering
void MaterialPipelines::createRenderPasses() {
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
    if (options::isHeadless()) {
        // If we're headless, then using PRESENT_SRC_KHR makes no sense. So just use COLOR_ATTACHMENT_OPTIMAL
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

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

    // Define the subpass for rendering our meshes
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

    VK_ERR(vkCreateRenderPass(renderInstance->device, &renderPassInfo, nullptr, &solidRenderPass), "failed to create render pass!");

    // Create the render pass for shadow mapping
    VkAttachmentDescription shadowDepthAttachment {
        .format = VK_FORMAT_D32_SFLOAT,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    VkAttachmentReference shadowDepthAttachmentRef {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    VkSubpassDescription shadowSubpassDesc {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 0,
        .pDepthStencilAttachment = &shadowDepthAttachmentRef
    };

    std::array<VkAttachmentDescription, 1> shadowAttachments = { shadowDepthAttachment };
    VkRenderPassCreateInfo shadowRenderPassInfo {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = static_cast<uint32_t>(shadowAttachments.size()),
        .pAttachments = shadowAttachments.data(),
        .subpassCount = 1,
        .pSubpasses = &shadowSubpassDesc,
    };

    VK_ERR(vkCreateRenderPass(renderInstance->device, &shadowRenderPassInfo, nullptr, &shadowRenderPass), "failed to create render pass!");
}

// Helper to create a shader module from a SPIRV array
VkShaderModule createShaderModule(RenderInstance const& renderInstance, const uint32_t* spirvCode, size_t spirvSize) {
    VkShaderModuleCreateInfo moduleCreateInfo {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirvSize,
        .pCode = spirvCode,
    };

    VkShaderModule shaderModule;
    VK_ERR(vkCreateShaderModule(renderInstance.device, &moduleCreateInfo, nullptr, &shaderModule), "failed to create shader module!");

    return shaderModule;
}

// Creates the basic graphics pipeline for our renderer
void MaterialPipelines::createPipelines() {
    // =========================================================================
    // Setup common attributes shared between each pipeline
    // =========================================================================

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
    // Multisampling determines anti-aliasing configuration
    VkPipelineMultisampleStateCreateInfo multisamplingInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
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
    // This push constant will be used to upload model transforms for each object
    VkPushConstantRange modelPushConstantInfo {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(Mat4<float>),
    };

    // =========================================================================
    // Setup attributes shared between each solid pipeline
    // =========================================================================

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

    // =========================================================================
    // Setup attributes used for the shadow map pipeline
    // =========================================================================

    // Rasterizer determines how we convert from primitives to fragments
    VkPipelineRasterizationStateCreateInfo shadowRasterizerStateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f,
    };
    // Color blending determines how we mix the pixels onto the framebuffer (we dont for shadow maps)
    VkPipelineColorBlendStateCreateInfo shadowColorBlendInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 0,
    };
    // This push constant will be used to upload model transforms for each object and determine which light to shadow map
    VkPushConstantRange modelLightPushConstantInfo {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(Mat4<float>) + sizeof(uint32_t),
    };

    // =========================================================================
    // Create the render pipeline layouts
    // =========================================================================

    std::array<VkDescriptorSetLayout, 2> simpleLayouts = { cameraInfoLayout, simpleEnvMirrorLayout };
    VkPipelineLayoutCreateInfo simplePipelineLayoutInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = simpleLayouts.size(),
        .pSetLayouts = simpleLayouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &modelPushConstantInfo,
    };
    VK_ERR(vkCreatePipelineLayout(renderInstance->device, &simplePipelineLayoutInfo, nullptr, &simplePipelineLayout), "failed to create pipeline layout!");

    std::array<VkDescriptorSetLayout, 3> envMirrorLayouts = { cameraInfoLayout, simpleEnvMirrorLayout, environmentLayout };
    VkPipelineLayoutCreateInfo envMirrorPipelineLayoutInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = envMirrorLayouts.size(),
        .pSetLayouts = envMirrorLayouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &modelPushConstantInfo,
    };
    VK_ERR(vkCreatePipelineLayout(renderInstance->device, &envMirrorPipelineLayoutInfo, nullptr, &envMirrorPipelineLayout), "failed to create pipeline layout!");

    std::array<VkDescriptorSetLayout, 4> lambertianLayouts = { cameraInfoLayout, lambertianLayout, environmentLayout, lightLayout };
    VkPipelineLayoutCreateInfo lambertianPipelineLayoutInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = lambertianLayouts.size(),
        .pSetLayouts = lambertianLayouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &modelPushConstantInfo,
    };
    VK_ERR(vkCreatePipelineLayout(renderInstance->device, &lambertianPipelineLayoutInfo, nullptr, &lambertianPipelineLayout), "failed to create pipeline layout!");

    std::array<VkDescriptorSetLayout, 4> pbrLayouts = { cameraInfoLayout, pbrLayout, environmentLayout, lightLayout };
    VkPipelineLayoutCreateInfo pbrPipelineLayoutInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = pbrLayouts.size(),
        .pSetLayouts = pbrLayouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &modelPushConstantInfo,
    };
    VK_ERR(vkCreatePipelineLayout(renderInstance->device, &pbrPipelineLayoutInfo, nullptr, &pbrPipelineLayout), "failed to create pipeline layout!");

    std::array<VkDescriptorSetLayout, 1> shadowLayouts = { lightLayout };
    VkPipelineLayoutCreateInfo shadowPipelineLayoutInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = shadowLayouts.size(),
        .pSetLayouts = shadowLayouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &modelLightPushConstantInfo,
    };
    VK_ERR(vkCreatePipelineLayout(renderInstance->device, &shadowPipelineLayoutInfo, nullptr, &shadowPipelineLayout), "failed to create pipeline layout!");

    // =========================================================================
    // Create the render pipelines
    // =========================================================================

    // Load in the shader modules
    VkShaderModule vertShaderModule = createShaderModule(*renderInstance, vertShaderArray, sizeof(vertShaderArray));
    VkShaderModule simpleFragShaderModule = createShaderModule(*renderInstance, simpleFragShaderArray, sizeof(simpleFragShaderArray));
    VkShaderModule environmentFragShaderModule = createShaderModule(*renderInstance, environmentFragShaderArray, sizeof(environmentFragShaderArray));
    VkShaderModule mirrorFragShaderModule = createShaderModule(*renderInstance, mirrorFragShaderArray, sizeof(mirrorFragShaderArray));
    VkShaderModule lambertianFragShaderModule = createShaderModule(*renderInstance, lambertianFragShaderArray, sizeof(lambertianFragShaderArray));
    VkShaderModule pbrFragShaderModule = createShaderModule(*renderInstance, pbrFragShaderArray, sizeof(pbrFragShaderArray));
    VkShaderModule shadowVertShaderModule = createShaderModule(*renderInstance, shadowVertShaderArray, sizeof(shadowVertShaderArray));

    // Create the shader stages for our pipeline
    VkPipelineShaderStageCreateInfo vertShaderInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertShaderModule,
        .pName = "main",
    };
    VkPipelineShaderStageCreateInfo simpleFragShaderInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = simpleFragShaderModule,
        .pName = "main",
    };
    VkPipelineShaderStageCreateInfo environmentFragShaderInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = environmentFragShaderModule,
        .pName = "main",
    };
    VkPipelineShaderStageCreateInfo mirrorFragShaderInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = mirrorFragShaderModule,
        .pName = "main",
    };
    VkPipelineShaderStageCreateInfo lambertianFragShaderInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = lambertianFragShaderModule,
        .pName = "main",
    };
    VkPipelineShaderStageCreateInfo pbrFragShaderInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = pbrFragShaderModule,
        .pName = "main",
    };
    VkPipelineShaderStageCreateInfo shadowVertShaderInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = shadowVertShaderModule,
        .pName = "main",
    };
    VkPipelineShaderStageCreateInfo simpleShaderStages[] = { vertShaderInfo, simpleFragShaderInfo };
    VkPipelineShaderStageCreateInfo environmentShaderStages[] = { vertShaderInfo, environmentFragShaderInfo };
    VkPipelineShaderStageCreateInfo mirrorShaderStages[] = { vertShaderInfo, mirrorFragShaderInfo };
    VkPipelineShaderStageCreateInfo lambertianShaderStages[] = { vertShaderInfo, lambertianFragShaderInfo };
    VkPipelineShaderStageCreateInfo pbrShaderStages[] = { vertShaderInfo, pbrFragShaderInfo };
    VkPipelineShaderStageCreateInfo shadowShaderStages[] = { shadowVertShaderInfo };

    // Finally create the render pipelines
    VkGraphicsPipelineCreateInfo renderPipelineInfos[] = {
        // Simple Material
        {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = simpleShaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssemblyInfo,
            .pViewportState = &viewportStateInfo,
            .pRasterizationState = &rasterizerStateInfo,
            .pMultisampleState = &multisamplingInfo,
            .pDepthStencilState = &depthStencilInfo,
            .pColorBlendState = &colorBlendInfo,
            .pDynamicState = &dynamicStateInfo,
            .layout = simplePipelineLayout,
            .renderPass = solidRenderPass,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        },
        // Environment Material
        {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = environmentShaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssemblyInfo,
            .pViewportState = &viewportStateInfo,
            .pRasterizationState = &rasterizerStateInfo,
            .pMultisampleState = &multisamplingInfo,
            .pDepthStencilState = &depthStencilInfo,
            .pColorBlendState = &colorBlendInfo,
            .pDynamicState = &dynamicStateInfo,
            .layout = envMirrorPipelineLayout,
            .renderPass = solidRenderPass,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        },
        // Mirror Material
        {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = mirrorShaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssemblyInfo,
            .pViewportState = &viewportStateInfo,
            .pRasterizationState = &rasterizerStateInfo,
            .pMultisampleState = &multisamplingInfo,
            .pDepthStencilState = &depthStencilInfo,
            .pColorBlendState = &colorBlendInfo,
            .pDynamicState = &dynamicStateInfo,
            .layout = envMirrorPipelineLayout,
            .renderPass = solidRenderPass,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        },
        // Lambertian Material
        {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = lambertianShaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssemblyInfo,
            .pViewportState = &viewportStateInfo,
            .pRasterizationState = &rasterizerStateInfo,
            .pMultisampleState = &multisamplingInfo,
            .pDepthStencilState = &depthStencilInfo,
            .pColorBlendState = &colorBlendInfo,
            .pDynamicState = &dynamicStateInfo,
            .layout = lambertianPipelineLayout,
            .renderPass = solidRenderPass,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        },
        // PBR Material
        {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = pbrShaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssemblyInfo,
            .pViewportState = &viewportStateInfo,
            .pRasterizationState = &rasterizerStateInfo,
            .pMultisampleState = &multisamplingInfo,
            .pDepthStencilState = &depthStencilInfo,
            .pColorBlendState = &colorBlendInfo,
            .pDynamicState = &dynamicStateInfo,
            .layout = pbrPipelineLayout,
            .renderPass = solidRenderPass,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        },
        // Shadow Map Pipeline
        {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 1,
            .pStages = shadowShaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssemblyInfo,
            .pViewportState = &viewportStateInfo,
            .pRasterizationState = &shadowRasterizerStateInfo,
            .pMultisampleState = &multisamplingInfo,
            .pDepthStencilState = &depthStencilInfo,
            .pColorBlendState = &shadowColorBlendInfo,
            .pDynamicState = &dynamicStateInfo,
            .layout = shadowPipelineLayout,
            .renderPass = shadowRenderPass,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        },
    };
    std::array<VkPipeline, 6> pipelineTargets;

    // This call takes in multiple pipelines to create. I'm guessing real apps need a bunch of pipelines, so they made a call to batch compile them.
    // In fact, this is probably where "shader compilation" and "shader caches" come in for games I play. Neat!
    VK_ERR(vkCreateGraphicsPipelines(renderInstance->device, VK_NULL_HANDLE, 6, renderPipelineInfos, nullptr, pipelineTargets.data()), "failed to create graphics pipeline!");

    simplePipeline = pipelineTargets[0];
    environmentPipeline = pipelineTargets[1];
    mirrorPipeline = pipelineTargets[2];
    lambertianPipeline = pipelineTargets[3];
    pbrPipeline = pipelineTargets[4];
    shadowPipeline = pipelineTargets[5];

    // Clean up our shader modules now that we are finished with them (the pipeline keeps its own copy)
    vkDestroyShaderModule(renderInstance->device, vertShaderModule, nullptr);
    vkDestroyShaderModule(renderInstance->device, simpleFragShaderModule, nullptr);
    vkDestroyShaderModule(renderInstance->device, environmentFragShaderModule, nullptr);
    vkDestroyShaderModule(renderInstance->device, mirrorFragShaderModule, nullptr);
    vkDestroyShaderModule(renderInstance->device, lambertianFragShaderModule, nullptr);
    vkDestroyShaderModule(renderInstance->device, pbrFragShaderModule, nullptr);
    vkDestroyShaderModule(renderInstance->device, shadowVertShaderModule, nullptr);
}
