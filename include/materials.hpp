#pragma once
#include <vulkan/vulkan.h>

#include <memory>

#include "instance.hpp"
#include "linear.hpp"

// Class that contains all the pipelines and descriptor layouts needed to render each material
// The descriptors themselves are created & managed by the owners of the resources they point to
class MaterialPipelines {
public:
    MaterialPipelines(std::shared_ptr<RenderInstance> renderInstanceIn, VkRenderPass renderPass);
    ~MaterialPipelines();

    // Pipelines for each material
    VkPipeline simplePipeline;
    VkPipeline environmentPipeline;
    VkPipeline mirrorPipeline;

    // Pipeline layouts
    VkPipelineLayout simplePipelineLayout;
    VkPipelineLayout envMirrorPipelineLayout; // Environment and Mirror share the same layout

    // Descriptor set layouts
    VkDescriptorSetLayout cameraInfoLayout;
    VkDescriptorSetLayout environmentLayout;
    VkDescriptorSetLayout simpleEnvMirrorLayout;
    VkDescriptorSetLayout lambertianLayout;
    VkDescriptorSetLayout pbrLayout;

private:
    void createDescriptorSetLayouts();
    void createPipelines(VkRenderPass renderPass);

    std::shared_ptr<RenderInstance> renderInstance;
};

// The common vertex data structure used for all pipelines
struct Vertex {
    Vec3<float> pos;
    Vec3<float> normal;
    Vec4<float> tangent;
    Vec2<float> uv;
    Vec4<uint8_t> color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription {
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 5> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 5> attributeDescriptions {{
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
                .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = offsetof(Vertex, tangent),
            },
            {
                .location = 3,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, uv),
            },
            {
                .location = 4,
                .binding = 0,
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .offset = offsetof(Vertex, color),
            }
        }};

        return attributeDescriptions;
    }
};

// Constant material values used for the various material properties that don't have a texture map provided
// Not all materials use all of these values, but each material uses at least one
// There is no point in compacting further since these get used as UBOs, and thus must be 256-byte aligned
struct alignas(256) MaterialConstants {
    Vec3<float> albedo;
    float roughness;
    float metalness;
    bool useNormalMap;
    bool useDisplacementMap;
    bool useAlbedoMap;
    bool useRoughnessMap;
    bool useMetalnessMap;
};
