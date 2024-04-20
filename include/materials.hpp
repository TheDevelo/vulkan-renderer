#pragma once
#include <vulkan/vulkan.h>

#include <memory>

#include "instance.hpp"
#include "linear.hpp"

// Forward declaration of Scene so that we can pass it into the constructor
class Scene;

// Class that contains all the pipelines and descriptor layouts needed to render each material
// The descriptors themselves are created & managed by the owners of the resources they point to
class MaterialPipelines {
public:
    MaterialPipelines(std::shared_ptr<RenderInstance> renderInstanceIn, Scene const& scene, VkFormat solidImageFormat);
    ~MaterialPipelines();

    // Render Passes
    VkRenderPass solidRenderPass;
    VkRenderPass shadowRenderPass;
    VkRenderPass mirrorLocalRenderPass;

    // Pipelines for each material + the shadow map
    VkPipeline simplePipeline;
    VkPipeline environmentPipeline;
    VkPipeline mirrorPipeline;
    VkPipeline lambertianPipeline;
    VkPipeline pbrPipeline;
    VkPipeline shadowPipeline;
    VkPipeline mirrorLocalPipeline;

    // Pipeline layouts
    VkPipelineLayout simplePipelineLayout;
    VkPipelineLayout envMirrorPipelineLayout; // Environment and Mirror share the same layout
    VkPipelineLayout lambertianPipelineLayout;
    VkPipelineLayout pbrPipelineLayout;
    VkPipelineLayout shadowPipelineLayout;
    VkPipelineLayout mirrorLocalPipelineLayout;

    // Descriptor set layouts
    VkDescriptorSetLayout cameraInfoLayout;
    VkDescriptorSetLayout lightLayout;
    VkDescriptorSetLayout environmentLayout;
    VkDescriptorSetLayout simpleEnvMirrorLayout;
    VkDescriptorSetLayout lambertianLayout;
    VkDescriptorSetLayout pbrLayout;

private:
    void createDescriptorSetLayouts(Scene const& scene);
    void createRenderPasses(VkFormat solidImageFormat);
    void createPipelines();

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
    // GLSL booleans are 4-byte aligned, so we have to match here as well.
    // Could be more efficient and pack this all into one uint32_t, but again, no point in compacting further since we aren't close to using all 256 bytes
    alignas(4) bool useNormalMap;
    alignas(4) bool useDisplacementMap;
    alignas(4) bool useAlbedoMap;
    alignas(4) bool useRoughnessMap;
    alignas(4) bool useMetalnessMap;
};
