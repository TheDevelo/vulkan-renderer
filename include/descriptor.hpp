#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

#include "buffer.hpp"
#include "materials.hpp"
#include "scene.hpp"

// Class that manages the descriptors and all shader-accessed resources which Scene doesn't own
class Descriptors {
public:
    void* cameraUniformMap;
    std::vector<void*> environmentUniformMaps;
    void* lightStorageMap;

    Descriptors(std::shared_ptr<RenderInstance> renderInstanceIn, Scene& scene, MaterialPipelines const& materialPipelines, uint32_t frameCopies);
    ~Descriptors();

private:
    void createTextureSampler();
    void createUniformBuffers(Scene const& scene, uint32_t frameCopies);
    void createDescriptorSets(Scene& scene, MaterialPipelines const& materialPipelines);

    VkSampler repeatingSampler;
    VkSampler clampedSampler;
    VkSampler shadowMapSampler;

    std::unique_ptr<CombinedBuffer> cameraUniformBuffer;
    std::vector<CombinedBuffer> environmentUniformBuffers;
    std::unique_ptr<CombinedImage> pbrBRDFImage;
    std::unique_ptr<CombinedBuffer> lightStorageBuffer;

    VkDescriptorPool descriptorPool;
    std::unique_ptr<CombinedImage> defaultDescriptorImage;

    std::shared_ptr<RenderInstance> renderInstance;
};
