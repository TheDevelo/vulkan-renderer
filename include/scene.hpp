#include <vulkan/vulkan.h>

#include <optional>
#include <string>
#include <vector>

#include "buffer.hpp"
#include "instance.hpp"
#include "linear.hpp"

struct SceneRenderInfo {
    VkCommandBuffer commandBuffer;
    VkPipelineLayout pipelineLayout;
};

struct Node {
    std::string name;

    // Node properties
    Mat4<float> transform;
    std::vector<uint32_t> childIndices;

    // Attached mesh/cameras
    std::optional<uint32_t> meshIndex;
    std::optional<uint32_t> cameraIndex;
};

struct Mesh {
    std::string name;

    // Vertex Buffer data
    uint32_t vertexCount;
    uint32_t vertexBufferIndex;
    uint32_t vertexBufferOffset;
};

class Scene {
public:
    void renderScene(SceneRenderInfo const& sceneRenderInfo);
//private:
    void renderNode(SceneRenderInfo const& sceneRenderInfo, uint32_t nodeId, Mat4<float> const& parentToWorldTransform);
    void renderMesh(SceneRenderInfo const& sceneRenderInfo, uint32_t meshId, Mat4<float> const& worldTransform);

    uint32_t vertexBufferFromBuffer(std::shared_ptr<RenderInstance>& renderInstance, const void* inBuffer, uint32_t size);

    std::vector<Node> nodes;
    std::vector<Mesh> meshes;
    std::vector<CombinedBuffer> buffers;
};
