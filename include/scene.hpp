#include <vulkan/vulkan.h>

#include <optional>
#include <string>
#include <vector>

#include "buffer.hpp"
#include "instance.hpp"
#include "linear.hpp"

// View and projection matrices for the vertex shader
struct ViewProjMatrices {
    Mat4<float> view;
    Mat4<float> proj;
};

// Data required for rendering, but not managed by the scene
struct SceneRenderInfo {
    VkCommandBuffer commandBuffer;
    VkPipelineLayout pipelineLayout;
};

// Scene class & container structs
struct Node {
    std::string name;

    // Node properties
    Mat4<float> transform;
    Mat4<float> invTransform;
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

struct Camera {
    std::string name;

    float aspectRatio;
    float vFov;
    float nearZ;
    std::optional<float> farZ;
};

class Scene {
public:
    Scene() = default;
    explicit Scene(std::shared_ptr<RenderInstance>& renderInstance, std::string const& filename);

    void renderScene(SceneRenderInfo const& sceneRenderInfo);
    void updateCameraTransform();

    // Cameras are public so that the "outside" can select change the selected camera
    uint32_t selectedCamera = 0;
    std::vector<Camera> cameras;

    // We store a copy of our viewProj matrices for culling as well
    ViewProjMatrices viewProj;
private:
    void renderNode(SceneRenderInfo const& sceneRenderInfo, uint32_t nodeId, Mat4<float> const& parentToWorldTransform);
    void renderMesh(SceneRenderInfo const& sceneRenderInfo, uint32_t meshId, Mat4<float> const& worldTransform);

    uint32_t vertexBufferFromBuffer(std::shared_ptr<RenderInstance>& renderInstance, const void* inBuffer, uint32_t size);
    std::optional<Mat4<float>> findCameraWTLTransform(uint32_t nodeId, uint32_t cameraId);

    std::vector<uint32_t> sceneRoots;
    std::vector<Node> nodes;
    std::vector<Mesh> meshes;
    std::vector<CombinedBuffer> buffers;
};
