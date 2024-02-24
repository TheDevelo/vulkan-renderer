#pragma once
#include <vulkan/vulkan.h>

#include <optional>
#include <set>
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

// Vertex struct used for our meshes
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

// Container for a local-space axis-aligned bounding box. Represented by the corners with minimal XYZ and maximal XYZ
struct AxisAlignedBoundingBox {
    Vec3<float> minCorner;
    Vec3<float> maxCorner;
};

// Scene class & container structs
struct Node {
    std::string name;

    // Node properties
    Mat4<float> transform;
    Mat4<float> invTransform;
    std::vector<uint32_t> childIndices;

    // Stored scale/rotation/translation for animation purposes (will be used to recalculate transform/invTransform)
    Vec3<float> translation;
    Vec4<float> rotation;
    Vec3<float> scale;

    // Attached mesh/cameras
    std::optional<uint32_t> meshIndex;
    std::optional<uint32_t> cameraIndex;
    std::optional<uint32_t> environmentIndex;

    // Bounding box used for BVH culling. Will be None if the node is dynamic (it or a child is animated)
    std::optional<AxisAlignedBoundingBox> bbox;

    void calculateTransforms();
};

struct Mesh {
    std::string name;

    // Vertex Buffer data
    uint32_t vertexCount;
    uint32_t vertexBufferIndex;
    uint32_t vertexBufferOffset;

    uint32_t materialIndex;

    // Bounding box data
    AxisAlignedBoundingBox bbox;
};

struct Camera {
    std::string name;

    float aspectRatio;
    float vFov;
    float nearZ;
    std::optional<float> farZ;
};

enum DriverChannel {
    DRIVER_TRANSLATION,
    DRIVER_SCALE,
    DRIVER_ROTATION,
};

enum DriverInterpolation {
    DRIVER_STEP,
    DRIVER_LINEAR,
    DRIVER_SLERP,
};

struct Driver {
    std::string name;

    uint32_t targetNode;
    DriverChannel channel;
    std::vector<float> keyTimes;
    std::vector<Vec4<float>> keyValues;
    DriverInterpolation interpolation;

    // Index of last used keyframe (first one greater than time), so that we can have O(1) keyframe lookup in the vast majority of cases (since time is almost always monotonically increasing)
    uint32_t lastKeyIndex;
};

enum class MaterialType {
    SIMPLE,
    ENVIRONMENT,
    MIRROR,
};

struct Material {
    std::string name;

    // Normal and displacement maps. nullptr means unspecified
    std::unique_ptr<CombinedImage> normalMap;
    std::unique_ptr<CombinedImage> displacementMap;

    MaterialType type;
};

struct Environment {
    std::string name;

    std::unique_ptr<CombinedCubemap> radiance;
};

struct UserCamera {
    float theta; // Measures Z angle, ranges from -PI/2 to PI/2
    float phi; // Measures XY angle
    Vec3<float> position;
};

// Camera information required to frustum cull a bounding box
struct CullingCamera {
    Mat4<float> viewMatrix;
    float halfNearWidth;
    float halfNearHeight;
    float nearZ;
    std::optional<float> farZ;
};

enum class CullingMode {
    OFF,
    FRUSTUM,
    BVH,
};

class Scene {
public:
    Scene() = default;
    explicit Scene(std::shared_ptr<RenderInstance>& renderInstance, std::string const& filename);

    void renderScene(SceneRenderInfo const& sceneRenderInfo);
    void updateCameraTransform(RenderInstance const& renderInstance); // Need render instance for the user camera aspect ratio calculation
    void moveUserCamera(UserCameraMoveEvent moveAmount, float dt);
    void rotateUserCamera(UserCameraRotateEvent rotateAmount);
    void updateAnimation(float time);
    void switchCameraByName(std::string name);

    // Cameras are public so that the "outside" can change the selected camera
    uint32_t selectedCamera = 0;
    std::vector<Camera> cameras;
    std::vector<Environment> environments; // TODO: Put back in private once I'm done setting up descriptors properly

    float minAnimTime;
    float maxAnimTime;

    bool useUserCamera;
    bool useDebugCamera;
    CullingMode cullingMode;

    ViewProjMatrices viewProj;
private:
    void renderNode(SceneRenderInfo const& sceneRenderInfo, uint32_t nodeId, Mat4<float> const& parentToWorldTransform);
    void renderMesh(SceneRenderInfo const& sceneRenderInfo, uint32_t meshId, Mat4<float> const& worldTransform);

    uint32_t vertexBufferFromBuffer(std::shared_ptr<RenderInstance>& renderInstance, const void* inBuffer, uint32_t size);
    bool computeNodeBBox(uint32_t nodeId, std::set<uint32_t>& dynamicNodes, std::set<uint32_t>& visitedNodes);

    std::optional<Mat4<float>> findCameraWTLTransform(uint32_t nodeId, uint32_t cameraId);
    bool bboxInViewFrustum(Mat4<float> const& worldTransform, AxisAlignedBoundingBox const& bbox);

    std::vector<uint32_t> sceneRoots;
    std::vector<Node> nodes;
    std::vector<Mesh> meshes;
    std::vector<Driver> drivers;
    std::vector<Material> materials;
    std::vector<CombinedBuffer> buffers;

    // We store a separate struct for the culling camera, which simplifies culling logic & makes debug mode possible
    CullingCamera cullingCamera;
    UserCamera userCamera;
};
