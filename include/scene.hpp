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

    // Stored scale/rotation/translation for animation purposes (will be used to recalculate transform/invTransform)
    Vec3<float> translation;
    Vec4<float> rotation;
    Vec3<float> scale;

    // Attached mesh/cameras
    std::optional<uint32_t> meshIndex;
    std::optional<uint32_t> cameraIndex;

    void calculateTransforms();
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

struct UserCamera {
    float theta; // Measures Z angle, ranges from -PI/2 to PI/2
    float phi; // Measures XY angle
    Vec3<float> position;
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

    // Cameras are public so that the "outside" can change the selected camera
    uint32_t selectedCamera = 0;
    std::vector<Camera> cameras;

    float minAnimTime;
    float maxAnimTime;

    bool useUserCamera;
    bool useDebugCamera;

    ViewProjMatrices viewProj;
private:
    void renderNode(SceneRenderInfo const& sceneRenderInfo, uint32_t nodeId, Mat4<float> const& parentToWorldTransform);
    void renderMesh(SceneRenderInfo const& sceneRenderInfo, uint32_t meshId, Mat4<float> const& worldTransform);

    uint32_t vertexBufferFromBuffer(std::shared_ptr<RenderInstance>& renderInstance, const void* inBuffer, uint32_t size);
    std::optional<Mat4<float>> findCameraWTLTransform(uint32_t nodeId, uint32_t cameraId);

    std::vector<uint32_t> sceneRoots;
    std::vector<Node> nodes;
    std::vector<Mesh> meshes;
    std::vector<Driver> drivers;
    std::vector<CombinedBuffer> buffers;

    // We store a copy of our viewProj matrices for culling as well, which lets us separate the view and culling camera for debug mode
    ViewProjMatrices cullingViewProj;
    UserCamera userCamera;
};
