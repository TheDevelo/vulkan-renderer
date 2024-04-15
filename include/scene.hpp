#pragma once
#include <vulkan/vulkan.h>

#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "buffer.hpp"
#include "instance.hpp"
#include "json.hpp"
#include "linear.hpp"
#include "materials.hpp"

// Container for a local-space axis-aligned bounding box. Represented by the corners with minimal XYZ and maximal XYZ
struct AxisAlignedBoundingBox {
    Vec3<float> minCorner;
    Vec3<float> maxCorner;
};

// Camera information that gets used as uniforms for the shaders
// Need to alignas(256), since uniform buffers must be aligned to some device limit.
// It will always be a power of 2 and at most 256, so 256 is the safe choice
struct alignas(256) CameraInfo {
    Mat4<float> view;
    Mat4<float> proj;
    Vec4<float> position;
    float exposure;
    alignas(4) bool tonemap;
};

// Environment information used in the uniforms.
struct alignas(256) EnvironmentInfo {
    Mat4<float> transform;
    AxisAlignedBoundingBox localBBox;
    uint32_t ggxMipLevels;
    alignas(4) bool local;
    alignas(4) bool empty; // Used if we have a blank cubemap
};

struct alignas(256) LightInfo {
    Mat4<float> transform;
    Mat4<float> projection;
    Vec3<float> tint;
    uint32_t type; // 0 = Sun, 1 = Sphere, 2 = Spot, uint32_t MAX = Disabled

    // Sun/Sphere/Spot Info
    float power; // Strength for Sun

    // Sun Info
    float angle;

    // Sphere/Spot Info
    float radius;
    float limit;
    alignas(4) bool useLimit;

    // Spot Info
    float fov;
    float blend;

    // Shadow Map Info if we have any
    alignas(4) bool useShadowMap;
    uint32_t shadowMapIndex;
};

// Data required for rendering, but not managed by the scene
enum class SceneRenderType {
    SOLID,
    SHADOW,
};

struct SceneRenderInfo {
    VkCommandBuffer commandBuffer;
    MaterialPipelines const& pipelines;
    SceneRenderType type;

    // Descriptor offsets are used for switching between different frame copies of descriptors
    uint32_t cameraDescriptorOffset;
    uint32_t environmentDescriptorOffset;
    uint32_t lightDescriptorOffset;

    // Shadow mapping info
    uint32_t lightIndex;
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
    std::optional<uint32_t> lightIndex;

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

    // Ancestor path to Camera, so that we can easily calculate the appropriate transforms
    std::vector<uint32_t> ancestors;
};

// Camera information required to frustum cull a bounding box
struct CullingCamera {
    Mat4<float> viewMatrix;
    float halfNearWidth;
    float halfNearHeight;
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
    LAMBERTIAN,
    PBR,
};

struct Material {
    std::string name;

    // Normal and displacement maps. nullptr means unspecified
    std::unique_ptr<CombinedImage> normalMap;
    std::unique_ptr<CombinedImage> displacementMap;

    // Albedo for Lambertian/PBR
    std::variant<Vec3<float>, std::unique_ptr<CombinedImage>> albedoMap;
    // Roughness/Metalness for PBR
    std::variant<float, std::unique_ptr<CombinedImage>> roughnessMap;
    std::variant<float, std::unique_ptr<CombinedImage>> metalnessMap;

    VkDescriptorSet descriptorSet;

    MaterialType type;
};

struct Environment {
    std::string name;

    std::unique_ptr<CombinedCubemap> radiance;
    std::unique_ptr<CombinedCubemap> lambertian;
    std::unique_ptr<CombinedCubemap> ggx;

    // Ancestor path to Environment, so that we can easily calculate the appropriate transforms
    std::vector<uint32_t> ancestors;
    EnvironmentInfo info;
    VkDescriptorSet descriptorSet;
};

// Unlike other node-instanced attachments, we have a Light copy for each instance
struct Light {
    std::string name;
    std::vector<uint32_t> ancestors;

    LightInfo info;
    uint32_t shadowMapSize;
    std::optional<CullingCamera> shadowMapCamera; // Used to cull objects during shadow map rendering
};

struct UserCamera {
    float theta; // Measures Z angle, ranges from -PI/2 to PI/2
    float phi; // Measures XY angle
    Vec3<float> position;
};

enum class CullingMode {
    OFF,
    FRUSTUM,
    BVH,
};

struct MaterialCounts {
    uint32_t simple;
    uint32_t environment;
    uint32_t mirror;
    uint32_t lambertian;
    uint32_t pbr;
};

class Scene {
public:
    Scene() = default;
    explicit Scene(std::shared_ptr<RenderInstance>& renderInstance, std::string const& filename);

    void renderScene(SceneRenderInfo const& sceneRenderInfo);

    void updateCameraTransform(RenderInstance const& renderInstance); // Need render instance for the user camera aspect ratio calculation
    void updateEnvironmentTransforms();
    void updateLightTransforms();

    void moveUserCamera(UserCameraMoveEvent moveAmount, float dt);
    void rotateUserCamera(UserCameraRotateEvent rotateAmount);
    void updateAnimation(float time);
    void switchCameraByName(std::string name);

    CombinedBuffer const& getMaterialConstantsBuffer();

    // Nodes are public for the local IBL utility's injection purpores
    std::vector<Node> nodes;

    // Cameras are public so that the "outside" can change the selected camera
    uint32_t selectedCamera = 0;
    std::vector<Camera> cameras;
    VkDescriptorSet cameraDescriptorSet;

    // Material/Environment/Light info used to create the descriptor pool, allocate descriptors, and update them
    MaterialCounts materialCounts;
    std::vector<Material> materials;
    std::vector<Environment> environments;
    std::vector<Light> lights;
    std::vector<CombinedImage> shadowMaps;
    VkDescriptorSet lightDescriptorSet;

    float minAnimTime;
    float maxAnimTime;

    bool useUserCamera;
    bool useDebugCamera;
    CullingMode cullingMode;

    CameraInfo cameraInfo;
private:
    void renderNode(SceneRenderInfo const& sceneRenderInfo, uint32_t nodeId, Mat4<float> const& parentToWorldTransform);
    void renderMesh(SceneRenderInfo const& sceneRenderInfo, uint32_t meshId, Mat4<float> const& worldTransform);

    std::vector<Vertex> loadVerticesFromAttributes(json::object const& attributeJson, uint32_t vertexCount, std::filesystem::path directory);
    void buildCombinedVertexBuffer(std::shared_ptr<RenderInstance>& renderInstance, std::vector<std::vector<Vertex>> const& meshVertices);
    void buildMaterialConstantsBuffer(std::shared_ptr<RenderInstance>& renderInstance);
    bool computeNodeBBox(uint32_t nodeId, std::set<uint32_t>& dynamicNodes, std::set<uint32_t>& visitedNodes);
    void calculateAncestors(std::vector<Light> const& lightTemplates);

    bool bboxInViewFrustum(Mat4<float> const& worldTransform, AxisAlignedBoundingBox const& bbox, CullingCamera const& cullingCamera);

    std::vector<uint32_t> sceneRoots;
    std::vector<Mesh> meshes;
    std::vector<Driver> drivers;
    std::vector<CombinedBuffer> buffers;

    // We store a separate struct for the culling camera, which simplifies culling logic & makes debug mode possible
    CullingCamera viewCullingCamera;
    UserCamera userCamera;

    uint32_t materialConstantsBufferIndex;
};
