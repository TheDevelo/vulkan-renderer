#include <vulkan/vulkan.h>

#include <filesystem>
#include <limits>
#include <map>
#include <set>

#include "instance.hpp"
#include "json.hpp"
#include "materials.hpp"
#include "options.hpp"
#include "scene.hpp"
#include "util.hpp"

// Helper macros for scene parsing
#define PANIC_JSON_MISSING(obj, entry, type, msg) if (!obj.contains(entry) || !obj.at(entry).is_ ## type()) { PANIC(msg); }

// Helpers for computing the bounding box of a mesh
void addVertToBBox(Vec3<float> const& position, AxisAlignedBoundingBox& bbox) {
    bbox.minCorner.x = std::min(position.x, bbox.minCorner.x);
    bbox.minCorner.y = std::min(position.y, bbox.minCorner.y);
    bbox.minCorner.z = std::min(position.z, bbox.minCorner.z);

    bbox.maxCorner.x = std::max(position.x, bbox.maxCorner.x);
    bbox.maxCorner.y = std::max(position.y, bbox.maxCorner.y);
    bbox.maxCorner.z = std::max(position.z, bbox.maxCorner.z);
};

void addTransformedBBoxToBBox(AxisAlignedBoundingBox const& srcBBox, Mat4<float> const& transform, AxisAlignedBoundingBox& dstBBox) {
    // Transform the srcBBox using a corner and 3 extents
    Vec3<float> transBBoxCorner = linear::mmul(transform, Vec4<float>(srcBBox.minCorner, 1.0f)).xyz;
    Vec3<float> transBBoxExtentX = linear::mmul(transform, Vec4<float>(srcBBox.maxCorner.x - srcBBox.minCorner.x, 0.0f, 0.0f, 0.0f)).xyz;
    Vec3<float> transBBoxExtentY = linear::mmul(transform, Vec4<float>(0.0f, srcBBox.maxCorner.y - srcBBox.minCorner.y, 0.0f, 0.0f)).xyz;
    Vec3<float> transBBoxExtentZ = linear::mmul(transform, Vec4<float>(0.0f, 0.0f, srcBBox.maxCorner.z - srcBBox.minCorner.z, 0.0f)).xyz;

    // Add each corner to the destination bounding box
    // TODO: I think theres a better way to add this, where you set transMin & transMax = corner, then add positive components of the extent to max and negative to min
    addVertToBBox(transBBoxCorner, dstBBox);
    addVertToBBox(transBBoxCorner + transBBoxExtentX, dstBBox);
    addVertToBBox(transBBoxCorner + transBBoxExtentY, dstBBox);
    addVertToBBox(transBBoxCorner + transBBoxExtentX + transBBoxExtentY, dstBBox);
    addVertToBBox(transBBoxCorner + transBBoxExtentZ, dstBBox);
    addVertToBBox(transBBoxCorner + transBBoxExtentX + transBBoxExtentZ, dstBBox);
    addVertToBBox(transBBoxCorner + transBBoxExtentY + transBBoxExtentZ, dstBBox);
    addVertToBBox(transBBoxCorner + transBBoxExtentX + transBBoxExtentY + transBBoxExtentZ, dstBBox);
};

// Constructor for scene directly from a JSON s72 file
Scene::Scene(std::shared_ptr<RenderInstance>& renderInstance, std::string const& filename) {
    std::filesystem::path directory = std::filesystem::absolute(filename).parent_path();
    // Load the file into a JSON tree
    json::Value sceneTree = json::load(filename);

    // Check we have a s72-v1 file
    if (!sceneTree.is_arr()) {
        PANIC("Scene loading error: scene JSON is not an array");
    }
    json::array const& sceneArr = sceneTree.as_arr();
    if (sceneArr.size() == 0 || !sceneArr[0].is_str() || sceneArr[0].as_str() != "s72-v1") {
        PANIC("Scene loading error: first element of scene JSON is not correct magic string");
    }

    // Iterate through all of our objects, and construct maps from global IDs to per-type IDs
    // This is because s72 refers to objects by global ID, while Scene uses per-type IDs
    uint32_t sceneId = 0;
    std::map<uint32_t, uint32_t> nodeIdMap;
    std::map<uint32_t, uint32_t> meshIdMap;
    std::map<uint32_t, uint32_t> cameraIdMap;
    std::map<uint32_t, uint32_t> driverIdMap;
    std::map<uint32_t, uint32_t> materialIdMap;
    std::map<uint32_t, uint32_t> environmentIdMap;
    std::map<uint32_t, uint32_t> lightIdMap;
    materialIdMap.insert_or_assign(0, 0); // Insert a dummy element to reserve for the default material

    for (auto e = sceneArr.begin() + 1; e != sceneArr.end(); e++) {
        uint32_t s72Id = std::distance(sceneArr.begin(), e);
        if (!e->is_obj()) {
            PANIC("Scene loading error: scene JSON has a non-object element");
        }
        json::object const& object = e->as_obj();

        // Check we have a valid object (needs a name and type string)
        PANIC_JSON_MISSING(object, "type", str, "Scene loading error: s72 object has an unspecified type");
        PANIC_JSON_MISSING(object, "name", str, "Scene loading error: s72 object has an invalid name");

        std::string const& type = object.at("type").as_str();
        if (type == "SCENE") {
            if (sceneId != 0) {
                PANIC("Scene loading error: s72 has multiple scene objects");
            }
            sceneId = s72Id;
        }
        else if (type == "NODE") {
            uint32_t nodeId = nodeIdMap.size();
            nodeIdMap.insert_or_assign(s72Id, nodeId);
        }
        else if (type == "MESH") {
            uint32_t meshId = meshIdMap.size();
            meshIdMap.insert_or_assign(s72Id, meshId);
        }
        else if (type == "CAMERA") {
            uint32_t cameraId = cameraIdMap.size();
            cameraIdMap.insert_or_assign(s72Id, cameraId);
        }
        else if (type == "DRIVER") {
            uint32_t driverId = driverIdMap.size();
            driverIdMap.insert_or_assign(s72Id, driverId);
        }
        else if (type == "MATERIAL") {
            uint32_t materialId = materialIdMap.size();
            materialIdMap.insert_or_assign(s72Id, materialId);
        }
        else if (type == "ENVIRONMENT") {
            uint32_t environmentId = environmentIdMap.size();
            environmentIdMap.insert_or_assign(s72Id, environmentId);
        }
        else if (type == "LIGHT") {
            uint32_t lightId = lightIdMap.size();
            lightIdMap.insert_or_assign(s72Id, lightId);
        }
        else {
            PANIC("Scene loading error: s72 object has an invalid type");
        }
    }

    // Set scene roots
    json::object const& sceneRootObj = sceneArr[sceneId].as_obj();
    PANIC_JSON_MISSING(sceneRootObj, "roots", arr, "Scene loading error: scene object doesn't have scene roots");
    for (json::Value const& rootVal : sceneRootObj.at("roots").as_arr()) {
        sceneRoots.emplace_back(nodeIdMap.at(rootVal.as_num()));
    }

    // Iterate through all the nodes and construct their Node representation
    nodes.reserve(nodeIdMap.size());
    for (auto idPair : nodeIdMap) {
        uint32_t s72Id = idPair.first;
        json::object const& nodeObj = sceneArr[s72Id].as_obj();

        // Get translation, rotation, and scale vectors
        Vec3<float> t = Vec3<float>(0.0f);
        if (nodeObj.contains("translation") && nodeObj.at("translation").is_vec3f()) {
            t = nodeObj.at("translation").as_vec3f();
        }
        Vec4<float> q = Vec4<float>(0.0f, 0.0f, 0.0f, 1.0f);
        if (nodeObj.contains("rotation") && nodeObj.at("rotation").is_vec4f()) {
            q = nodeObj.at("rotation").as_vec4f();
        }
        Vec3<float> s = Vec3<float>(1.0f);
        if (nodeObj.contains("scale") && nodeObj.at("scale").is_vec3f()) {
            s = nodeObj.at("scale").as_vec3f();
        }

        Node node {
            .name = nodeObj.at("name").as_str(),
            .translation = t,
            .rotation = q,
            .scale = s,
        };
        node.calculateTransforms();

        // Add mesh, camera, and environment indices
        if (nodeObj.contains("mesh")) {
            uint32_t jsonId = nodeObj.at("mesh").as_num();
            node.meshIndex = meshIdMap.at(jsonId);
        }
        if (nodeObj.contains("camera")) {
            uint32_t jsonId = nodeObj.at("camera").as_num();
            node.cameraIndex = cameraIdMap.at(jsonId);
        }
        if (nodeObj.contains("environment")) {
            uint32_t jsonId = nodeObj.at("environment").as_num();
            node.environmentIndex = environmentIdMap.at(jsonId);
        }
        if (nodeObj.contains("light")) {
            uint32_t jsonId = nodeObj.at("light").as_num();
            node.lightIndex = lightIdMap.at(jsonId);
        }

        // Add child nodes
        if (nodeObj.contains("children") && nodeObj.at("children").is_arr()) {
            for (json::Value const& e : nodeObj.at("children").as_arr()) {
                uint32_t jsonId = e.as_num();
                node.childIndices.emplace_back(nodeIdMap.at(jsonId));
            }
        }

        nodes.push_back(node);
    }

    // Iterate through all the meshes and construct their Mesh representation, as well as loading their vertices
    // TODO: Properly handle topology and indexed meshes
    std::vector<std::vector<Vertex>> meshVertices;
    meshes.reserve(meshIdMap.size());
    meshVertices.reserve(meshIdMap.size());
    for (auto idPair : meshIdMap) {
        uint32_t s72Id = idPair.first;
        json::object const& meshObj = sceneArr[s72Id].as_obj();

        PANIC_JSON_MISSING(meshObj, "count", num, "Scene loading error: mesh is missing vertex count");
        if (meshObj.at("count").as_num() == 0.0) {
            PANIC("Scene loading error: mesh has no vertices");
        }
        PANIC_JSON_MISSING(meshObj, "attributes", obj, "Scene loading error: mesh doesn't have attributes");

        json::object const& attribObj = meshObj.at("attributes").as_obj();

        Mesh mesh {
            .name = meshObj.at("name").as_str(),
            .vertexCount = static_cast<uint32_t>(meshObj.at("count").as_num()),

            .bbox = AxisAlignedBoundingBox {
                .minCorner = Vec3<float>(std::numeric_limits<float>::max()),
                .maxCorner = Vec3<float>(std::numeric_limits<float>::min()),
            }
        };

        // Load the mesh attributes into a vertex buffer
        std::vector<Vertex> vertices = loadVerticesFromAttributes(attribObj, mesh.vertexCount, directory);

        // Calculate the bounding box for our mesh
        for (uint32_t i = 0; i < mesh.vertexCount; i++) {
            addVertToBBox(vertices[i].pos, mesh.bbox);
        }

        // Add the material
        if (meshObj.contains("material") && meshObj.at("material").is_num()) {
            uint32_t jsonId = meshObj.at("material").as_num();
            mesh.materialIndex = materialIdMap.at(jsonId);
        }
        else {
            // Material Index of 0 = default material
            mesh.materialIndex = 0;
        }

        meshes.push_back(mesh);
        meshVertices.push_back(std::move(vertices));
    }

    // Create the combined vertex buffer from the loaded mesh vertices
    buildCombinedVertexBuffer(renderInstance, meshVertices);

    // Iterate through all the cameras and construct their Camera representation
    cameras.reserve(cameraIdMap.size());
    for (auto idPair : cameraIdMap) {
        uint32_t s72Id = idPair.first;
        json::object const& cameraObj = sceneArr[s72Id].as_obj();

        PANIC_JSON_MISSING(cameraObj, "perspective", obj, "Scene loading error: camera doesn't have perspective values");
        json::object const& perspObj = cameraObj.at("perspective").as_obj();
        PANIC_JSON_MISSING(perspObj, "aspect", num, "Scene loading error: camera perspective is incomplete");
        PANIC_JSON_MISSING(perspObj, "vfov", num, "Scene loading error: camera perspective is incomplete");
        PANIC_JSON_MISSING(perspObj, "near", num, "Scene loading error: camera perspective is incomplete");

        Camera camera {
            .name = cameraObj.at("name").as_str(),
            .aspectRatio = static_cast<float>(perspObj.at("aspect").as_num()),
            .vFov = static_cast<float>(perspObj.at("vfov").as_num()),
            .nearZ = static_cast<float>(perspObj.at("near").as_num()),
        };
        if (perspObj.contains("far")) {
            camera.farZ = perspObj.at("far").as_num();
        }

        cameras.push_back(camera);
    }

    // Iterate through all the drivers and construct their Driver representation
    std::set<uint32_t> dynamicNodes;
    minAnimTime = std::numeric_limits<float>::max();
    maxAnimTime = std::numeric_limits<float>::min();
    drivers.reserve(driverIdMap.size());
    for (auto idPair : driverIdMap) {
        uint32_t s72Id = idPair.first;
        json::object const& driverObj = sceneArr[s72Id].as_obj();

        // Get the driver's target
        PANIC_JSON_MISSING(driverObj, "node", num, "Scene loading error: driver object doesn't have a target node");
        uint32_t targetJsonId = driverObj.at("node").as_num();

        // Get the target channel
        PANIC_JSON_MISSING(driverObj, "channel", str, "Scene loading error: driver object doesn't have a target channel");
        std::string const& channelStr = driverObj.at("channel").as_str();
        DriverChannel channel;
        uint32_t channelWidth;
        if (channelStr == "translation") {
            channel = DRIVER_TRANSLATION;
            channelWidth = 3;
        }
        else if (channelStr == "rotation") {
            channel = DRIVER_ROTATION;
            channelWidth = 4;
        }
        else if (channelStr == "scale") {
            channel = DRIVER_SCALE;
            channelWidth = 3;
        }
        else {
            PANIC("Scene loading error: driver object has an invalid channel");
        }

        Driver driver {
            .targetNode = nodeIdMap.at(targetJsonId),
            .channel = channel,
            .interpolation = DRIVER_LINEAR,
            .lastKeyIndex = 0,
        };

        // Get the keyframe times
        PANIC_JSON_MISSING(driverObj, "times", arr, "Scene loading error: driver object doesn't have keyframe times");
        for (json::Value const& time : driverObj.at("times").as_arr()) {
            if (!time.is_num()) {
                PANIC("Scene loading error: driver keyframe times contains invalid entry");
            }
            driver.keyTimes.emplace_back(time.as_num());
        }
        if (driver.keyTimes.size() < 2) {
            PANIC("Scene loading error: driver does not have enough key frames (>= 2)");
        }

        // Get the keyframe values
        PANIC_JSON_MISSING(driverObj, "values", arr, "Scene loading error: driver object doesn't have keyframe values");
        if (driverObj.at("values").as_arr().size() != channelWidth * driver.keyTimes.size()) {
            PANIC("Scene loading error: driver has wrong number of keyframe values");
        }
        json::array const& driverValues = driverObj.at("values").as_arr();
        for (uint32_t i = 0; i < driverValues.size(); i += channelWidth) {
            Vec4<float> value = Vec4<float>(0.0f);
            for (uint32_t j = 0; j < channelWidth; j++) {
                if (!driverValues[i + j].is_num()) {
                    PANIC("Scene loading error: driver has an invalid keyframe value");
                }
                value[j] = driverValues[i + j].as_num();
            }

            driver.keyValues.emplace_back(value);
        }

        // Get the keyframe interpolation
        if (driverObj.contains("interpolation") && driverObj.at("interpolation").is_str()) {
            std::string const& interpStr = driverObj.at("interpolation").as_str();
            if (interpStr == "STEP") {
                driver.interpolation = DRIVER_STEP;
            }
            else if (interpStr == "LINEAR") {
                driver.interpolation = DRIVER_LINEAR;
            }
            else if (interpStr == "SLERP") {
                driver.interpolation = DRIVER_SLERP;
            }
            else {
                PANIC("Scene loading error: driver has an invalid interpolation value");
            }
        }

        minAnimTime = std::min(driver.keyTimes[0], minAnimTime);
        maxAnimTime = std::max(driver.keyTimes[driver.keyTimes.size() - 1], maxAnimTime);
        dynamicNodes.insert(driver.targetNode);
        drivers.push_back(driver);
    }

    auto loadTextureObj = [&](json::object const& obj, std::string expectedType) {
        PANIC_JSON_MISSING(obj, "src", str, "Scene loading error: texture does not have a source");

        std::string textureType = "2D";
        if (obj.contains("type") && obj.at("type").is_str()) {
            textureType = obj.at("type").as_str();
        }
        if (expectedType != textureType) {
            PANIC("Scene loading error: texture has an unexpected type");
        }

        std::string imageFormat = "linear";
        if (obj.contains("format") && obj.at("format").is_str()) {
            imageFormat = obj.at("format").as_str();
        }
        // TODO: Actually use the image format to do something

        return directory / obj.at("src").as_str();
    };

    // Add the default material to the material list. Then erase the default material ID entry so we don't try to load it.
    materials.reserve(materialIdMap.size());
    materials.push_back(Material {
        .name = "Default Material",
        .type = MaterialType::SIMPLE,
    });
    materialIdMap.erase(0);

    // Iterate through all the materials and construct their Material representation
    materialCounts.simple = 1; // Have 1 simple material from the default material
    materialCounts.environment = 0;
    materialCounts.mirror = 0;
    materialCounts.lambertian = 0;
    materialCounts.pbr = 0;
    for (auto idPair : materialIdMap) {
        uint32_t s72Id = idPair.first;
        json::object const& materialObj = sceneArr[s72Id].as_obj();

        Material material {
            .name = materialObj.at("name").as_str(),
            .normalMap = nullptr,
            .displacementMap = nullptr,
            .albedoMap = Vec3<float>(1.0),
            .roughnessMap = 1.0f,
            .metalnessMap = 0.0f,
        };

        // Load the normal & displacement maps
        if (materialObj.contains("normalMap") && materialObj.at("normalMap").is_obj()) {
            json::object const& normalMapObj = materialObj.at("normalMap").as_obj();
            std::filesystem::path filePath = loadTextureObj(normalMapObj, "2D");
            material.normalMap = loadImage(renderInstance, filePath.string(), VK_FORMAT_R8G8B8A8_UNORM);
        }
        if (materialObj.contains("displacementMap") && materialObj.at("displacementMap").is_obj()) {
            json::object const& dispMapObj = materialObj.at("displacementMap").as_obj();
            std::filesystem::path filePath = loadTextureObj(dispMapObj, "2D");
            material.displacementMap = loadImage(renderInstance, filePath.string(), VK_FORMAT_R8G8B8A8_UNORM);
        }

        // Load material specific attributes
        if (materialObj.contains("simple")) {
            material.type = MaterialType::SIMPLE;
            materialCounts.simple += 1;
        }
        else if (materialObj.contains("environment")) {
            material.type = MaterialType::ENVIRONMENT;
            materialCounts.environment += 1;
        }
        else if (materialObj.contains("mirror")) {
            material.type = MaterialType::MIRROR;
            materialCounts.mirror += 1;
        }
        else if (materialObj.contains("lambertian") && materialObj.at("lambertian").is_obj()) {
            material.type = MaterialType::LAMBERTIAN;
            materialCounts.lambertian += 1;
            json::object const& lambertianObj = materialObj.at("lambertian").as_obj();

            // Load Lambertian albedo
            if (lambertianObj.contains("albedo") && lambertianObj.at("albedo").is_vec3f()) {
                material.albedoMap = lambertianObj.at("albedo").as_vec3f();
            }
            else if (lambertianObj.contains("albedo") && lambertianObj.at("albedo").is_obj()) {
                json::object const& albedoObj = lambertianObj.at("albedo").as_obj();
                std::filesystem::path filePath = loadTextureObj(albedoObj, "2D");
                material.albedoMap = loadImage(renderInstance, filePath.string(), VK_FORMAT_R8G8B8A8_SRGB);
            }
        }
        else if (materialObj.contains("pbr") && materialObj.at("pbr").is_obj()) {
            material.type = MaterialType::PBR;
            materialCounts.pbr += 1;
            json::object const& pbrObj = materialObj.at("pbr").as_obj();

            // Load PBR albedo
            if (pbrObj.contains("albedo") && pbrObj.at("albedo").is_vec3f()) {
                material.albedoMap = pbrObj.at("albedo").as_vec3f();
            }
            else if (pbrObj.contains("albedo") && pbrObj.at("albedo").is_obj()) {
                json::object const& albedoObj = pbrObj.at("albedo").as_obj();
                std::filesystem::path filePath = loadTextureObj(albedoObj, "2D");
                material.albedoMap = loadImage(renderInstance, filePath.string(), VK_FORMAT_R8G8B8A8_SRGB);
            }

            // Load PBR roughness
            if (pbrObj.contains("roughness") && pbrObj.at("roughness").is_num()) {
                material.roughnessMap = static_cast<float>(pbrObj.at("roughness").as_num());
            }
            else if (pbrObj.contains("roughness") && pbrObj.at("roughness").is_obj()) {
                json::object const& roughnessObj = pbrObj.at("roughness").as_obj();
                std::filesystem::path filePath = loadTextureObj(roughnessObj, "2D");
                material.roughnessMap = loadImage(renderInstance, filePath.string(), VK_FORMAT_R8G8B8A8_UNORM);
            }

            // Load PBR metalness
            if (pbrObj.contains("metalness") && pbrObj.at("metalness").is_num()) {
                material.metalnessMap = static_cast<float>(pbrObj.at("metalness").as_num());
            }
            else if (pbrObj.contains("metalness") && pbrObj.at("metalness").is_obj()) {
                json::object const& metalnessObj = pbrObj.at("metalness").as_obj();
                std::filesystem::path filePath = loadTextureObj(metalnessObj, "2D");
                material.metalnessMap = loadImage(renderInstance, filePath.string(), VK_FORMAT_R8G8B8A8_UNORM);
            }
        }
        else {
            PANIC("Scene loading error: material does not contain a valid type");
        }

        materials.push_back(std::move(material));
    }

    // Build a buffer containing a MaterialConstants for each material
    buildMaterialConstantsBuffer(renderInstance);

    // Iterate through all the environments and construct their Environment representation
    environments.reserve(environmentIdMap.size());
    for (auto idPair : environmentIdMap) {
        uint32_t s72Id = idPair.first;
        json::object const& envObj = sceneArr[s72Id].as_obj();

        Environment env {
            .name = envObj.at("name").as_str(),
            .info = EnvironmentInfo {
                .ggxMipLevels = 1,
            },
        };

        // Load the radiance map
        PANIC_JSON_MISSING(envObj, "radiance", obj, "Scene loading error: environment does not contain a radiance map");
        json::object const& radianceObj = envObj.at("radiance").as_obj();
        std::filesystem::path filePath = loadTextureObj(radianceObj, "cube");
        env.radiance = loadCubemap(renderInstance, filePath.string());

        // Load the pre-integrated Lambertian cubemap
        std::filesystem::path lambertianPath = filePath;
        lambertianPath.replace_extension(".lambertian.png");
        env.lambertian = loadCubemap(renderInstance, lambertianPath.string());

        // Load the GGX cubemap stack (with the standard environment as level 0)
        while (true) {
            std::filesystem::path ggxMipmapPath = filePath;
            ggxMipmapPath.replace_extension(string_format(".ggx%u.png", env.info.ggxMipLevels));
            if (!std::filesystem::exists(ggxMipmapPath)) {
                break;
            }
            env.info.ggxMipLevels += 1;
        }
        env.ggx = loadCubemap(renderInstance, filePath.string(), env.info.ggxMipLevels);
        for (uint32_t mipLevel = 1; mipLevel < env.info.ggxMipLevels; mipLevel++) {
            std::filesystem::path ggxMipmapPath = filePath;
            ggxMipmapPath.replace_extension(string_format(".ggx%u.png", mipLevel));
            loadMipmapIntoCubemap(renderInstance, *env.ggx, ggxMipmapPath.string(), mipLevel);
        }

        environments.push_back(std::move(env));
    }

    // Iterate through all the lights and construct their Light representation
    for (auto idPair : lightIdMap) {
        uint32_t s72Id = idPair.first;
        json::object const& lightObj = sceneArr[s72Id].as_obj();

        Light light {
            .name = lightObj.at("name").as_str(),
            .tint = Vec3<float>(1.0),
            .shadowMapSize = 0,
        };

        if (lightObj.contains("tint") && lightObj.at("tint").is_vec3f()) {
            light.tint = lightObj.at("tint").as_vec3f();
        }

        if (lightObj.contains("shadow") && lightObj.at("shadow").is_num()) {
            light.shadowMapSize = lightObj.at("shadow").as_num();
        }

        if (lightObj.contains("sun") && lightObj.at("sun").is_obj()) {
            json::object const& sunObj = lightObj.at("sun").as_obj();
            PANIC_JSON_MISSING(sunObj, "strength", num, "Scene loading error: sun light missing strength");
            PANIC_JSON_MISSING(sunObj, "angle", num, "Scene loading error: sun light missing angle");

            light.lightInfo = LightSun {
                .strength = static_cast<float>(sunObj.at("strength").as_num()),
                .angle = static_cast<float>(sunObj.at("angle").as_num()),
            };
        }
        else if (lightObj.contains("sphere") && lightObj.at("sphere").is_obj()) {
            json::object const& sphereObj = lightObj.at("sphere").as_obj();
            PANIC_JSON_MISSING(sphereObj, "radius", num, "Scene loading error: sphere light missing radius");
            PANIC_JSON_MISSING(sphereObj, "power", num, "Scene loading error: sphere light missing power");

            LightSphere sphere {
                .radius = static_cast<float>(sphereObj.at("radius").as_num()),
                .power = static_cast<float>(sphereObj.at("power").as_num()),
            };
            if (sphereObj.contains("limit") && sphereObj.at("limit").is_num()) {
                sphere.limit = sphereObj.at("limit").as_num();
            }

            light.lightInfo = sphere;
        }
        else if (lightObj.contains("spot") && lightObj.at("spot").is_obj()) {
            json::object const& spotObj = lightObj.at("spot").as_obj();
            PANIC_JSON_MISSING(spotObj, "radius", num, "Scene loading error: spot light missing radius");
            PANIC_JSON_MISSING(spotObj, "power", num, "Scene loading error: spot light missing power");
            PANIC_JSON_MISSING(spotObj, "fov", num, "Scene loading error: spot light missing fov");
            PANIC_JSON_MISSING(spotObj, "blend", num, "Scene loading error: spot light missing blend");

            LightSpot spot {
                .radius = static_cast<float>(spotObj.at("radius").as_num()),
                .power = static_cast<float>(spotObj.at("power").as_num()),
                .fov = static_cast<float>(spotObj.at("fov").as_num()),
                .blend = static_cast<float>(spotObj.at("blend").as_num()),
            };
            if (spotObj.contains("limit") && spotObj.at("limit").is_num()) {
                spot.limit = spotObj.at("limit").as_num();
            }

            light.lightInfo = spot;
        }
        else {
            PANIC("Scene loading error: light doesn't contain a valid light type");
        }

        lights.push_back(std::move(light));
    }

    // Set the default camera
    if (options::getDefaultCamera().has_value()) {
        // Start with the selected camera as the one with the specified name
        std::string cameraName = options::getDefaultCamera().value();
        switchCameraByName(cameraName);
        useUserCamera = false;
    }
    else {
        // Default case: just use the user camera
        selectedCamera = 0;
        useUserCamera = true;
    }
    useDebugCamera = false;
    userCamera = UserCamera {
        .theta = 0.0f,
        .phi = 0.0f,
        .position = Vec3<float>(0.0f),
    };

    // Compute bounding boxes for all nodes that aren't animated, and don't have animated children
    std::set<uint32_t> visitedNodes;
    for (uint32_t rootNode : sceneRoots) {
        computeNodeBBox(rootNode, dynamicNodes, visitedNodes);
    }

    calculateAncestors();
    cullingMode = options::getDefaultCullingMode();
    cameraInfo.exposure = 1.0f;
}

// Helper to recursively compute the bounding box of a node with dynamic programming
// Return value is whether you are a static node: dynamic nodes return false
bool Scene::computeNodeBBox(uint32_t nodeId, std::set<uint32_t>& dynamicNodes, std::set<uint32_t>& visitedNodes) {
    // Return if we have already visited the node before
    if (visitedNodes.contains(nodeId)) {
        return !dynamicNodes.contains(nodeId);
    }
    Node& node = nodes[nodeId];

    bool bboxChanged = false;
    AxisAlignedBoundingBox bbox {
        .minCorner = Vec3<float>(std::numeric_limits<float>::max()),
        .maxCorner = Vec3<float>(std::numeric_limits<float>::min()),
    };

    // Add the bounding box of the attached mesh, if we have one
    if (node.meshIndex.has_value()) {
        Mesh const& mesh = meshes[node.meshIndex.value()];
        addTransformedBBoxToBBox(mesh.bbox, node.transform, bbox);
        bboxChanged = true;
    }

    // Add the bounding boxes of all the children
    // NOTE: even if we find a dynamic child, we still need to go through the rest so that their bounding boxes can be computed
    bool dynamic = dynamicNodes.contains(nodeId);
    for (uint32_t childId : node.childIndices) {
        if (computeNodeBBox(childId, dynamicNodes, visitedNodes)) {
            Node& child = nodes[childId];
            if (!dynamic && child.bbox.has_value()) {
                // Add the child's bounding box to ours
                addTransformedBBoxToBBox(child.bbox.value(), node.transform, bbox);
                bboxChanged = true;
            }
        }
        else {
            // Child node is dynamic, so we are dynamic now
            dynamic = true;
        }
    }

    if (dynamic) {
        dynamicNodes.insert(nodeId);
    }
    else if (bboxChanged) {
        node.bbox = bbox;
    }

    visitedNodes.insert(nodeId);
    return !dynamic;
}

void Node::calculateTransforms() {
    transform = linear::localToParent(translation, rotation, scale);
    invTransform = linear::parentToLocal(translation, rotation, scale);
}

void Scene::switchCameraByName(std::string name) {
    bool foundCamera = false;
    for (uint32_t i = 0; i < cameras.size(); i++) {
        if (cameras[i].name == name) {
            selectedCamera = i;
            foundCamera = true;
            break;
        }
    }

    if (!foundCamera) {
        PANIC(string_format("failed to find camera %s", name.c_str()));
    }
}

// Helper to calculate the ancestor path to each Camera/Environment for easy transform calculation
void Scene::calculateAncestors() {
    std::vector<uint32_t> ancestors;
    auto traverse = [&](const auto& recurse, uint32_t nodeId) -> void {
        ancestors.push_back(nodeId);

        if (nodeId >= nodes.size()) {
            PANIC(string_format("tried to traverse node %u out of range!", nodeId));
        }
        Node& node = nodes[nodeId];

        // Add ancestors to any child cameras/environments
        if (node.cameraIndex.has_value()) {
            if (node.cameraIndex.value() >= cameras.size()) {
                PANIC(string_format("tried to traverse camera %u out of range!", node.cameraIndex.value()));
            }
            Camera& camera = cameras[node.cameraIndex.value()];

            if (camera.ancestors.size() == 0) {
                camera.ancestors = ancestors;
            }
        }
        if (node.environmentIndex.has_value()) {
            if (node.environmentIndex.value() >= environments.size()) {
                PANIC(string_format("tried to traverse environment %u out of range!", node.environmentIndex.value()));
            }
            Environment& environment = environments[node.environmentIndex.value()];

            if (environment.ancestors.size() == 0) {
                environment.ancestors = ancestors;
            }
        }

        for (uint32_t childId : node.childIndices) {
            recurse(recurse, childId);
        }

        ancestors.pop_back();
    };

    for (uint32_t rootNode : sceneRoots) {
        traverse(traverse, rootNode);
    }
}

CombinedBuffer const& Scene::getMaterialConstantsBuffer() {
    return buffers[materialConstantsBufferIndex];
}

void Scene::buildMaterialConstantsBuffer(std::shared_ptr<RenderInstance>& renderInstance) {
    uint32_t size = materials.size() * sizeof(MaterialConstants);

    // Create the appropriate buffers
    CombinedBuffer& materialConstantsBuffer = buffers.emplace_back(renderInstance, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CombinedBuffer stagingBuffer = CombinedBuffer(renderInstance, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    materialConstantsBufferIndex = buffers.size() - 1;

    // Fill the staging buffer with the appropriate material constants
    void* stagingBufferMap;
    vkMapMemory(renderInstance->device, stagingBuffer.bufferMemory, 0, size, 0, &stagingBufferMap);
    for (size_t i = 0; i < materials.size(); i++) {
        Material const& material = materials[i];
        MaterialConstants* dest = reinterpret_cast<MaterialConstants*>(stagingBufferMap) + i;

        if (material.normalMap == nullptr) {
            dest->useNormalMap = false;
        }
        else {
            dest->useNormalMap = true;
        }

        if (material.displacementMap == nullptr) {
            dest->useDisplacementMap = false;
        }
        else {
            dest->useDisplacementMap = true;
        }

        if (holds_alternative<Vec3<float>>(material.albedoMap)) {
            // Albedo is a constant
            dest->useAlbedoMap = false;
            dest->albedo = get<Vec3<float>>(material.albedoMap);
        }
        else {
            // Albedo uses an actual map
            dest->useAlbedoMap = true;
        }

        if (holds_alternative<float>(material.roughnessMap)) {
            // Roughness is a constant
            dest->useRoughnessMap = false;
            dest->roughness = get<float>(material.roughnessMap);
        }
        else {
            // Roughness uses an actual map
            dest->useRoughnessMap = true;
        }

        if (holds_alternative<float>(material.metalnessMap)) {
            // Metalness is a constant
            dest->useMetalnessMap = false;
            dest->metalness = get<float>(material.metalnessMap);
        }
        else {
            // Metalness uses an actual map
            dest->useMetalnessMap = true;
        }
    }
    vkUnmapMemory(renderInstance->device, stagingBuffer.bufferMemory);

    // Copy from staging buffer to vertex buffer
    VkCommandBuffer commandBuffer = beginSingleUseCBuffer(*renderInstance);
    BufferCopy bufferCopyInfos[] = {
        {
            .srcBuffer = stagingBuffer.buffer,
            .srcOffset = 0,
            .dstBuffer = materialConstantsBuffer.buffer,
            .dstOffset = 0,
            .size = size,
        },
    };
    copyBuffers(commandBuffer, bufferCopyInfos, 1);
    endSingleUseCBuffer(*renderInstance, commandBuffer);
}
