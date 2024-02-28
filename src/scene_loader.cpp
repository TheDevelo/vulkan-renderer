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
    materialIdMap.insert_or_assign(0, 0); // Insert a dummy element to reserve for the default material

    for (auto e = sceneArr.begin() + 1; e != sceneArr.end(); e++) {
        uint32_t s72Id = std::distance(sceneArr.begin(), e);
        if (!e->is_obj()) {
            PANIC("Scene loading error: scene JSON has a non-object element");
        }
        json::object const& object = e->as_obj();

        // Check we have a valid object (needs a name and type string)
        if (!object.contains("type") || !object.at("type").is_str()) {
            PANIC("Scene loading error: s72 object has an unspecified type");
        }
        if (!object.contains("name") || !object.at("name").is_str()) {
            PANIC("Scene loading error: s72 object has an invalid name");
        }

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
        else {
            PANIC("Scene loading error: s72 object has an invalid type");
        }
    }

    // Set scene roots
    json::object const& sceneRootObj = sceneArr[sceneId].as_obj();
    if (!sceneRootObj.contains("roots") || !sceneRootObj.at("roots").is_arr()) {
        PANIC("Scene loading error: scene object doesn't have scene roots");
    }
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
        if (nodeObj.contains("translation") && nodeObj.at("translation").is_arr() && nodeObj.at("translation").as_arr().size() == 3) {
            json::array const& vec = nodeObj.at("translation").as_arr();
            t.x = vec[0].as_num();
            t.y = vec[1].as_num();
            t.z = vec[2].as_num();
        }
        Vec4<float> q = Vec4<float>(0.0f, 0.0f, 0.0f, 1.0f);
        if (nodeObj.contains("rotation") && nodeObj.at("rotation").is_arr() && nodeObj.at("rotation").as_arr().size() == 4) {
            json::array const& vec = nodeObj.at("rotation").as_arr();
            q.x = vec[0].as_num();
            q.y = vec[1].as_num();
            q.z = vec[2].as_num();
            q.w = vec[3].as_num();
        }
        Vec3<float> s = Vec3<float>(1.0f);
        if (nodeObj.contains("scale") && nodeObj.at("scale").is_arr() && nodeObj.at("scale").as_arr().size() == 3) {
            json::array const& vec = nodeObj.at("scale").as_arr();
            s.x = vec[0].as_num();
            s.y = vec[1].as_num();
            s.z = vec[2].as_num();
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

        if (!meshObj.contains("count") || !meshObj.at("count").is_num()) {
            PANIC("Scene loading error: mesh is missing vertex count");
        }
        else if (meshObj.at("count").as_num() == 0.0) {
            PANIC("Scene loading error: mesh has no vertices");
        }

        if (!meshObj.contains("attributes") || !meshObj.at("attributes").is_obj()) {
            PANIC("Scene loading error: mesh doesn't have attributes");
        }
        json::object const& attribObj = meshObj.at("attributes").as_obj();

        Mesh mesh {
            .name = meshObj.at("name").as_str(),
            .vertexCount = static_cast<uint32_t>(meshObj.at("count").as_num()),
            /*
            .vertexBufferIndex = 0,
            .vertexBufferOffset = curBufferCount * static_cast<uint32_t>(sizeof(Vertex)),
            */

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

        if (!cameraObj.contains("perspective") || !cameraObj.at("perspective").is_obj()) {
            PANIC("Scene loading error: camera doesn't have perspective values");
        }
        json::object const& perspObj = cameraObj.at("perspective").as_obj();
        if (!perspObj.contains("aspect") || !perspObj.contains("vfov") || !perspObj.contains("near")) {
            PANIC("Scene loading error: camera perspective is incomplete");
        }

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
        if (!driverObj.contains("node") || !driverObj.at("node").is_num()) {
            PANIC("Scene loading error: driver object doesn't have a target node");
        }
        uint32_t targetJsonId = driverObj.at("node").as_num();

        // Get the target channel
        if (!driverObj.contains("channel") || !driverObj.at("channel").is_str()) {
            PANIC("Scene loading error: driver object doesn't have a target channel");
        }
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
        if (!driverObj.contains("times") || !driverObj.at("times").is_arr()) {
            PANIC("Scene loading error: driver object doesn't have keyframe times");
        }
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
        if (!driverObj.contains("values") || !driverObj.at("values").is_arr()) {
            PANIC("Scene loading error: driver object doesn't have keyframe values");
        }
        else if (driverObj.at("values").as_arr().size() != channelWidth * driver.keyTimes.size()) {
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
        };

        // Load the normal & displacement maps
        if (materialObj.contains("normalMap") && materialObj.at("normalMap").is_obj()) {
            json::object const& normalMapObj = materialObj.at("normalMap").as_obj();
            if (!normalMapObj.contains("src") || !normalMapObj.at("src").is_str()) {
                PANIC("Scene loading error: normal map does not have a source");
            }
            // TODO: Verify that the texture is actually a 2D RGB texture

            std::filesystem::path filePath = directory / normalMapObj.at("src").as_str();
            material.normalMap = loadImage(renderInstance, filePath.string(), VK_FORMAT_R8G8B8A8_UNORM);
        }
        if (materialObj.contains("displacementMap") && materialObj.at("displacementMap").is_obj()) {
            json::object const& dispMapObj = materialObj.at("displacementMap").as_obj();
            if (!dispMapObj.contains("src") || !dispMapObj.at("src").is_str()) {
                PANIC("Scene loading error: displacement map does not have a source");
            }
            // TODO: Verify that the texture is actually a 2D RGB texture

            std::filesystem::path filePath = directory / dispMapObj.at("src").as_str();
            material.displacementMap = loadImage(renderInstance, filePath.string(), VK_FORMAT_R8G8B8A8_UNORM);
        }

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

            if (lambertianObj.contains("albedo") && lambertianObj.at("albedo").is_arr()) {
                json::array const& albedoArr = lambertianObj.at("albedo").as_arr();
                if (albedoArr.size() != 3 || !albedoArr[0].is_num() || !albedoArr[1].is_num() || !albedoArr[2].is_num()) {
                    PANIC("Scene loading error: invalid albedo vector");
                }
                material.albedoMap = Vec3<float>(albedoArr[0].as_num(), albedoArr[1].as_num(), albedoArr[2].as_num());
            }
            else if (lambertianObj.contains("albedo") && lambertianObj.at("albedo").is_obj()) {
                // TODO: Make loading a texture a helper
                json::object const& albedoObj = lambertianObj.at("albedo").as_obj();
                if (!albedoObj.contains("src") || !albedoObj.at("src").is_str()) {
                    PANIC("Scene loading error: albedo map does not have a source");
                }

                std::filesystem::path filePath = directory / albedoObj.at("src").as_str();
                material.albedoMap = loadImage(renderInstance, filePath.string(), VK_FORMAT_R8G8B8A8_SRGB);
            }
        }
        else {
            PANIC("Scene loading error: material does not contain a valid type");
        }

        materials.push_back(std::move(material));
    }

    // Build a buffer containing a MaterialConstants for each material
    buildMaterialConstantsBuffer(renderInstance);

    // Iterate through all the materials and construct their Material representation
    environments.reserve(environmentIdMap.size());
    for (auto idPair : environmentIdMap) {
        uint32_t s72Id = idPair.first;
        json::object const& envObj = sceneArr[s72Id].as_obj();

        Environment env {
            .name = envObj.at("name").as_str(),
        };

        // Load the radiance map
        if (!envObj.contains("radiance") && !envObj.at("radiance").is_obj()) {
            PANIC("Scene loading error: environment does not contain a radiance map");
        }
        json::object const& radianceObj = envObj.at("radiance").as_obj();
        if (!radianceObj.contains("src") || !radianceObj.at("src").is_str()) {
            PANIC("Scene loading error: environment radiance does not have a source");
        }
        // TODO: Verify that the texture is actually a cube RGBE texture

        std::filesystem::path filePath = directory / radianceObj.at("src").as_str();
        env.radiance = loadCubemap(renderInstance, filePath.string());

        // Also go ahead and load the pre-integrated Lambertian cubemap
        filePath.replace_extension(".lambertian.png");
        env.lambertian = loadCubemap(renderInstance, filePath.string());

        environments.push_back(std::move(env));
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
