#include <vulkan/vulkan.h>

#include <filesystem>
#include <map>
#include <cstring>

#include "instance.hpp"
#include "json.hpp"
#include "options.hpp"
#include "scene.hpp"
#include "util.hpp"

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
            .transform = linear::localToParent(t, q, s),
            .invTransform = linear::parentToLocal(t, q, s),
        };
        // Add mesh and camera indices
        if (nodeObj.contains("mesh")) {
            uint32_t jsonId = nodeObj.at("mesh").as_num();
            node.meshIndex = meshIdMap.at(jsonId);
        }
        if (nodeObj.contains("camera")) {
            uint32_t jsonId = nodeObj.at("camera").as_num();
            node.cameraIndex = cameraIdMap.at(jsonId);
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

    // Iterate through all the meshes, and accumulate all the vertex data into one big buffer
    // TODO: properly handle topology, index buffers, and attribute sets aside from the assumed given one
    // First, loop through the mesh objects just so we can reserver a big enough buffer
    uint32_t totalBufferSize = 0;
    for (auto idPair : meshIdMap) {
        uint32_t s72Id = idPair.first;
        json::object const& meshObj = sceneArr[s72Id].as_obj();

        if (!meshObj.contains("count") || !meshObj.at("count").is_num()) {
            PANIC("Scene loading error: mesh is missing vertex count");
        }

        // TODO: account for different attribute sets
        // Each vertex has a size of 28, so size is vertexCount * 28
        totalBufferSize += static_cast<uint32_t>(meshObj.at("count").as_num()) * 28;
    }

    // Now, create the mesh objects and copy into the CPU buffer
    uint32_t curBufferSize = 0;
    std::unique_ptr<char[]> buffer(new char[totalBufferSize]);
    for (auto idPair : meshIdMap) {
        uint32_t s72Id = idPair.first;
        json::object const& meshObj = sceneArr[s72Id].as_obj();

        if (!meshObj.contains("attributes") || !meshObj.at("attributes").is_obj()) {
            PANIC("Scene loading error: mesh doesn't have attributes");
        }
        json::object const& attribObj = meshObj.at("attributes").as_obj();

        Mesh mesh {
            .name = meshObj.at("name").as_str(),
            .vertexCount = static_cast<uint32_t>(meshObj.at("count").as_num()),
            .vertexBufferIndex = 0,
            .vertexBufferOffset = curBufferSize,
        };

        meshes.push_back(mesh);

        // Load the file and copy into the total buffer
        // TODO: actually handle the attributes
        json::object const& posObj = attribObj.at("POSITION").as_obj();
        uint32_t fileOffset = posObj.at("offset").as_num();
        uint32_t fileSize = static_cast<uint32_t>(meshObj.at("count").as_num()) * 28;
        std::filesystem::path filePath = directory / posObj.at("src").as_str();
        std::vector<char> file = readFile(filePath.string());
        memcpy(&buffer[curBufferSize], file.data() + fileOffset, fileSize);

        curBufferSize += fileSize;
    }

    // Create the vertex buffer from our accumulated buffer
    vertexBufferFromBuffer(renderInstance, buffer.get(), totalBufferSize);

    // Iterate through all the cameras and construct their Camera representation
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

    // TODO: construct driver objects

    // Set the default camera
    if (options::getDefaultCamera().has_value()) {
        // Start with the selected camera as the one with the specified name
        std::string cameraName = options::getDefaultCamera().value();
        bool foundCamera = false;
        for (uint32_t i = 0; i < cameras.size(); i++) {
            if (cameras[i].name == cameraName) {
                selectedCamera = i;
                foundCamera = true;
                break;
            }
        }

        if (!foundCamera) {
            PANIC("failed to find camera specified by --camera");
        }
        useUserCamera = false;
    }
    else {
        // Default case: just use the user camera
        selectedCamera = 0;
        useUserCamera = true;
    }
    useDebugCamera = false;
}

// Helper to construct a vertex buffer from a CPU buffer
uint32_t Scene::vertexBufferFromBuffer(std::shared_ptr<RenderInstance>& renderInstance, const void* inBuffer, uint32_t size) {
    // The final vertex buffer we want to use
    CombinedBuffer& vertexBuffer = buffers.emplace_back(renderInstance, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // Staging buffer will location on CPU so that we can directly copy to it. We then use it to transfer fully over to the GPU
    CombinedBuffer stagingBuffer = CombinedBuffer(renderInstance, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data;
    vkMapMemory(renderInstance->device, stagingBuffer.bufferMemory, 0, size, 0, &data);
    memcpy(data, inBuffer, (size_t) size);
    vkUnmapMemory(renderInstance->device, stagingBuffer.bufferMemory);

    BufferCopy bufferCopyInfos[] = {
        {
            .srcBuffer = stagingBuffer.buffer,
            .srcOffset = 0,
            .dstBuffer = vertexBuffer.buffer,
            .dstOffset = 0,
            .size = size,
        },
    };
    copyBuffers(*renderInstance, bufferCopyInfos, 1);

    return buffers.size() - 1;
}
