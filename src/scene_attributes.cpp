#include <vulkan/vulkan.h>

#include <filesystem>
#include <map>
#include <optional>
#include <cstring>

#include "linear.hpp"
#include "materials.hpp"
#include "scene.hpp"
#include "util.hpp"

// Helper class for loading from the attribute streams
// TODO: Support converting from one format to another instead of just panicing if we don't have the format we want.
class AttributeStream {
public:
    AttributeStream(std::vector<uint8_t> const& srcIn, uint32_t offsetIn, uint32_t strideIn, VkFormat formatIn) : source(srcIn), offset(offsetIn), stride(strideIn), format(formatIn) {}

    Vec2<float> nextVec2f() {
        if (format != VK_FORMAT_R32G32_SFLOAT) {
            PANIC("Scene loading error: attribute stream has unexpected format");
        }

        // Get the appropriate data pointer from the source and offset, then increment the offset by the stride
        const uint8_t* rawData = source.data() + offset;
        const Vec2<float>* data = reinterpret_cast<const Vec2<float>*>(rawData);
        offset += stride;

        return *data;
    }

    Vec3<float> nextVec3f() {
        if (format != VK_FORMAT_R32G32B32_SFLOAT) {
            PANIC("Scene loading error: attribute stream has unexpected format");
        }

        // Get the appropriate data pointer from the source and offset, then increment the offset by the stride
        const uint8_t* rawData = source.data() + offset;
        const Vec3<float>* data = reinterpret_cast<const Vec3<float>*>(rawData);
        offset += stride;

        return *data;
    }

    Vec4<float> nextVec4f() {
        if (format != VK_FORMAT_R32G32B32A32_SFLOAT) {
            PANIC("Scene loading error: attribute stream has unexpected format");
        }

        // Get the appropriate data pointer from the source and offset, then increment the offset by the stride
        const uint8_t* rawData = source.data() + offset;
        const Vec4<float>* data = reinterpret_cast<const Vec4<float>*>(rawData);
        offset += stride;

        return *data;
    }

    Vec4<uint8_t> nextVec4u8() {
        if (format != VK_FORMAT_R8G8B8A8_UNORM) {
            PANIC("Scene loading error: attribute stream has unexpected format");
        }

        // Get the appropriate data pointer from the source and offset, then increment the offset by the stride
        const uint8_t* rawData = source.data() + offset;
        const Vec4<uint8_t>* data = reinterpret_cast<const Vec4<uint8_t>*>(rawData);
        offset += stride;

        return *data;
    }

private:
    std::vector<uint8_t> const& source;
    uint32_t offset;
    uint32_t stride;
    VkFormat format;
};

// Helper to convert strings to their equivalent Vulkan formats
VkFormat formatFromString(std::string const& str) {
    if (str == "R32G32_SFLOAT") {
        return VK_FORMAT_R32G32_SFLOAT;
    }
    else if (str == "R32G32B32_SFLOAT") {
        return VK_FORMAT_R32G32B32_SFLOAT;
    }
    else if (str == "R32G32B32A32_SFLOAT") {
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    else if (str == "R8G8B8A8_UNORM") {
        return VK_FORMAT_R8G8B8A8_UNORM;
    }
    else {
        PANIC("Scene loading error: invalid format specified!");
    }
}

// Loads a stream of vertices from the mesh attributes into the unified Vertex format.
std::vector<Vertex> Scene::loadVerticesFromAttributes(json::object const& attributeJson, uint32_t vertexCount, std::filesystem::path directory) {
    // Setup the attribute streams
    std::map<std::string, std::vector<uint8_t>> fileMap;
    std::optional<AttributeStream> positionStream;
    std::optional<AttributeStream> normalStream;
    std::optional<AttributeStream> tangentStream;
    std::optional<AttributeStream> uvStream;
    std::optional<AttributeStream> colorStream;

    auto assignStream = [&](std::string streamName, std::optional<AttributeStream>& stream) {
        if (attributeJson.contains(streamName) && attributeJson.at(streamName).is_obj()) {
            json::object const& streamObj = attributeJson.at(streamName).as_obj();

            if (!streamObj.contains("src") || !streamObj.at("src").is_str()) {
                PANIC("Scene loading error: attribute stream doesn't contain a source");
            }
            std::string const& filename = streamObj.at("src").as_str();
            if (!fileMap.contains(filename)) {
                // Load the source file if we don't already have it loaded
                fileMap.insert_or_assign(filename, readFile((directory / filename).string()));
            }

            if (!streamObj.contains("offset") || !streamObj.at("offset").is_num()) {
                PANIC("Scene loading error: attribute stream doesn't contain an offset");
            }
            uint32_t offset = streamObj.at("offset").as_num();

            if (!streamObj.contains("stride") || !streamObj.at("stride").is_num()) {
                PANIC("Scene loading error: attribute stream doesn't contain a stride");
            }
            uint32_t stride = streamObj.at("stride").as_num();

            if (!streamObj.contains("format") || !streamObj.at("format").is_str()) {
                PANIC("Scene loading error: attribute stream doesn't contain a format");
            }
            VkFormat format = formatFromString(streamObj.at("format").as_str());

            stream.emplace(fileMap.at(filename), offset, stride, format);
        }
    };

    assignStream("POSITION", positionStream);
    assignStream("NORMAL", normalStream);
    assignStream("TANGENT", tangentStream);
    assignStream("TEXCOORD", uvStream);
    assignStream("COLOR", colorStream);

    // Construct the vertex stream from the attribute streams
    std::vector<Vertex> vertices;
    vertices.resize(vertexCount);
    for (uint32_t i = 0; i < vertexCount; i++) {
        if (positionStream.has_value()) {
            vertices[i].pos = positionStream.value().nextVec3f();
        }
        else {
            PANIC("Scene loading error: mesh doesn't contain a POSITION stream");
        }

        if (normalStream.has_value()) {
            vertices[i].normal = normalStream.value().nextVec3f();
        }
        else {
            // TODO: Maybe calculate face normals from vertex positions? Would have to do in a second pass, and would require the index buffer if we have one.
            PANIC("Scene loading error: mesh doesn't contain a NORMAL stream");
        }

        if (tangentStream.has_value()) {
            vertices[i].tangent = tangentStream.value().nextVec4f();
            vertices[i].tangent.w *= -1;
        }
        else {
            // TODO: Maybe pick a tangent vector that is actually tangent to the normal? Shouldn't matter to set as 0, since no tangent means no normal mapping
            vertices[i].tangent = Vec4<float>(0.0f);
        }

        if (uvStream.has_value()) {
            vertices[i].uv = uvStream.value().nextVec2f();
            vertices[i].uv.v = 1 - vertices[i].uv.v;
        }
        else {
            vertices[i].uv = Vec2<float>(0.0f);
        }

        if (colorStream.has_value()) {
            vertices[i].color = colorStream.value().nextVec4u8();
        }
        else {
            vertices[i].color = Vec4<uint8_t>(255);
        }
    }

    return vertices;
}

// Creates a combined vertex buffer from the vertices of each and every mesh
void Scene::buildCombinedVertexBuffer(std::shared_ptr<RenderInstance>& renderInstance, std::vector<std::vector<Vertex>> const& meshVertices) {
    // Get the total size of the combined vertex buffer
    uint32_t totalSize = 0;
    for (auto vertices : meshVertices) {
        totalSize += vertices.size();
    }
    totalSize *= sizeof(Vertex);

    // The final vertex buffer we want to use
    CombinedBuffer& vertexBuffer = buffers.emplace_back(renderInstance, totalSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    uint32_t bufferIndex = buffers.size() - 1;
    // Staging buffer will location on CPU so that we can directly copy to it. We then use it to transfer fully over to the GPU
    CombinedBuffer stagingBuffer = CombinedBuffer(renderInstance, totalSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Load the meshVertices array into the staging buffer, as well as setting the appropriate mesh properties to match the combined vertex buffer
    void* data;
    vkMapMemory(renderInstance->device, stagingBuffer.bufferMemory, 0, totalSize, 0, &data);
    uint32_t currentOffset = 0;
    for (size_t i = 0; i < meshes.size(); i++) {
        Mesh& mesh = meshes[i];
        mesh.vertexBufferIndex = bufferIndex;
        mesh.vertexBufferOffset = currentOffset;
        size_t meshSize = meshVertices[i].size() * sizeof(Vertex);

        uint8_t* dst = reinterpret_cast<uint8_t*>(data) + currentOffset;
        memcpy(dst, meshVertices[i].data(), meshSize);

        currentOffset += meshSize;
    }
    vkUnmapMemory(renderInstance->device, stagingBuffer.bufferMemory);

    // Copy from staging buffer to vertex buffer
    VkCommandBuffer commandBuffer = beginSingleUseCBuffer(*renderInstance);
    BufferCopy bufferCopyInfos[] = {
        {
            .srcBuffer = stagingBuffer.buffer,
            .srcOffset = 0,
            .dstBuffer = vertexBuffer.buffer,
            .dstOffset = 0,
            .size = totalSize,
        },
    };
    copyBuffers(commandBuffer, bufferCopyInfos, 1);
    endSingleUseCBuffer(*renderInstance, commandBuffer);
}
