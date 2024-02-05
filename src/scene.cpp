#include <vulkan/vulkan.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <cstring>

#include "buffer.hpp"
#include "instance.hpp"
#include "linear.hpp"
#include "scene.hpp"
#include "util.hpp"

void Scene::renderScene(SceneRenderInfo const& sceneRenderInfo) {
    // Temporary scene renderer: act as the first node is the only root
    renderNode(sceneRenderInfo, 0, linear::M4F_IDENTITY);
}

void Scene::renderNode(SceneRenderInfo const& sceneRenderInfo, uint32_t nodeId, Mat4<float> const& parentToWorldTransform) {
    // Sanity check that we are rendering a valid node - we should already be checking in the relevant areas
    if (nodeId >= nodes.size()) {
        throw std::runtime_error(string_format("tried to render node %u out of range!", nodeId));
    }
    Node& node = nodes[nodeId];

    // Accumulate node's transform
    Mat4<float> worldTransform = linear::mmul(parentToWorldTransform, node.transform);

    // Render the attached mesh if we have one
    if (node.meshIndex.has_value()) {
        if (node.meshIndex.value() >= meshes.size()) {
            throw std::runtime_error(string_format("node %s tried to render mesh %u out of range!", node.name.c_str(), node.meshIndex.value()));
        }
        renderMesh(sceneRenderInfo, node.meshIndex.value(), worldTransform);
    }

    // Render any child nodes
    for (uint32_t childId : node.childIndices) {
        if (childId >= nodes.size()) {
            throw std::runtime_error(string_format("node %s tried to render node %u out of range!", node.name.c_str(), childId));
        }
        renderNode(sceneRenderInfo, childId, worldTransform);
    }
}

void Scene::renderMesh(SceneRenderInfo const& sceneRenderInfo, uint32_t meshId, Mat4<float> const& worldTransform) {
    // Sanity check that we are rendering a valid mesh - we should already be checking in renderNode so that we have a more specific error message
    if (meshId >= meshes.size()) {
        throw std::runtime_error(string_format("tried to render mesh %u out of range!", meshId));
    }

    Mesh& mesh = meshes[meshId];

    if (mesh.vertexBufferIndex >= buffers.size()) {
        throw std::runtime_error(string_format("mesh %s includes vertex buffer %u out of range!", mesh.name.c_str(), mesh.vertexBufferIndex));
    }
    VkBuffer meshVertBuffer = buffers[mesh.vertexBufferIndex].buffer;

    VkDeviceSize offsets[] = {mesh.vertexBufferOffset};
    vkCmdBindVertexBuffers(sceneRenderInfo.commandBuffer, 0, 1, &meshVertBuffer, offsets);
    vkCmdPushConstants(sceneRenderInfo.commandBuffer, sceneRenderInfo.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4<float>), &worldTransform);

    vkCmdDraw(sceneRenderInfo.commandBuffer, mesh.vertexCount, 1, 0, 0);
}

uint32_t Scene::vertexBufferFromBuffer(std::shared_ptr<RenderInstance>& renderInstance, const void* inBuffer, uint32_t size) {
    // The final vertex buffer we want to use
    CombinedBuffer& vertexBuffer = buffers.emplace_back(renderInstance, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // Staging buffer will contain both our data for the vertex and index buffer. We'll then copy both simultaneously.
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
