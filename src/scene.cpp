#include <vulkan/vulkan.h>

#include <iostream>
#include <memory>
#include <stdexcept>

#include "buffer.hpp"
#include "instance.hpp"
#include "linear.hpp"
#include "scene.hpp"
#include "util.hpp"

void Scene::renderScene(SceneRenderInfo const& sceneRenderInfo) {
    for (uint32_t rootNode : sceneRoots) {
        if (rootNode >= nodes.size()) {
            PANIC(string_format("node %u out of range is listed as scene root!", rootNode));
        }
        renderNode(sceneRenderInfo, rootNode, linear::M4F_IDENTITY);
    }
}

void Scene::renderNode(SceneRenderInfo const& sceneRenderInfo, uint32_t nodeId, Mat4<float> const& parentToWorldTransform) {
    // Sanity check that we are rendering a valid node - we should already be checking in the relevant areas
    if (nodeId >= nodes.size()) {
        PANIC(string_format("tried to render node %u out of range!", nodeId));
    }
    Node& node = nodes[nodeId];

    // Accumulate node's transform
    Mat4<float> worldTransform = linear::mmul(parentToWorldTransform, node.transform);

    // Render the attached mesh if we have one
    if (node.meshIndex.has_value()) {
        if (node.meshIndex.value() >= meshes.size()) {
            PANIC(string_format("node %s tried to render mesh %u out of range!", node.name.c_str(), node.meshIndex.value()));
        }
        renderMesh(sceneRenderInfo, node.meshIndex.value(), worldTransform);
    }

    // Render any child nodes
    for (uint32_t childId : node.childIndices) {
        if (childId >= nodes.size()) {
            PANIC(string_format("node %s tried to render node %u out of range!", node.name.c_str(), childId));
        }
        renderNode(sceneRenderInfo, childId, worldTransform);
    }
}

void Scene::renderMesh(SceneRenderInfo const& sceneRenderInfo, uint32_t meshId, Mat4<float> const& worldTransform) {
    // Sanity check that we are rendering a valid mesh - we should already be checking in renderNode so that we have a more specific error message
    if (meshId >= meshes.size()) {
        PANIC(string_format("tried to render mesh %u out of range!", meshId));
    }

    Mesh& mesh = meshes[meshId];

    if (mesh.vertexBufferIndex >= buffers.size()) {
        PANIC(string_format("mesh %s includes vertex buffer %u out of range!", mesh.name.c_str(), mesh.vertexBufferIndex));
    }
    VkBuffer meshVertBuffer = buffers[mesh.vertexBufferIndex].buffer;

    VkDeviceSize offsets[] = {mesh.vertexBufferOffset};
    vkCmdBindVertexBuffers(sceneRenderInfo.commandBuffer, 0, 1, &meshVertBuffer, offsets);
    vkCmdPushConstants(sceneRenderInfo.commandBuffer, sceneRenderInfo.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4<float>), &worldTransform);

    vkCmdDraw(sceneRenderInfo.commandBuffer, mesh.vertexCount, 1, 0, 0);
}

void Scene::updateCameraTransform() {
    // Calculate the projection matrix
    if (selectedCamera >= cameras.size()) {
        PANIC(string_format("selected camera %u is out of range!", selectedCamera));
    }
    Camera& camera = cameras[selectedCamera];

    if (camera.farZ.has_value()) {
        viewProj.proj = linear::perspective(camera.vFov, camera.aspectRatio, camera.nearZ, camera.farZ.value());
    }
    else {
        viewProj.proj = linear::infinitePerspective(camera.vFov, camera.aspectRatio, camera.nearZ);
    }

    // Find the worldToLocal transform for the selected camera to serve as the view matrix
    bool foundCamera = false;
    for (uint32_t rootNode : sceneRoots) {
        if (rootNode >= nodes.size()) {
            PANIC(string_format("node %u out of range is listed as scene root!", rootNode));
        }

        std::optional<Mat4<float>> worldToLocal = findCameraWTLTransform(rootNode, selectedCamera);
        if (worldToLocal.has_value()) {
            viewProj.view = worldToLocal.value();
            foundCamera = true;
            break;
        }
    }

    if (!foundCamera) {
        PANIC(string_format("selected camera %u is not in scene!", selectedCamera));
    }
}

std::optional<Mat4<float>> Scene::findCameraWTLTransform(uint32_t nodeId, uint32_t cameraId) {
    // Sanity check that we are rendering a valid node - we should already be checking in the relevant areas
    if (nodeId >= nodes.size()) {
        PANIC(string_format("tried to render node %u out of range!", nodeId));
    }
    Node& node = nodes[nodeId];

    // If node has camera, send parentToLocal transform
    if (node.cameraIndex.has_value() && node.cameraIndex.value() == cameraId) {
        return node.invTransform;
    };

    for (uint32_t childId : node.childIndices) {
        if (childId >= nodes.size()) {
            PANIC(string_format("node %s has node %u out of range as child!", node.name.c_str(), childId));
        }

        std::optional<Mat4<float>> localToChild = findCameraWTLTransform(childId, cameraId);
        if (localToChild.has_value()) {
            // Found camera in child, so accumulate transform upwards
            return linear::mmul(node.invTransform, localToChild.value());
        }
    }

    return std::nullopt;
}
