#include <vulkan/vulkan.h>

#include "scene.hpp"

void Scene::renderScene(VkCommandBuffer commandBuffer) {
    // Temporary scene renderer: draw each mesh once
    for (auto mesh = meshes.begin(); mesh != meshes.end(); mesh++) {
        VkBuffer meshVertBuffer = buffers[mesh->vertexBufferIndex];

        VkDeviceSize offsets[] = {mesh->vertexBufferOffset};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &meshVertBuffer, offsets);

        vkCmdDraw(commandBuffer, mesh->vertexCount, 1, 0, 0);
    }
}
