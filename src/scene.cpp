#include <vulkan/vulkan.h>

#include <algorithm>
#include <memory>
#include <set>
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

void Scene::updateCameraTransform(RenderInstance const& renderInstance) {
    if (useUserCamera) {
        // User camera
        // Calculate the projection matrix
        viewProj.proj = linear::infinitePerspective(DEG2RADF(60.0f), renderInstance.renderImageExtent.width / (float) renderInstance.renderImageExtent.height, 0.1f);

        // Calculate the view matrix
        float sinPhi = std::sin(userCamera.phi);
        float cosPhi = std::cos(userCamera.phi);
        float sinTheta = std::sin(userCamera.theta);
        float cosTheta = std::cos(userCamera.theta);
        Vec3<float> viewDirection(cosPhi * cosTheta, sinPhi * cosTheta, sinTheta);
        viewProj.view = linear::lookAt(userCamera.position, viewDirection + userCamera.position, Vec3<float>(0.0f, 0.0f, 1.0f));
    }
    else {
        // Scene camera
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
                // After getting worldToLocal, we need to flip the Y and Z coordinates to get our camera propertly oriented
                viewProj.view = worldToLocal.value();
                foundCamera = true;
                break;
            }
        }

        if (!foundCamera) {
            PANIC(string_format("selected camera %u is not in scene!", selectedCamera));
        }
    }

    // Copy our updated viewProj to the cullingViewProj if we don't have debug camera on
    if (!useDebugCamera) {
        cullingViewProj = viewProj;
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
            return linear::mmul(localToChild.value(), node.invTransform);
        }
    }

    return std::nullopt;
}

void Scene::moveUserCamera(UserCameraMoveEvent moveAmount, float dt) {
    constexpr float speed = 1.0f;
    // Calculate the view direction to determine where to move the camera
    float sin_phi = std::sin(userCamera.phi);
    float cos_phi = std::cos(userCamera.phi);
    float sin_theta = std::sin(userCamera.theta);
    float cos_theta = std::cos(userCamera.theta);
    Vec3<float> viewDirection(cos_phi * cos_theta, sin_phi * cos_theta, sin_theta);

    // Calculate the side direction where thetaSide = 0 & phiSide + pi/2. Use trig to get the below formula
    Vec3<float> sideDirection(-sin_phi, cos_phi, 0.0f);

    Vec3<float> delta = static_cast<float>(moveAmount.forwardAmount) * viewDirection + static_cast<float>(moveAmount.sideAmount) * sideDirection;
    userCamera.position = userCamera.position + speed * dt * delta;
}

void Scene::rotateUserCamera(UserCameraRotateEvent rotateAmount) {
    constexpr float EPSILON = 0.00001; // Epsilon so that we don't have our camera go perfectly vertical (causes issues)
    userCamera.phi = std::fmod(userCamera.phi + rotateAmount.xyRadians, 2.0f * M_PI);
    userCamera.theta = std::clamp(userCamera.theta + rotateAmount.zRadians, -static_cast<float>(M_PI) / 2.0f + EPSILON, static_cast<float>(M_PI) / 2.0f - EPSILON);
}

void Scene::updateAnimation(float time) {
    constexpr float EPSILON = 0.00001;

    std::set<uint32_t> updatedNodes;
    for (Driver& driver : drivers) {
        // Get the keyframe values required
        uint32_t keyIndex;
        bool doBinarySearch = false;
        if (driver.lastKeyIndex == 0) {
            // Special case: we are on first index, so just need to check if we are below the keyframe time
            if (time < driver.keyTimes[0]) {
                keyIndex = 0;
            }
            // If not, check the next pair in case time increased to the next keyframe
            else if (time >= driver.keyTimes[0] && time < driver.keyTimes[1]) {
                keyIndex = 1;
            }
            else {
                doBinarySearch = true;
            }
        }
        else if (driver.lastKeyIndex == driver.keyTimes.size()) {
            // Special case: we are on last index, so just need to check if we are above the keyframe time
            if (time >= driver.keyTimes[driver.lastKeyIndex - 1]) {
                keyIndex = driver.lastKeyIndex;
            }
            else {
                doBinarySearch = true;
            }
        }
        else {
            // Check if time is directly between lastKeyIndex - 1 and lastKeyIndex
            if (time >= driver.keyTimes[driver.lastKeyIndex - 1] && time < driver.keyTimes[driver.lastKeyIndex]) {
                keyIndex = driver.lastKeyIndex;
            }
            else if (driver.lastKeyIndex == driver.keyTimes.size() - 1) {
                // If lastKeyIndex is the last key time, then just check if we are past the end
                if (time >= driver.keyTimes[driver.lastKeyIndex]) {
                    keyIndex = driver.lastKeyIndex + 1;
                }
                else {
                    doBinarySearch = true;
                }
            }
            else {
                // If not, check the next pair in case time increased to the next keyframe
                if (time >= driver.keyTimes[driver.lastKeyIndex] && time < driver.keyTimes[driver.lastKeyIndex + 1]) {
                    keyIndex = driver.lastKeyIndex + 1;
                }
                else {
                    doBinarySearch = true;
                }
            }
        }

        // If the appropriate index is not the same or directly afterwards, then do binary search to quickly look through the rest
        // This should only come up if we skip or the animation resets
        if (doBinarySearch) {
            auto it = std::upper_bound(driver.keyTimes.begin(), driver.keyTimes.end(), time);
            keyIndex = std::distance(driver.keyTimes.begin(), it);
        }
        driver.lastKeyIndex = keyIndex;

        // Interpolate the value
        Vec4<float> interpolatedValue;
        if (keyIndex == 0) {
            // Special case - time is before the start, so just set interpolated value to be the first value
            interpolatedValue = driver.keyValues[0];
        }
        else if (keyIndex == driver.keyTimes.size()) {
            // Special case - time is past the end, so just set interpolated value to be the final value
            interpolatedValue = driver.keyValues[keyIndex - 1];
        }
        else {
            // We are in the middle of two keyframes, so interpolate between.
            if (driver.interpolation == DRIVER_STEP) {
                // Step, so just set the value to keyValues[keyIndex - 1]
                interpolatedValue = driver.keyValues[keyIndex - 1];
            }
            else if (driver.interpolation == DRIVER_LINEAR) {
                Vec4<float> first = driver.keyValues[keyIndex - 1];
                Vec4<float> last = driver.keyValues[keyIndex];
                float t = (time - driver.keyTimes[keyIndex - 1]) / (driver.keyTimes[keyIndex] - driver.keyTimes[keyIndex - 1]);
                interpolatedValue = first * (1 - t) + last * t;
            }
            else if (driver.interpolation == DRIVER_SLERP) {
                Vec4<float> first = driver.keyValues[keyIndex - 1];
                Vec4<float> last = driver.keyValues[keyIndex];
                float t = (time - driver.keyTimes[keyIndex - 1]) / (driver.keyTimes[keyIndex] - driver.keyTimes[keyIndex - 1]);

                // Calculate the angle between the first and last values for use in slerp
                float cosAngle = linear::dot(first, last) / (linear::length(first) * linear::length(last));
                if (cosAngle < 0) {
                    // If cosAngle is negative, then we need to flip cosAngle and first
                    cosAngle = -cosAngle;
                    first = -1.0f * first;
                }

                if (cosAngle > 1.0f - EPSILON) {
                    // If angle is sufficiently close to 0, then just linearly interpolate
                    interpolatedValue = first * (1 - t) + last * t;
                }
                else {
                    float angle = std::acos(cosAngle);
                    float invSinAngle = 1.0f / std::sin(angle);
                    interpolatedValue = std::sin((1.0f - t) * angle) * invSinAngle * first + std::sin(t * angle) * invSinAngle * last;
                }
            }
        }

        // Set the appropriate channel
        if (driver.targetNode >= nodes.size()) {
            PANIC(string_format("driver %s references node %u out of range!", driver.name.c_str(), driver.targetNode));
        }
        Node& node = nodes[driver.targetNode];
        if (driver.channel == DRIVER_TRANSLATION) {
            node.translation = interpolatedValue.xyz;
        }
        else if (driver.channel == DRIVER_ROTATION) {
            node.rotation = interpolatedValue;
        }
        else if (driver.channel == DRIVER_SCALE) {
            node.scale = interpolatedValue.xyz;
        }

        updatedNodes.insert(driver.targetNode);
    }

    // Update the transforms of our updated nodes
    for (uint32_t nodeId : updatedNodes) {
        nodes[nodeId].calculateTransforms();
    }
}
