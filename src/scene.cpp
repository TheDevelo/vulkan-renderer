#include <vulkan/vulkan.h>

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>

#include "buffer.hpp"
#include "instance.hpp"
#include "linear.hpp"
#include "options.hpp"
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

    // Cull the node
    if (cullingMode == CullingMode::BVH && node.bbox.has_value() && !bboxInViewFrustum(parentToWorldTransform, node.bbox.value())) {
        return;
    }

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

    // Cull the mesh
    if (cullingMode != CullingMode::OFF && !bboxInViewFrustum(worldTransform, mesh.bbox)) {
        return;
    }

    // Check we have a valid material
    if (mesh.materialIndex >= materials.size()) {
        PANIC(string_format("tried to render mesh %u with material %u out of range!", meshId, mesh.materialIndex));
    }
    Material& material = materials[mesh.materialIndex];

    // Bind the appropriate pipeline and descriptors
    VkPipelineLayout layout;
    if (material.type == MaterialType::SIMPLE) {
        layout = sceneRenderInfo.pipelines.simplePipelineLayout;
        vkCmdBindPipeline(sceneRenderInfo.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneRenderInfo.pipelines.simplePipeline);
    }
    else if (material.type == MaterialType::ENVIRONMENT) {
        layout = sceneRenderInfo.pipelines.envMirrorPipelineLayout;
        vkCmdBindPipeline(sceneRenderInfo.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneRenderInfo.pipelines.environmentPipeline);
    }
    else if (material.type == MaterialType::MIRROR) {
        layout = sceneRenderInfo.pipelines.envMirrorPipelineLayout;
        vkCmdBindPipeline(sceneRenderInfo.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneRenderInfo.pipelines.mirrorPipeline);
    }
    else if (material.type == MaterialType::LAMBERTIAN) {
        layout = sceneRenderInfo.pipelines.lambertianPipelineLayout;
        vkCmdBindPipeline(sceneRenderInfo.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneRenderInfo.pipelines.lambertianPipeline);
    }
    else {
        PANIC("mesh contains invalid material!");
    }

    vkCmdBindDescriptorSets(sceneRenderInfo.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &cameraDescriptorSet, 1, &sceneRenderInfo.cameraDescriptorOffset);
    vkCmdBindDescriptorSets(sceneRenderInfo.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1, &material.descriptorSet, 0, nullptr);
    if (material.type != MaterialType::SIMPLE) {
        vkCmdBindDescriptorSets(sceneRenderInfo.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 2, 1, &environments[0].descriptorSet, 1, &sceneRenderInfo.environmentDescriptorOffset);
    }

    if (mesh.vertexBufferIndex >= buffers.size()) {
        PANIC(string_format("mesh %s includes vertex buffer %u out of range!", mesh.name.c_str(), mesh.vertexBufferIndex));
    }
    VkBuffer meshVertBuffer = buffers[mesh.vertexBufferIndex].buffer;

    VkDeviceSize offsets[] = {mesh.vertexBufferOffset};
    vkCmdBindVertexBuffers(sceneRenderInfo.commandBuffer, 0, 1, &meshVertBuffer, offsets);
    vkCmdPushConstants(sceneRenderInfo.commandBuffer, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4<float>), &worldTransform);

    vkCmdDraw(sceneRenderInfo.commandBuffer, mesh.vertexCount, 1, 0, 0);
}

void Scene::updateCameraTransform(RenderInstance const& renderInstance) {
    float aspectRatio;
    float vFov;
    float nearZ;
    std::optional<float> farZ;

    if (useUserCamera) {
        // User camera
        aspectRatio = renderInstance.renderImageExtent.width / (float) renderInstance.renderImageExtent.height;
        vFov = DEG2RADF(60.0f);
        nearZ = 0.1f;
        cameraInfo.position = Vec4<float>(userCamera.position, 1.0);

        // Calculate the projection matrix
        cameraInfo.proj = linear::infinitePerspective(vFov, aspectRatio, nearZ);

        // Calculate the view matrix
        float sinPhi = std::sin(userCamera.phi);
        float cosPhi = std::cos(userCamera.phi);
        float sinTheta = std::sin(userCamera.theta);
        float cosTheta = std::cos(userCamera.theta);
        Vec3<float> viewDirection(cosPhi * cosTheta, sinPhi * cosTheta, sinTheta);
        cameraInfo.view = linear::lookAt(userCamera.position, viewDirection + userCamera.position, Vec3<float>(0.0f, 0.0f, 1.0f));
    }
    else {
        // Scene camera
        if (selectedCamera >= cameras.size()) {
            PANIC(string_format("selected camera %u is out of range!", selectedCamera));
        }
        Camera& camera = cameras[selectedCamera];
        aspectRatio = camera.aspectRatio;
        vFov = camera.vFov;
        nearZ = camera.nearZ;
        farZ = camera.farZ;

        // Calculate the projection matrix
        if (camera.farZ.has_value()) {
            cameraInfo.proj = linear::perspective(camera.vFov, camera.aspectRatio, camera.nearZ, camera.farZ.value());
        }
        else {
            cameraInfo.proj = linear::infinitePerspective(camera.vFov, camera.aspectRatio, camera.nearZ);
        }

        // Calculate the view matrix and position from the camera's ancestors
        if (camera.ancestors.size() == 0) {
            PANIC(string_format("selected camera %u is not in the scene tree!", selectedCamera));
        }

        Mat4<float> viewMatrix = linear::M4F_IDENTITY;
        for (uint32_t nodeId : camera.ancestors) {
            if (nodeId >= nodes.size()) {
                PANIC(string_format("node %u out of range is an ancestor to a camera!", nodeId));
            }
            Node& node = nodes[nodeId];

            viewMatrix = linear::mmul(node.invTransform, viewMatrix);
        }
        cameraInfo.view = viewMatrix;
        cameraInfo.position = Vec4<float>(userCamera.position, 1.0);
        for (auto ancestor = camera.ancestors.rbegin(); ancestor != camera.ancestors.rend(); ancestor++) {
            Node& node = nodes[*ancestor];
            cameraInfo.position = linear::mmul(node.transform, cameraInfo.position);
        }
    }

    // Update our culling camera if we don't have debug camera on
    if (!useDebugCamera) {
        // Half near height = nearZ * tan(vFov/2), and Half near width = Half near width * aspectRatio
        float halfNearHeight = nearZ * std::tan(vFov / 2.0f);
        float halfNearWidth = halfNearHeight * aspectRatio;
        cullingCamera = CullingCamera {
            .viewMatrix = cameraInfo.view,
            .halfNearWidth = halfNearWidth,
            .halfNearHeight = halfNearHeight,
            .nearZ = nearZ,
            .farZ = farZ,
        };
    }
}

void Scene::updateEnvironmentTransforms() {
    for (Environment& environment : environments) {
        if (environment.ancestors.size() == 0) {
            PANIC("environment is not in the scene tree!");
        }

        // Calculate the transform matrix from the environment's ancestors
        Mat4<float> worldToEnvMatrix = linear::M4F_IDENTITY;
        for (uint32_t nodeId : environment.ancestors) {
            if (nodeId >= nodes.size()) {
                PANIC(string_format("node %u out of range is an ancestor to a camera!", nodeId));
            }
            Node& node = nodes[nodeId];

            worldToEnvMatrix = linear::mmul(worldToEnvMatrix, node.invTransform);
        }
        environment.worldToEnv = worldToEnvMatrix;
    }
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

// Helper for bounding box-plane tests
// Idea: We can check if a point x is in-front/behind a plane given by point p and normal n by checking if dot(x - p, n) > 0.
// If all 8 corners of a bounding box lies behind a plane of the frustum, then the bounding box must lie outside of the frustum.
// We can actually simplify checking all 8 corners into 4 checks by using one corner c and 3 extents of the bounding box x,y,z.
// A corner of the bounding box will be C = c + (0/1)x + (0/1)y + (0/1)z. Distributing the dot product, we get dot(C - p, n) = dot(c - p, n) + (0/1)dot(x, n) + (0/1)dot(y, n) + (0/1)dot(z, n)
// Since we want to check dot(C - p, n) < 0 for all C, we only need to check the max dot product over all corners. To get the max, we can take the max of each component, and then sum.
// This gives our final bounding box-plane check of dot(c - p, n) + max(dot(x, n), 0) + max(dot(y, n), 0) + max(dot(z, n), 0) < 0. If true, BBox is fully behind the plane
bool bboxBehindPlane(Vec3<float> const& planePos, Vec3<float> const& planeNormal, Vec3<float> const& bboxCorner,
                     Vec3<float> const& bboxExtentX, Vec3<float> const& bboxExtentY, Vec3<float> const& bboxExtentZ) {
    constexpr float EPSILON = 0.00001;
    float cornerDot = linear::dot(bboxCorner - planePos, planeNormal);
    float extentXDot = std::max(linear::dot(bboxExtentX, planeNormal), 0.0f);
    float extentYDot = std::max(linear::dot(bboxExtentY, planeNormal), 0.0f);
    float extentZDot = std::max(linear::dot(bboxExtentZ, planeNormal), 0.0f);

    // Check if the corner dot is below -EPSILON instead of 0 to not exclude border cases
    return cornerDot + extentXDot + extentYDot + extentZDot < -EPSILON;
}

bool Scene::bboxInViewFrustum(Mat4<float> const& worldTransform, AxisAlignedBoundingBox const& bbox) {
    // Transform the bounding box into view space by transforming one corner of the bounding box, and the XYZ extents to the opposite corner
    Mat4<float> localToView = linear::mmul(cullingCamera.viewMatrix, worldTransform);
    Vec3<float> bboxCorner = linear::mmul(localToView, Vec4<float>(bbox.minCorner, 1.0f)).xyz;
    Vec3<float> bboxExtentX = linear::mmul(localToView, Vec4<float>(bbox.maxCorner.x - bbox.minCorner.x, 0.0f, 0.0f, 0.0f)).xyz;
    Vec3<float> bboxExtentY = linear::mmul(localToView, Vec4<float>(0.0f, bbox.maxCorner.y - bbox.minCorner.y, 0.0f, 0.0f)).xyz;
    Vec3<float> bboxExtentZ = linear::mmul(localToView, Vec4<float>(0.0f, 0.0f, bbox.maxCorner.z - bbox.minCorner.z, 0.0f)).xyz;

    // Compute the normals/positions for our frustum. Note that our bboxBehindPlane check doesn't depend on the normal being unit, as we are only checking sign.
    Vec3<float> topNormal = Vec3<float>(0.0f, -cullingCamera.nearZ, -cullingCamera.halfNearHeight);
    Vec3<float> bottomNormal = Vec3<float>(0.0f, cullingCamera.nearZ, -cullingCamera.halfNearHeight);
    Vec3<float> leftNormal = Vec3<float>(cullingCamera.nearZ, 0.0f, -cullingCamera.halfNearWidth);
    Vec3<float> rightNormal = Vec3<float>(-cullingCamera.nearZ, 0.0f, -cullingCamera.halfNearWidth);
    Vec3<float> frontNormal = Vec3<float>(0.0f, 0.0f, -1.0f);
    Vec3<float> backNormal = Vec3<float>(0.0f, 0.0f, 1.0f);

    // Check the bounding box against each normal
    if (bboxBehindPlane(Vec3<float>(0.0f), topNormal,    bboxCorner, bboxExtentX, bboxExtentY, bboxExtentZ) ||
        bboxBehindPlane(Vec3<float>(0.0f), bottomNormal, bboxCorner, bboxExtentX, bboxExtentY, bboxExtentZ) ||
        bboxBehindPlane(Vec3<float>(0.0f), leftNormal,   bboxCorner, bboxExtentX, bboxExtentY, bboxExtentZ) ||
        bboxBehindPlane(Vec3<float>(0.0f), rightNormal,  bboxCorner, bboxExtentX, bboxExtentY, bboxExtentZ) ||
        bboxBehindPlane(Vec3<float>(0.0f, 0.0f, -cullingCamera.nearZ), frontNormal, bboxCorner, bboxExtentX, bboxExtentY, bboxExtentZ)) {
        return false;
    }
    // Only check back face if we have one
    if (cullingCamera.farZ.has_value() && bboxBehindPlane(Vec3<float>(0.0f, 0.0f, -cullingCamera.farZ.value()), backNormal, bboxCorner, bboxExtentX, bboxExtentY, bboxExtentZ)) {
        return false;
    }

    return true;
}
