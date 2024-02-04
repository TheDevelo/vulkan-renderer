#include <vulkan/vulkan.h>

#include <string>
#include <vector>

struct Mesh {
    std::string name;
    // Vertex Buffer data
    uint32_t vertexCount;
    uint32_t vertexBufferIndex;
    uint32_t vertexBufferOffset;
};

class Scene {
public:
    void renderScene(VkCommandBuffer commandBuffer);
//private:
    std::vector<Mesh> meshes;
    std::vector<VkBuffer> buffers;
};
