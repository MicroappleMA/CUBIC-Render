#pragma once

#include "VulkanBuffer.h"
#include "VulkanSharedBuffer.h"
#include "VulkanImage.h"
#include "main/rhi.h"
#include "vulkan/vulkan.h"
#include "glfw/glfw3.h"
#include "glm/glm.hpp"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <functional>

class RHIVK: public RHI {
public:
    void init(int width, int height, bool vsync) override;
    void setCallback(PFN_cursorPosCallback cursorPosCallback,
                     PFN_scrollCallback scrollCallback,
                     PFN_mouseButtonCallback mouseButtonCallback,
                     PFN_keyCallback keyCallback) override;
    void pollEvents() override;
    void* mapBuffer() override;
    void unmapBuffer() override;
    void draw(const char* title) override;
    void destroy() override;

private:
    static std::unordered_map<int, RHIKeyCode> actionCodeMap;
    static std::unordered_map<int, RHIKeyCode> keyCodeMap;
    static PFN_keyCallback keyCallback;
    static PFN_mouseButtonCallback mouseButtonCallback;
    static PFN_scrollCallback scrollCallback;
    static PFN_cursorPosCallback cursorPosCallback;
    static void glfwErrorCallback(int error, const char *description);
    static void glfwKeyCallback(GLFWwindow* window,int key, int scancode, int action, int mods);
    static void glfwMouseButtonCallback(GLFWwindow* window,int button, int action, int mods);
    static void glfwScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void glfwCursorPosCallback(GLFWwindow* window, double xpos, double ypos);

    void createInstance();
    void enableValidationLayer(VkInstanceCreateInfo &createInfo);

    void initSurface();

    void setPhysicalDevice();
    bool checkDeviceSuitability(const VkPhysicalDevice &device);

    void createLogicalDevice();

    void createSwapChain();

    VkShaderModule createShaderModule(const VkShaderStageFlagBits shaderStage, const char* shaderCode);

    void createRenderPass();

    void createPipelineLayout();

    void initPipeline();

    void createFramebuffer();

    void createCommandPoolAndBuffer();

    void createSyncObjects();

    void createVertexBuffer();

    void createIndexBuffer();

    void createSharedBuffer();

    void createImage();

    void createDescriptorSet();

    void generateCommandBuffer(const uint32_t framebufferIndex);

    void submitCommand(const std::function<void(void)> &command);

    void copyCudaImage();

    struct VertexInput{
        glm::vec3 position;
        glm::vec3 color;
    };

    const char* const VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation";
    const uint32_t NVIDIA_VENDOR_ID = 0x10de;
    const float HIGHEST_QUEUE_PRIORITY = 1.0f;

    const std::vector<const char*> CUSTOM_INSTANCE_EXTENSIONS = {
            VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    };

    const std::vector<const char*> CUSTOM_DEVICE_EXTENSIONS = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
            VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME
    };

    const std::vector<VertexInput> DEFAULT_VERTEX_BUFFER = {
            {{-1.0, -1.0,  0.0}, {0.00, 0.00, 0.01}},
            {{ 1.0, -1.0,  0.0}, {0.00, 0.01, 0.00}},
            {{ 1.0,  1.0,  0.0}, {0.01, 0.00, 0.00}},
            {{-1.0,  1.0,  0.0}, {0.00, 0.01, 0.00}},
    };

    const std::vector<uint32_t> DEFAULT_INDEX_BUFFER = {
            0,1,2,0,2,3
    };

    int width, height;
    bool vsync;
    GLFWwindow *window;
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    struct QueueFamily{
        bool hasGraphics;
        uint32_t graphics;
        // bool hasCompute;
        // uint32_t compute;
        // bool hasTransfer;
        // uint32_t transfer;
    }queueFamily;
    VkDevice device;
    struct Queue{
        VkQueue graphics;
        // VkQueue compute;
        // VkQueue transfer;
    }queue;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;
    VkSurfaceFormatKHR swapChainFormat;
    VkExtent2D swapChainExtent;
    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    std::vector<VkFramebuffer> framebuffers;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkSemaphore framebufferReadyForRender;
    VkSemaphore framebufferReadyForPresent;
    VkFence commandBufferFinish;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VulkanBuffer* vertexBuffer;
    VulkanBuffer* indexBuffer;
    VulkanSharedBuffer* sharedBuffer;
    VulkanImage* image;
};

