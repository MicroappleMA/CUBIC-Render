#pragma once

#include "main/rhi.h"
#include "vulkan/vulkan.h"
#include "glfw/glfw3.h"

#include <vector>
#include <unordered_map>

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

    void initPipeline();

    void createFramebuffer();

    void createCommandPoolAndBuffer();

    void createSyncObjects();

    void generateCommandBuffer(const uint32_t framebufferIndex);

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
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    std::vector<VkFramebuffer> framebuffers;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkSemaphore framebufferReadyForRender;
    VkSemaphore framebufferReadyForPresent;
    VkFence commandBufferFinish;
};

