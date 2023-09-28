#pragma once

#include "main/rhi.h"
#include "vulkan/vulkan.h"
#include "glfw/glfw3.h"

#include <vector>

class RHIVK: public RHI {
public:
    void init(int width, int height, bool vsync) override;
    void setCallback(PFN_cursorPosCallback cursorPosCallback,
                     PFN_scrollCallback scrollCallback,
                     PFN_mouseButtonCallback mouseButtonCallback,
                     PFN_keyCallback keyCallback) override;
    void* mapBuffer() override;
    void unmapBuffer() override;
    void draw(const char* title) override;
    void destroy() override;

private:
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


    const char* VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation";
    const char* SWAPCHAIN_EXTENSION_NAME = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    const uint32_t NVIDIA_VENDOR_ID = 0x10de;
    const float HIGHEST_QUEUE_PRIORITY = 1.0f;

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
};

