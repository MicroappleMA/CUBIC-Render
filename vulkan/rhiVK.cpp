#include "rhiVK.h"
#include "main/rhi.h"
#include "vulkan/vulkan.h"
#include "glfw/glfw3.h"
#include "VulkanMacro.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <cstring>

#ifndef NDEBUG
#define VK_VALIDATION_LAYER
#endif

void RHIVK::init() {
    createInstance();
    setPhysicalDevice();
    createLogicalDevice();
}

void RHIVK::initSurface(int width, int height, bool vsync) {
    this->width = width;
    this->height = height;
    this->vsync = vsync;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(width, height, "", nullptr, nullptr);

}

void RHIVK::initPipeline() {

}

void RHIVK::setCallback(PFN_cursorPosCallback cursorPosCallback, PFN_scrollCallback scrollCallback,
                        PFN_mouseButtonCallback mouseButtonCallback, PFN_keyCallback keyCallback) {

}

void *RHIVK::mapBuffer() {
    return nullptr;
}

void RHIVK::unmapBuffer() {

}

void RHIVK::draw(const char *title) {

}

void RHIVK::destroy() {
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
}

void RHIVK::createInstance() {
    instance = VK_NULL_HANDLE;
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "CUBIC Render";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "CUBIC Render";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions;
    createInfo.enabledLayerCount = 0;

#ifdef VK_VALIDATION_LAYER
    enableValidationLayer(createInfo);
#endif

    VK_CHECK_RESULT(vkCreateInstance(&createInfo, nullptr, &instance));
    assert(instance!=VK_NULL_HANDLE);
}

void RHIVK::enableValidationLayer(VkInstanceCreateInfo &createInfo) {
    // Check if validation layer available
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    bool foundValidationLayer = false;
    for (const auto& layer : availableLayers)
    {
        if(strcmp(layer.layerName, VALIDATION_LAYER_NAME) == 0)
        {
            foundValidationLayer = true;
            break;
        }
    }

    if(!foundValidationLayer)
    {
        std::cout<<"[Warning] Vulkan validation layer not support.\n";
        return;
    }

    createInfo.enabledLayerCount = 1;
    createInfo.ppEnabledLayerNames = &VALIDATION_LAYER_NAME;
}

void RHIVK::setPhysicalDevice() {
    physicalDevice = VK_NULL_HANDLE;
    uint32_t physicalDeviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
    assert(physicalDeviceCount > 0);
    std::vector<VkPhysicalDevice> availablePhysicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, availablePhysicalDevices.data());

    for(const auto& thisPhysicalDevice:availablePhysicalDevices)
    {
        if(checkDeviceSuitability(thisPhysicalDevice))
        {
            break;
        }
    }
    assert(physicalDevice!=VK_NULL_HANDLE);
}

bool RHIVK::checkDeviceSuitability(const VkPhysicalDevice &device) {
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceProperties(device, &properties);
    vkGetPhysicalDeviceFeatures(device, &features);

    // Check Device Properties and Features Here
    // Only Support NVIDIA CUDA GPU
    if(properties.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && properties.vendorID == NVIDIA_VENDOR_ID)
    {
        // Check Queue Family Here
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device,&queueFamilyCount,nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device,&queueFamilyCount,queueFamilyProperties.data());

        uint32_t index = 0;
        QueueFamily thisQueueFamily{};
        for(const auto thisQueueFamilyProperty:queueFamilyProperties)
        {
            if(thisQueueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                thisQueueFamily.hasGraphics = true;
                thisQueueFamily.graphics = index;
            }
            // if(thisQueueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
            // {
            //     thisQueueFamily.hasCompute = true;
            //     thisQueueFamily.compute = index;
            // }
            // if(thisQueueFamilyProperty.queueFlags & VK_QUEUE_TRANSFER_BIT)
            // {
            //     thisQueueFamily.hasTransfer = true;
            //     thisQueueFamily.transfer = index;
            // }
            index++;
        }

        // Only Support GPU With Graphics Queue Family
        if (thisQueueFamily.hasGraphics)
        {
            std::cout << "[Log] Using " << properties.deviceName << "\n";
            physicalDevice = device;
            queueFamily = thisQueueFamily;
            return true;
        }
    }
    return false;
}

void RHIVK::createLogicalDevice() {
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamily.graphics;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &HIGHEST_QUEUE_PRIORITY;

    VkPhysicalDeviceFeatures deviceFeatures{};
    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
    deviceCreateInfo.enabledExtensionCount = 0;

    VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

    vkGetDeviceQueue(device, queueFamily.graphics, 0, &queue.graphics);
}

