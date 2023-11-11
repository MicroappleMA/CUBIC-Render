#pragma once

#include "vulkan/vulkan.h"
#include "VulkanMacro.h"

class VulkanBuffer {
public:
    VulkanBuffer(VkPhysicalDevice _physicalDevice, VkDevice _device, VkDeviceSize _size, VkBufferUsageFlags _usage, VkMemoryPropertyFlags _property);
    ~VulkanBuffer();

    inline VkBuffer getBuffer() const {return buffer;}
    inline VkDeviceMemory getMemory() const {return memory;}
    inline VkDeviceSize getSize() const {return size;}
    inline VkBufferUsageFlags getUsage() const {return usage;}
    inline VkMemoryPropertyFlags getProperty() const {return property;}

    void* mapMemory();
    void unmapMemory();

private:
    uint32_t getMemoryTypeIndex(uint32_t memoryTypeBits);

    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    VkMemoryPropertyFlags property;
    void* pointer = nullptr;
};