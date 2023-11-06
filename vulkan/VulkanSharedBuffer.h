#pragma once

#include "vulkan/vulkan.h"
#include "VulkanMacro.h"

#include <cuda_runtime_api.h>

class VulkanSharedBuffer {
public:
    VulkanSharedBuffer(VkPhysicalDevice _physicalDevice, VkDevice _device, VkDeviceSize _size, VkBufferUsageFlags _usage);
    ~VulkanSharedBuffer();

    inline VkBuffer getBuffer() const {return buffer;};
    inline VkDeviceMemory getMemory() const {return memory;};
    inline VkDeviceSize getSize() const {return size;};

private:

    const VkExternalMemoryHandleTypeFlagBits vulkanExternalMemoryHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    const cudaExternalMemoryHandleType cudaExternalMemoryHandleType = cudaExternalMemoryHandleTypeOpaqueWin32;
    const VkMemoryPropertyFlags memoryPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    void createWindowsSecurityAttributes();
    void destroyWindowsSecurityAttributes();
    void createVulkanBuffer();
    void destroyVulkanBuffer();
    void mapCudaBuffer();
    void unmapCudaBuffer();

    uint32_t getMemoryTypeIndex(uint32_t memoryTypeBits);

    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
    VkBufferUsageFlags usage;

    cudaExternalMemory_t cudaMemory;
    void* cudaBuffer;

    SECURITY_ATTRIBUTES securityAttributes;
    PSECURITY_DESCRIPTOR psecurityDescriptor;
};