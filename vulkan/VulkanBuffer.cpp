#include "VulkanBuffer.h"

VulkanBuffer::VulkanBuffer(VkPhysicalDevice _physicalDevice, VkDevice _device, VkDeviceSize _size, VkBufferUsageFlags _usage, VkMemoryPropertyFlags _property):
physicalDevice(_physicalDevice), device(_device), size(_size), usage(_usage), property(_property) {
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer));

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

    VkMemoryAllocateInfo memoryAllocateInfo{};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = getMemoryTypeIndex(memoryRequirements.memoryTypeBits);

    VK_CHECK_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &memory));

    VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, memory, 0));
}

VulkanBuffer::~VulkanBuffer() {
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
}

uint32_t VulkanBuffer::getMemoryTypeIndex(uint32_t memoryTypeBits) {
    VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemoryProperties);
    for (size_t i=0; i<physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        if (
                (memoryTypeBits & (1 << i)) &&
                ((property & physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags) == property)
                )
        {
            return i;
        }
    }
    return ~0;
}

void *VulkanBuffer::mapMemory() {
    if (property & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    {
        void* pointer;
        VK_CHECK_RESULT(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &pointer));
        return pointer;
    }
    return nullptr;
}

void VulkanBuffer::unmapMemory() {
    if (property & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    {
        vkUnmapMemory(device, memory);
    }
}
