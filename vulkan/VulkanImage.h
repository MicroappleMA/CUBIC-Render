#pragma once

#include "vulkan/vulkan.h"
#include "VulkanMacro.h"

class VulkanImage {
public:
    VulkanImage(VkPhysicalDevice _physicalDevice, VkDevice _device, VkExtent2D _extent, VkFormat _format, VkImageUsageFlags _usage, VkMemoryPropertyFlags _property);
    ~VulkanImage();

    inline VkImage getImage() const {return image;}
    inline VkImageView getImageView() const {return imageView;}
    inline VkSampler getSampler() const {return sampler;}
    inline VkFormat getFormat() const {return format;}
    inline VkImageLayout getLayout() const {return layout;}
    inline VkExtent2D getExtent() const {return extent;}
    inline VkDeviceMemory getMemory() const {return memory;}
    inline VkDeviceSize getSize() const {return size;}
    inline VkBufferUsageFlags getUsage() const {return usage;}
    inline VkMemoryPropertyFlags getProperty() const {return property;}

    // Temporal Solution: Class Can't Enclose Image Layout Transition
    inline void setLayout(VkImageLayout newLayout) {layout = newLayout;}

    void* mapMemory();
    void unmapMemory();
private:
    void createImage();
    void createImageView();
    void createSampler();

    uint32_t getMemoryTypeIndex(uint32_t memoryTypeBits);

    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkExtent2D extent;
    VkImageLayout layout;
    VkFormat format;
    VkImageUsageFlags usage;
    VkMemoryPropertyFlags property;
    VkDeviceSize size;
    VkImage image;
    VkImageView imageView;
    VkSampler sampler;
    VkDeviceMemory memory;

    void* pointer = nullptr;
};

