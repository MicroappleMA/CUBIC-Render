#include "VulkanImage.h"

#include <algorithm>

VulkanImage::VulkanImage(VkPhysicalDevice _physicalDevice, VkDevice _device, VkExtent2D _extent, VkFormat _format, VkImageUsageFlags _usage, VkMemoryPropertyFlags _property):
physicalDevice(_physicalDevice), device(_device), extent(_extent), format(_format), usage(_usage), property(_property)
{
    createImage();
    createImageView();
    createSampler();
}


VulkanImage::~VulkanImage() {
    if (sampler!=VK_NULL_HANDLE)
        vkDestroySampler(device, sampler, nullptr);
    if (imageView!=VK_NULL_HANDLE)
        vkDestroyImageView(device, imageView, nullptr);
    if (image!=VK_NULL_HANDLE)
        vkDestroyImage(device, image, nullptr);
    if (device!=VK_NULL_HANDLE)
        vkFreeMemory(device, memory, nullptr);
}

void VulkanImage::createImage() {
    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.extent.width = extent.width;
    imageCreateInfo.extent.height = extent.height;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    layout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.initialLayout = layout;
    imageCreateInfo.format = format;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = usage;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.flags = 0;

    VK_CHECK_RESULT(vkCreateImage(device, &imageCreateInfo, nullptr, &image));

    VkMemoryRequirements memoryRequirements{};
    vkGetImageMemoryRequirements(device, image, &memoryRequirements);
    size = memoryRequirements.size;

    VkMemoryAllocateInfo memoryAllocateInfo{};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = getMemoryTypeIndex(memoryRequirements.memoryTypeBits);

    VK_CHECK_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &memory));

    VK_CHECK_RESULT(vkBindImageMemory(device, image, memory, 0));
}

void VulkanImage::createImageView() {
    VkImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format = format;
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.layerCount = 1;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

    VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCreateInfo, nullptr, &imageView));
}

void VulkanImage::createSampler() {
    VkSamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerCreateInfo.compareEnable = VK_FALSE;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.maxLod = 0.0f;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.maxAnisotropy = 1.0f;
    samplerCreateInfo.unnormalizedCoordinates = VK_TRUE;

    VK_CHECK_RESULT(vkCreateSampler(device, &samplerCreateInfo, nullptr, &sampler));
}



uint32_t VulkanImage::getMemoryTypeIndex(uint32_t memoryTypeBits) {
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


void *VulkanImage::mapMemory() {
    if (pointer)
    {
        return pointer;
    }
    else if (property & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    {
        VK_CHECK_RESULT(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &pointer));
        return pointer;
    }
    return nullptr;
}

void VulkanImage::unmapMemory() {
    if (pointer)
    {
        vkUnmapMemory(device, memory);
    }
}
