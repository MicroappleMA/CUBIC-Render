#include "VulkanSharedBuffer.h"
#include "VulkanMacro.h"
#include "util/checkCUDAError.h"
#include "vulkan/vulkan.h"

#include <dxgi1_2.h>
#include <aclapi.h>
#include <cuda_runtime_api.h>

VulkanSharedBuffer::VulkanSharedBuffer(VkPhysicalDevice _physicalDevice, VkDevice _device, VkDeviceSize _size, VkBufferUsageFlags _usage):
physicalDevice(_physicalDevice), device(_device), size(_size), usage(_usage)
{
    createWindowsSecurityAttributes();
    createVulkanBuffer();
    mapCudaBuffer();
}

VulkanSharedBuffer::~VulkanSharedBuffer()
{
    unmapCudaBuffer();
    destroyVulkanBuffer();
    destroyWindowsSecurityAttributes();
}

void VulkanSharedBuffer::createWindowsSecurityAttributes() {
    psecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(
            1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
    if (!psecurityDescriptor) {
        throw std::runtime_error(
                "Failed to allocate memory for security descriptor");
    }

    PSID *ppSID = (PSID *)((PBYTE)psecurityDescriptor +
                           SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

    InitializeSecurityDescriptor(psecurityDescriptor,
                                 SECURITY_DESCRIPTOR_REVISION);

    SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority =
            SECURITY_WORLD_SID_AUTHORITY;
    AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0,
                             0, 0, 0, 0, 0, ppSID);

    EXPLICIT_ACCESS explicitAccess;
    ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
    explicitAccess.grfAccessPermissions =
            STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
    explicitAccess.grfAccessMode = SET_ACCESS;
    explicitAccess.grfInheritance = INHERIT_ONLY;
    explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
    explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
    explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

    SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

    SetSecurityDescriptorDacl(psecurityDescriptor, TRUE, *ppACL, FALSE);

    securityAttributes.nLength = sizeof(securityAttributes);
    securityAttributes.lpSecurityDescriptor = psecurityDescriptor;
    securityAttributes.bInheritHandle = TRUE;
}

void VulkanSharedBuffer::destroyWindowsSecurityAttributes() {
    PSID *ppSID = (PSID *)((PBYTE)psecurityDescriptor +
                           SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

    if (*ppSID) {
        FreeSid(*ppSID);
    }
    if (*ppACL) {
        LocalFree(*ppACL);
    }
    free(psecurityDescriptor);
}

void VulkanSharedBuffer::createVulkanBuffer() {
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = usage;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkExternalMemoryBufferCreateInfo externalMemoryBufferCreateInfo{};
    externalMemoryBufferCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalMemoryBufferCreateInfo.handleTypes = vulkanExternalMemoryHandleType;

    bufferCreateInfo.pNext = &externalMemoryBufferCreateInfo;

    VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer));

    VkMemoryRequirements memoryRequirements{};

    vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

    VkExportMemoryWin32HandleInfoKHR exportMemoryWin32HandleInfo{};
    exportMemoryWin32HandleInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    exportMemoryWin32HandleInfo.name = nullptr;
    exportMemoryWin32HandleInfo.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
    exportMemoryWin32HandleInfo.pAttributes = &securityAttributes;

    VkExportMemoryAllocateInfo exportMemoryAllocateInfo{};

    exportMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportMemoryAllocateInfo.handleTypes = vulkanExternalMemoryHandleType;
    exportMemoryAllocateInfo.pNext = &exportMemoryWin32HandleInfo;

    VkMemoryAllocateInfo memoryAllocateInfo{};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = getMemoryTypeIndex(memoryRequirements.memoryTypeBits);
    memoryAllocateInfo.pNext = &exportMemoryAllocateInfo;

    VK_CHECK_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &memory));

    VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, memory, 0));
}

void VulkanSharedBuffer::destroyVulkanBuffer() {
    if (buffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, buffer, nullptr);
    if (memory != VK_NULL_HANDLE)
        vkFreeMemory(device, memory, nullptr);
}

void VulkanSharedBuffer::mapCudaBuffer() {
    VkMemoryGetWin32HandleInfoKHR memoryGetWin32HandleInfo{};
    memoryGetWin32HandleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    memoryGetWin32HandleInfo.memory = memory;
    memoryGetWin32HandleInfo.handleType = vulkanExternalMemoryHandleType;

    HANDLE handle;
    auto vkGetMemoryWin32HandleKHR = VK_DEVICE_LOAD(device, vkGetMemoryWin32HandleKHR);
    assert(vkGetMemoryWin32HandleKHR!=VK_NULL_HANDLE);
    VK_CHECK_RESULT(vkGetMemoryWin32HandleKHR(device, &memoryGetWin32HandleInfo, &handle));
    assert(handle!=nullptr);

    cudaExternalMemoryHandleDesc cudaHandleDesc{};
    cudaHandleDesc.type = cudaExternalMemoryHandleType;
    cudaHandleDesc.size = size;
    cudaHandleDesc.handle.win32.handle = handle;
    cudaHandleDesc.handle.win32.name = nullptr;

    cudaImportExternalMemory(&cudaMemory, &cudaHandleDesc);
    checkCUDAError("cudaImportExternalMemory");

    cudaExternalMemoryBufferDesc cudaBufferDesc{};
    cudaBufferDesc.size = size;
    cudaBufferDesc.offset = 0;
    cudaBufferDesc.flags = 0;

    cudaExternalMemoryGetMappedBuffer(&cudaBuffer, cudaMemory, &cudaBufferDesc);
    checkCUDAError("cudaExternalMemoryGetMappedBuffer");
}

void VulkanSharedBuffer::unmapCudaBuffer() {
    if (cudaBuffer)
    {
        cudaDestroyExternalMemory(cudaMemory);
        checkCUDAError("cudaDestroyExternalMemory");
    }
}

uint32_t VulkanSharedBuffer::getMemoryTypeIndex(uint32_t memoryTypeBits) {
    VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemoryProperties);
    for (size_t i=0; i<physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        if (
                (memoryTypeBits & (1 << i)) &&
                ((memoryPropertyFlags & physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags) == memoryPropertyFlags)
                )
        {
            return i;
        }
    }
    return ~0;
}