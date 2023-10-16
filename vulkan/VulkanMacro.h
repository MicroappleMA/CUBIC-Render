/*
* Assorted Vulkan helper functions
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include "vulkan/vulkan.h"
#include <cassert>
#include <iostream>

#define ERROR_TO_STRING(errorCode) (\
    errorCode == VK_NOT_READY ? "NOT_READY" : \
    errorCode == VK_TIMEOUT ? "TIMEOUT" : \
    errorCode == VK_EVENT_SET ? "EVENT_SET" : \
    errorCode == VK_EVENT_RESET ? "EVENT_RESET" : \
    errorCode == VK_INCOMPLETE ? "INCOMPLETE" : \
    errorCode == VK_ERROR_OUT_OF_HOST_MEMORY ? "ERROR_OUT_OF_HOST_MEMORY" : \
    errorCode == VK_ERROR_OUT_OF_DEVICE_MEMORY ? "ERROR_OUT_OF_DEVICE_MEMORY" : \
    errorCode == VK_ERROR_INITIALIZATION_FAILED ? "ERROR_INITIALIZATION_FAILED" : \
    errorCode == VK_ERROR_DEVICE_LOST ? "ERROR_DEVICE_LOST" : \
    errorCode == VK_ERROR_MEMORY_MAP_FAILED ? "ERROR_MEMORY_MAP_FAILED" : \
    errorCode == VK_ERROR_LAYER_NOT_PRESENT ? "ERROR_LAYER_NOT_PRESENT" : \
    errorCode == VK_ERROR_EXTENSION_NOT_PRESENT ? "ERROR_EXTENSION_NOT_PRESENT" : \
    errorCode == VK_ERROR_FEATURE_NOT_PRESENT ? "ERROR_FEATURE_NOT_PRESENT" : \
    errorCode == VK_ERROR_INCOMPATIBLE_DRIVER ? "ERROR_INCOMPATIBLE_DRIVER" : \
    errorCode == VK_ERROR_TOO_MANY_OBJECTS ? "ERROR_TOO_MANY_OBJECTS" : \
    errorCode == VK_ERROR_FORMAT_NOT_SUPPORTED ? "ERROR_FORMAT_NOT_SUPPORTED" : \
    errorCode == VK_ERROR_SURFACE_LOST_KHR ? "ERROR_SURFACE_LOST_KHR" : \
    errorCode == VK_ERROR_NATIVE_WINDOW_IN_USE_KHR ? "ERROR_NATIVE_WINDOW_IN_USE_KHR" : \
    errorCode == VK_SUBOPTIMAL_KHR ? "SUBOPTIMAL_KHR" : \
    errorCode == VK_ERROR_OUT_OF_DATE_KHR ? "ERROR_OUT_OF_DATE_KHR" : \
    errorCode == VK_ERROR_INCOMPATIBLE_DISPLAY_KHR ? "ERROR_INCOMPATIBLE_DISPLAY_KHR" : \
    errorCode == VK_ERROR_VALIDATION_FAILED_EXT ? "ERROR_VALIDATION_FAILED_EXT" : \
    errorCode == VK_ERROR_INVALID_SHADER_NV ? "ERROR_INVALID_SHADER_NV" : \
    "UNKNOWN_ERROR" \
)

#define VK_CHECK_RESULT(f)																				\
{																										\
	VkResult res = (f);																					\
	if (res != VK_SUCCESS)																				\
	{																									\
		std::cout << "Fatal : VkResult is \"" << ERROR_TO_STRING(res) << "\" in " << __FILE__ << " at line " << __LINE__ << "\n"; \
		assert(res == VK_SUCCESS);																		\
	}																									\
}