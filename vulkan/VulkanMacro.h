/*
* Assorted Vulkan helper functions
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include "vulkan/vk_enum_string_helper.h"
#include <cassert>
#include <iostream>

#define VK_CHECK_RESULT(f)																				          \
{																										          \
	VkResult res = (f);																					          \
	if (res != VK_SUCCESS)																				          \
	{																									          \
		std::cerr << "Fatal : " << string_VkResult(res) << " in " << __FILE__ << " at line " << __LINE__ << "\n"; \
		abort();																		                          \
	}																									          \
}

#define VK_DEVICE_LOAD(device, func)         reinterpret_cast<PFN_##func>(vkGetDeviceProcAddr(device,#func))
#define VK_INSTANCE_LOAD(instance, func)     reinterpret_cast<PFN_##func>(vkGetInstanceProcAddr(instance,#func))