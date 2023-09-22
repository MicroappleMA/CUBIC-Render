cmake_minimum_required(VERSION 3.24)

# Only support Windows and Visual Studio build toolchain
project(CUBIC-Render CUDA CXX)

# CUDA
find_package(CUDA 12.0 REQUIRED)

# Set up include and lib paths
include_directories(.)
include_directories("external")
include_directories("external/include")
include_directories(${CUDA_INCLUDE_DIRS})

set(EXTERNAL_LIB_PATH "external/lib/win")
link_directories(${EXTERNAL_LIB_PATH})
list(APPEND CMAKE_LIBRARY_PATH "${EXTERNAL_LIB_PATH}")

# Find up and set up core dependency libs

# GLFW
set(GLFW_LIBRARY_DIR "${EXTERNAL_LIB_PATH}/glfw")
find_library(GLFW_LIBRARY "glfw3" HINTS "${GLFW_LIBRARY_DIR}" REQUIRED)
list(APPEND CORELIBS ${GLFW_LIBRARY})

# GLEW
set(GLEW_LIBRARY_DIR "${EXTERNAL_LIB_PATH}/glew")
add_definitions(-DGLEW_STATIC)
find_library(GLEW_LIBRARY "glew32s" HINTS "${GLEW_LIBRARY_DIR}" REQUIRED)
list(APPEND CORELIBS ${GLEW_LIBRARY})

# OpenGL
find_package(OpenGL REQUIRED)
list(APPEND CORELIBS ${OPENGL_LIBRARY})

# Vulkan
set(VK_LIBRARY_DIR "${EXTERNAL_LIB_PATH}/vulkan")
add_definitions(-DVK_USE_PLATFORM_WIN32_KHR)
find_library(VK_LIBRARY "vulkan-1" HINTS "${VK_LIBRARY_DIR}" REQUIRED)
list(APPEND CORELIBS ${VK_LIBRARY})

# glslang
set(GLSLANG_LIBRARY_DIR "${EXTERNAL_LIB_PATH}/glslang")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(GLSLANG_SUB_LIB_NAMES
        "GenericCodeGend" "glslang-default-resource-limitsd" "glslangd" "HLSLd" "MachineIndependentd"
        "OGLCompilerd" "OSDependentd" "SPIRV-Tools-optd" "SPIRV-Toolsd" "SPIRVd" "SPVRemapperd")
else()
    set(GLSLANG_SUB_LIB_NAMES
        "GenericCodeGen" "glslang-default-resource-limits" "glslang" "HLSL" "MachineIndependent"
        "OGLCompiler" "OSDependent" "SPIRV-Tools-opt" "SPIRV-Tools" "SPIRV" "SPVRemapper")
endif()
foreach(GLSLANG_SUB_LIB_NAME IN LISTS GLSLANG_SUB_LIB_NAMES)
    find_library(GLSLANG_LIBRARY_${GLSLANG_SUB_LIB_NAME} ${GLSLANG_SUB_LIB_NAME} HINTS "${GLSLANG_LIBRARY_DIR}" REQUIRED)
    list(APPEND CORELIBS ${GLSLANG_LIBRARY_${GLSLANG_SUB_LIB_NAME}})
endforeach()


set(STDIO_LIBRARY "legacy_stdio_definitions.lib")
list(APPEND CORELIBS ${STDIO_LIBRARY})

# Enable C++11 for host code
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Set up different build configurations
set(CMAKE_CONFIGURATION_TYPES Debug;RelWithDebInfo;Release
    CACHE STRING "Set configuration types" FORCE)
set(CMAKE_CUDA_FLAGS_DEBUG                "-O0 -G -g -DDEBUG")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO       "-O2 -g -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE              "-O3    -DNDEBUG")
set(CMAKE_CXX_FLAGS                       "/MD /EHsc /D _CRT_SECURE_NO_WARNINGS")
set(CMAKE_CXX_FLAGS_DEBUG                 "/Od /Zi /D DEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO        "/O2 /Zi")
set(CMAKE_CXX_FLAGS_RELEASE               "/O2 /D NDEBUG")
set(CUDA_VERBOSE_BUILD ON)
set(CUDA_PROPAGATE_HOST_FLAGS ON)

set(UTIL_FILES
    "util/utilityCore.cpp"
    "util/glslUtility.cpp")

set(RENDER_FILES
    "render/render.cu"
    "render/renderFunc.cu")

set(RHI_FILES
    "gl/rhiGL.cpp"
    "vulkan/rhiVK.cpp")

set(MAIN_FILES
    "main/main.cpp")

add_executable(${CMAKE_PROJECT_NAME} ${UTIL_FILES} ${RENDER_FILES} ${RHI_FILES} ${MAIN_FILES})

# CUDA linker options
find_package(Threads REQUIRED)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)

target_link_libraries(${CMAKE_PROJECT_NAME} ${CORELIBS})
