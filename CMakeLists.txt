cmake_minimum_required(VERSION 3.24)

# Only support Windows and Visual Studio build toolchain
project(cuda_rasterizer CUDA CXX)

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
set(GLFW_INCLUDE_DIR "external/include")
set(GLFW_LIBRARY_DIR "${EXTERNAL_LIB_PATH}")
find_library(GLFW_LIBRARY "glfw3" HINTS "${GLFW_LIBRARY_DIR}" REQUIRED)

# GLEW
set(GLEW_INCLUDE_DIR "external/include")
set(GLEW_LIBRARY_DIR "${EXTERNAL_LIB_PATH}/glew32s.lib")
add_definitions(-DGLEW_STATIC)
find_library(GLEW_LIBRARY "glew32s" HINTS "${GLEW_LIBRARY_DIR}" REQUIRED)

# OpenGL
find_package(OpenGL REQUIRED)

set(STDIO_LIBRARY "legacy_stdio_definitions.lib")

set(CORELIBS
    "${GLFW_LIBRARY}"
    "${OPENGL_LIBRARY}"
    "${GLEW_LIBRARY}"
    "${STDIO_LIBRARY}"
    )

# Enable C++11 for host code
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

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
    "util/utilityCore.hpp"
    "util/utilityCore.cpp"
    "util/glslUtility.hpp"
    "util/glslUtility.cpp"
    "util/tiny_gltf_loader.h")

set(CUDA_FILES
    "src/rasterize.cu"
    "src/rasterize.h"
    "src/rasterizeTools.h"
    "src/shader.h"
    "src/data.h")

set(MAIN_FILES
    "src/main.hpp"
    "src/main.cpp")

add_executable(${CMAKE_PROJECT_NAME} ${UTIL_FILES} ${CUDA_FILES} ${MAIN_FILES})

# CUDA linker options
find_package(Threads REQUIRED)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)

target_link_libraries(${CMAKE_PROJECT_NAME} ${CORELIBS})
