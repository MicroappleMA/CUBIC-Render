cmake_minimum_required(VERSION 3.24)

# Only support Windows and Visual Studio build toolchain
project(cuda_rasterizer CUDA CXX)

# Set up include and lib paths
include_directories(.)
include_directories("external")
include_directories("external/include")

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

# CUDA
find_package(CUDA 12.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)

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
list(APPEND CUDA_NVCC_FLAGS_DEBUG          -O0 -g -G -DDEBUG)
list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBINFO -O2 -g -lineinfo)
list(APPEND CUDA_NVCC_FLAGS_RELEASE        -O3    -DNDEBUG)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CMAKE_CXX_FLAGS                       "/MD /EHsc /D _CRT_SECURE_NO_WARNINGS")
set(CMAKE_CXX_FLAGS_DEBUG                 "/Od /Zi /D DEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO        "/O2 /Zi")
set(CMAKE_CXX_FLAGS_RELEASE               "/O2 /D NDEBUG")

add_subdirectory(src)
add_subdirectory(util)

cuda_add_executable(${CMAKE_PROJECT_NAME}
        "src/main.hpp"
        "src/main.cpp"
        )


# CUDA linker options
find_package(Threads REQUIRED)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${CMAKE_PROJECT_NAME}
    src
    util
    ${CORELIBS}
    )

#add_custom_command(
#    TARGET ${CMAKE_PROJECT_NAME}
#    POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_directory
#        ${CMAKE_SOURCE_DIR}/shaders
#        ${CMAKE_BINARY_DIR}/shaders
#    )
