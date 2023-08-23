/**
 * @file      render.h
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Jiayi Chen, Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2023
 * @copyright Jiayi Chen, University of Pennsylvania
 */

#pragma once

#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_LOADER_IMPLEMENTATION
#include "util/tiny_gltf_loader.h"
#include "dataType.h"

class Render {
public:
    void init(const tinygltf::Scene & scene, const std::vector<Light> &light, const int &w, const int &h);
    void render(const glm::mat4 & M, const glm::mat4 & V, const glm::mat4 & P,
                const int &beginW, const int &beginH, const int &bufferW, const int &bufferH, uchar4* const pbo);
    void inverseRender(const int &beginW, const int &beginH, const int &bufferW, const int &bufferH, uchar4* const pbo);
    void renderTex(int texIndex, const int &beginW, const int &beginH, const int &bufferW, const int &bufferH, uchar4* const pbo);
    void free();

    MaterialType overrideMaterial = Invalid;

private:
    int width = 0;
    int height = 0;

    dim3 blockSize2d = {0,0,0};
    dim3 blockCount2d = {0,0,0};

    SceneInfo sceneInfo;

    Primitive *dev_primitives = nullptr;
    Fragment *dev_fragmentBuffer = nullptr;
    Tile *dev_tileBuffer = nullptr;
    glm::vec3 *dev_framebuffer = nullptr;

    Light *dev_lights = nullptr;

    glm::mat4 M,V,P;
};


