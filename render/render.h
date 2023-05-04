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
    static Render& getInstance(){
        static Render instance;
        return instance;
    }

    void init(const tinygltf::Scene & scene, const std::vector<Light> &light,
              const int &w, const int &h, const int &beginW, const int &beginH,
              const int &bufferW, const int &bufferH, uchar4* const pbo);
    void render(const glm::mat4 & M, const glm::mat4 & V, const glm::mat4 & P);
    void free();
    void setPboConfig(const int &beginW = -1, const int &beginH = -1,
                      const int &bufferW = -1, const int &bufferH = -1,
                      uchar4* const pbo = nullptr);

    MaterialType overrideMaterial = Invalid;

private:
    Render () = default;
    ~Render () = default;
    Render (const Render &) = delete;
    Render & operator=(const Render &) = delete;

    int width = 0;
    int height = 0;

    int bufferBeginWidth = 0;
    int bufferBeginHeight = 0;

    int bufferWidth = 0;
    int bufferHeight = 0;

    uchar4 *buffer = nullptr;

    SceneInfo sceneInfo;

    Primitive *dev_primitives = nullptr;
    Fragment *dev_fragmentBuffer = nullptr;
    Tile *dev_tileBuffer = nullptr;
    glm::vec3 *dev_framebuffer = nullptr;

    Light *dev_lights = nullptr;
};


