/**
 * @file      main.cpp
 * @brief     Main file for CUDA rasterizer. Handles CUDA-GL interop for display.
 * @authors   Jiayi Chen, Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2023
 * @copyright Jiayi Chen, University of Pennsylvania
 */


#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <cstdint>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_LOADER_IMPLEMENTATION
#include "util/tiny_gltf_loader.h"
#include "util/glslUtility.hpp"
#include "util/utilityCore.hpp"
#include "util/json.hpp"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/transform.hpp"
#include "render/render.h"
#include "rhi.h"
#include "gl/rhiGL.h"
#include "vulkan/rhiVK.h"

int main(int argc, char **argv);
void callRender();

// Callback Functions
void keyCallback(RHIKeyCode key, RHIKeyCode action);
void mouseButtonCallback(RHIKeyCode button, RHIKeyCode action);
void mouseMotionCallback(double xpos, double ypos);
void mouseWheelCallback(double xoffset, double yoffset);

/////////Global Variable/////////

struct
{
    std::unique_ptr<Render> render = nullptr;
    std::unique_ptr<RHI> rhi = nullptr;

    std::string backend;
    int width;
    int height;
    bool vsync;
    bool inverseRender;

    glm::vec3 transition = {0.0f,0.0f,5.0f};
    glm::vec2 rotation = {0.0f, (float)PI};
    MaterialType currentMaterial = Invalid;

    enum {
        None = 0,
        Rotate = 1,
        Translate = 2
    } mouseState = None;

    bool shouldExit = false;
}global;


/////////Entry Function/////////

int main(int argc, char **argv) {
#ifdef DEBUG
    std::cout << "CUBIC Render :: Debug Version\n";
#else
    std::cout << "CUBIC Render :: Release Version\n";
#endif
    std::string configPath;
    if (argc != 2) {
        std::cout << "[Info] Using Default Config Path\n";
        configPath = "./defaultConfig.json";
    }
    else
    {
        std::cout << "[Info] Using Custom Config Path\n";
        configPath = argv[1];
    }

    std::cout<<"[Info] Config = "<<configPath<<"\n";

    std::ifstream ifs(configPath);

    if (!ifs) {
        std::cerr << "[Error] Unable to open config file. Press Enter to exit\n";
        getchar();
        return -1;
    }

    nlohmann::json config = nlohmann::json::parse(ifs);

    global.backend = config["backend"];
    global.width = config["width"];
    global.height = config["height"];
    global.vsync = config["vsync"];
    global.inverseRender = config["inverseRender"];
    std::string modelPath = config["model"];

    std::cout <<  "[Info] Backend = " << global.backend <<
                "\n[Info] Width   = " << global.width   <<
                "\n[Info] Height  = " << global.height  <<
                "\n[Info] VSync   = " << global.vsync   <<
                "\n[Info] Model   = " << modelPath      << "\n";

    // Load light info from json
    std::vector<Light> light;
    for (const auto &l:config["lights"])
    {
        const std::string& typeString = l["type"];
        LightType type = (typeString == "directional" ? DirectionalLight :
                         (typeString == "point" ? PointLight :
                          InvalidLight));

        if(type == InvalidLight)
        {
            std::cout << "[Error] Config Light Type Invalid\n";
            return -1;
        }

        light.push_back({
                            type,
                            {l["position"]["x"],l["position"]["y"],l["position"]["z"]},
                            glm::normalize(glm::vec3{l["direction"]["x"],l["direction"]["y"],l["direction"]["z"]}),
                            {l["color"]["r"],l["color"]["g"],l["color"]["b"]},
                            l["intensity"]
                        });
    }


    // Load scene from disk
    tinygltf::Scene scene;
    tinygltf::TinyGLTFLoader loader;
    std::string err;
    std::string ext = (modelPath.find_last_of('.') != std::string::npos) ?
                        modelPath.substr(modelPath.find_last_of('.') + 1):
                        "";

    bool ret = false;
    if (ext == "glb") {
        // assume binary glTF.
        ret = loader.LoadBinaryFromFile(&scene, &err, modelPath);
    } else {
        // assume ascii glTF.
        ret = loader.LoadASCIIFromFile(&scene, &err, modelPath);
    }

    if (!err.empty() || !ret) {
        std::cout<<"[Error] Failed to parse glTF\n" << err;
        getchar();
        return -1;
    }

    global.rhi = (global.backend == "vulkan" ? std::unique_ptr<RHI>(std::make_unique<RHIVK>()) :
                 (global.backend == "opengl" ? std::unique_ptr<RHI>(std::make_unique<RHIGL>()) :
                  nullptr));

    if (!global.rhi) {
        std::cout<<"[Error] Invalid Backend\n";
        return -1;
    }

    global.rhi->init(global.inverseRender?3 * global.width:global.width, global.height, global.vsync);
    global.rhi->setCallback(mouseMotionCallback,mouseWheelCallback,mouseButtonCallback,keyCallback);

    global.render = std::make_unique<Render>();
    global.render->init(scene, light, global.width, global.height);

    int64_t fps = 0;
    int64_t fpsCounter = 0;
    time_t fpsTime0 = time(nullptr);
    time_t fpsTime1 = time(nullptr);

    while (!global.shouldExit) {
        global.rhi->pollEvents();

        callRender();

        fpsCounter++;
        fpsTime1 = time(nullptr);
        if (fpsTime1 - fpsTime0 >= 1) {
            fps = fpsCounter / (fpsTime1 - fpsTime0);
            fpsCounter = 0;
            fpsTime0 = fpsTime1;
        }

#ifdef DEBUG
        std::string title = "CUBIC Render(debug/" + global.backend + ") | " + std::to_string(fps) + " FPS";
#else
        std::string title = "CUBIC Render(release/" + global.backend + ") | " + std::to_string(fps) + " FPS";
#endif
        global.rhi->draw(title.c_str());
    }

    global.rhi->destroy();
    global.render->free();

    return 0;
}

/////////Render Function/////////

void callRender() {

    const float scale = 1.0f;
    glm::mat4 P = glm::frustum<float>(-scale * ((float)global.width) / ((float)global.height),
                                      scale * ((float)global.width / (float)global.height),
                                      -scale, scale, 1.0, 1000.0);

    glm::mat4 V = glm::rotate((float)PI, glm::vec3{0.0f, 1.0f, 0.0f});

    glm::mat4 M =
            glm::translate(global.transition)
            * glm::rotate(global.rotation.x, glm::vec3{1.0f, 0.0f, 0.0f})
            * glm::rotate(global.rotation.y, glm::vec3{0.0f, 1.0f, 0.0f});

    global.render->overrideMaterial = global.currentMaterial;

    uchar4* buf = reinterpret_cast<uchar4*>(global.rhi->mapBuffer());

    if (global.inverseRender)
    {
        global.render->render(M, V, P, 2 * global.width, 0, 3 * global.width, global.height, buf);
        cudaDeviceSynchronize();

        global.render->inverseRender(global.width, 0, 3 * global.width, global.height, buf);
        cudaDeviceSynchronize();

        global.render->renderTex(6, 0, 0, 3 * global.width, global.height, buf); // Render Baked Texture
        cudaDeviceSynchronize();
    }
    else
    {
        global.render->render(M, V, P, 0, 0, global.width, global.height, buf);
        cudaDeviceSynchronize();
    }

    global.rhi->unmapBuffer();
}

/////////Callback Function/////////

void keyCallback(RHIKeyCode key, RHIKeyCode action)
{
    if (action == PRESS)
    {
        if (key == KEY_ESCAPE) {
            global.shouldExit = true;
        }
        switch (key) {
            case KEY_0: global.currentMaterial = (MaterialType)0; break; // Invalid
            case KEY_1: global.currentMaterial = (MaterialType)1; break; // Depth
            case KEY_2: global.currentMaterial = (MaterialType)2; break; // Mesh
            case KEY_3: global.currentMaterial = (MaterialType)3; break; // UV
            case KEY_4: global.currentMaterial = (MaterialType)4; break; // Normal
            case KEY_5: global.currentMaterial = (MaterialType)5; break; // Texture
            case KEY_6: global.currentMaterial = (MaterialType)6; break; // Environment
            case KEY_7: global.currentMaterial = (MaterialType)7; break; // PBR
            case KEY_8: global.currentMaterial = (MaterialType)8; break; // NPR
            default: global.currentMaterial = (MaterialType)0; break;    // Invalid
        }
    }
}


void mouseButtonCallback(RHIKeyCode button, RHIKeyCode action)
{
    if (action == PRESS)
    {
        if (button == MOUSE_BUTTON_LEFT)
        {
            global.mouseState = global.Rotate;
        }
        else if (button == MOUSE_BUTTON_RIGHT)
        {
            global.mouseState = global.Translate;
        }

    }
    else if (action == RELEASE)
    {
        global.mouseState = global.None;
    }
}

void mouseMotionCallback(double xpos, double ypos)
{
    const double s_r = 0.01;
    const double s_t = 0.01;

    static double lastx = 0;
    static double lasty = 0;

    double diffx = xpos - lastx;
    double diffy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    if (global.mouseState == global.Rotate)
    {
        //rotate
        global.rotation.x -= (float)s_r * diffy;
        global.rotation.y += (float)s_r * diffx;
    }
    else if (global.mouseState == global.Translate)
    {
        //translate
        global.transition.x += (float)(s_t * diffx);
        global.transition.y += (float)(-s_t * diffy);
    }
}

void mouseWheelCallback(double xoffset, double yoffset)
{
    const double sensitivity = 0.3;
    global.transition.z -= (float)(sensitivity * yoffset);
}