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

using namespace std;

int main(int argc, char **argv);
void callRender();
void keyCallback(int key, int action);
void mouseButtonCallback(int button, int action);
void mouseMotionCallback(double xpos, double ypos);
void mouseWheelCallback(double xoffset, double yoffset);

/////////Global Variable/////////

unique_ptr<Render> render = nullptr;
unique_ptr<RHI> rhi = nullptr;

int frame = 0;
int fpstracker = 0;
double seconds = time(NULL);
int fps = 0;

bool vsync;
int width;
int height;
bool inverseRender;

float scale = 1.0f;
float x_trans = 0.0f, y_trans = 0.0f, z_trans = 5.0f;
float x_angle = 0.0f, y_angle = (float)PI;
MaterialType currentMaterial = Invalid;

double lastx = 0;
double lasty = 0;

enum ControlState { NONE = 0, ROTATE, TRANSLATE };
ControlState mouseState = NONE;

bool shouldExit = false;

/////////Entry Function/////////

int main(int argc, char **argv) {
#ifdef DEBUG
    cout << "CUBIC Render :: Debug Version\n";
#else
    cout << "CUBIC Render :: Release Version\n";
#endif
    if (argc != 2) {
        cout << "Usage: [Config Path]. Press Enter to exit\n";
        getchar();
        return 0;
    }

    cout<<"Config = "<<argv[1]<<"\n";

    std::ifstream ifs(argv[1]);

    if (!ifs) {
        std::cerr << "Unable to open config file. Press Enter to exit\n";
        getchar();
        return 1;   // return with error code 1
    }

    nlohmann::json config = nlohmann::json::parse(ifs);

    vsync = config["vsync"];
    width = config["width"];
    height = config["height"];
    inverseRender = config["inverseRender"];
    string modelPath = config["model"];

    cout<<"Width = "<<width<<", Height = "<<height<<", VSync = "<<vsync<<"\nModel = "<<modelPath<<"\n";

    // Load light info from json
    vector<Light> light;
    for (auto &&l:config["lights"])
    {
        string typeString = l["type"];
        LightType type = (typeString == "directional" ? DirectionalLight :
                         (typeString == "point" ? PointLight : InvalidLight));

        if(type == InvalidLight)
        {
            cout << "[Error] Config Light Type Invalid\n";
            return 2;
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

    if (!err.empty()) {
        cout<<"Err: "+err;
    }

    if (!ret) {
        cout<<"Failed to parse glTF\n";
        getchar();
        return -1;
    }

    render = make_unique<Render>();
    rhi = make_unique<RHIGL>();

    rhi->init(inverseRender?3 * width:width, height, vsync);
    rhi->setCallback(mouseMotionCallback,mouseWheelCallback,mouseButtonCallback,keyCallback);

    render->init(scene, light, width, height);

    while (!shouldExit) {
        rhi->pollEvents();

        callRender();

        time_t seconds2 = time (NULL);
        if (seconds2 - seconds >= 1) {
            fps = fpstracker / (seconds2 - seconds);
            fpstracker = 0;
            seconds = seconds2;
        }

        string title = "CUBIC Render | " + to_string((int)fps) + " FPS";
        rhi->draw(title.c_str());
    }

    rhi->destroy();
    render->free();

    return 0;
}


/////////Render Function/////////

void callRender() {

    glm::mat4 P = glm::frustum<float>(-scale * ((float)width) / ((float)height),
                                      scale * ((float)width / (float)height),
                                      -scale, scale, 1.0, 1000.0);

    glm::mat4 V = glm::rotate((float)PI, glm::vec3{0.0f, 1.0f, 0.0f});

    glm::mat4 M =
            glm::translate(glm::vec3(x_trans, y_trans, z_trans))
            * glm::rotate(x_angle, glm::vec3{1.0f, 0.0f, 0.0f})
            * glm::rotate(y_angle, glm::vec3{0.0f, 1.0f, 0.0f});

    glm::mat3 MV_normal = glm::transpose(glm::inverse(glm::mat3(V) * glm::mat3(M)));
    glm::mat4 MV = V * M;
    glm::mat4 MVP = P * MV;

    render->overrideMaterial = currentMaterial;

    uchar4* buf = reinterpret_cast<uchar4*>(rhi->mapBuffer());

    if (inverseRender)
    {
        render->render(M, V, P, 2 * width, 0, 3 * width, height, buf);
        cudaDeviceSynchronize();

        render->inverseRender(width, 0, 3 * width, height, buf);
        cudaDeviceSynchronize();

        render->renderTex(6, 0, 0, 3 * width, height, buf); // Render Baked Texture
        cudaDeviceSynchronize();
    }
    else
    {
        render->render(M, V, P, 0, 0, width, height, buf);
        cudaDeviceSynchronize();
    }

    rhi->unmapBuffer();

    frame++;
    fpstracker++;
}

/////////Callback Function/////////

// Index must in range[0,9]
#define BIND_MATERIAL_KEY(index) \
{if (key == GLFW_KEY_0 + (index)) {currentMaterial = (MaterialType)(index); }}

void keyCallback(int key, int action)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_ESCAPE) {
            shouldExit = true;
        }
        BIND_MATERIAL_KEY(0);
        BIND_MATERIAL_KEY(1);
        BIND_MATERIAL_KEY(2);
        BIND_MATERIAL_KEY(3);
        BIND_MATERIAL_KEY(4);
        BIND_MATERIAL_KEY(5);
        BIND_MATERIAL_KEY(6);
        BIND_MATERIAL_KEY(7);
        BIND_MATERIAL_KEY(8);
    }
}


void mouseButtonCallback(int button, int action)
{
    if (action == GLFW_PRESS)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            mouseState = ROTATE;
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            mouseState = TRANSLATE;
        }

    }
    else if (action == GLFW_RELEASE)
    {
        mouseState = NONE;
    }
}

void mouseMotionCallback(double xpos, double ypos)
{
    const double s_r = 0.01;
    const double s_t = 0.01;

    double diffx = xpos - lastx;
    double diffy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    if (mouseState == ROTATE)
    {
        //rotate
        x_angle -= (float)s_r * diffy;
        y_angle += (float)s_r * diffx;
    }
    else if (mouseState == TRANSLATE)
    {
        //translate
        x_trans += (float)(s_t * diffx);
        y_trans += (float)(-s_t * diffy);
    }
}

void mouseWheelCallback(double xoffset, double yoffset)
{
    const double sensitivity = 0.3;
    z_trans -= (float)(sensitivity * yoffset);
}