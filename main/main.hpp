/**
 * @file      main.hpp
 * @brief     Main file for CUDA rasterizer. Handles CUDA-GL interop for display.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania
 */

#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "util/json.hpp"
#include "util/glslUtility.hpp"
#include "util/utilityCore.hpp"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/transform.hpp"
#include "render/render.h"
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "util/tiny_gltf.h"

using namespace std;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------
int frame;
int fpstracker;
double seconds;
int fps = 0;
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char *attributeLocations[] = { "Position", "Tex" };
GLuint pbo = (GLuint)NULL;
GLuint displayImage;
uchar4 *dptr;

GLFWwindow *window;

//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width = 0;
int height = 0;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char **argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------
bool init(const tinygltf::Model & model, const vector<Light> & light);
void initPBO();
void initCuda();
void initTextures();
void initVAO();
GLuint initShader();

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deletePBO(GLuint *pbo);
void deleteTexture(GLuint *tex);

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------
void mainLoop();
void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);

//----------------------------
//----- util -----------------
//----------------------------
std::string getFilePathExtension(const std::string &FileName);

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mouseMotionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseWheelCallback(GLFWwindow* window, double xoffset, double yoffset);