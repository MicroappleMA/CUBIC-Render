#pragma once

#include "main/rhi.h"
#include "gl/glew.h"
#include "glfw/glfw3.h"

class rhiGL: public rhi {
public:
    void init() override;
    void initSurface(int width, int height, bool vsync) override;
    void initPipeline() override;
    void setCallback(PFN_cursorPosCallback cursorPosCallback,
                             PFN_scrollCallback scrollCallback,
                             PFN_mouseButtonCallback mouseButtonCallback,
                             PFN_keyCallback keyCallback) override;
    void* mapBuffer() override;
    void unmapBuffer() override;
    void draw(const char* title) override;
    void destroy() override;

private:
    static void glfwErrorCallback(int error, const char *description);
    static void glfwKeyCallback(GLFWwindow* window,int key, int scancode, int action, int mods);
    static void glfwMouseButtonCallback(GLFWwindow* window,int button, int action, int mods);
    static void glfwScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void glfwCursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    void initTextures();
    void initVAO();
    GLuint initShader();
    void initPBO();
    void destroyPBO();
    void destroyTexture();


    static PFN_keyCallback keyCallback;
    static PFN_mouseButtonCallback mouseButtonCallback;
    static PFN_scrollCallback scrollCallback;
    static PFN_cursorPosCallback cursorPosCallback;
    int width, height;
    bool vsync;
    GLuint positionLocation = 0;
    GLuint texcoordsLocation = 1;
    const char *attributeLocations[2] = { "Position", "Tex" };
    GLuint pbo = (GLuint)NULL;
    GLuint displayImage;

    GLFWwindow *window;
};


