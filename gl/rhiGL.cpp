#include "rhiGL.h"
#include "gl/glew.h"
#include "glfw/glfw3.h"
#include "util/glslUtility.hpp"

#include <cstdio>
#include <cassert>
#include <cuda_gl_interop.h>

using namespace std;

PFN_keyCallback rhiGL::keyCallback = nullptr;
PFN_mouseButtonCallback rhiGL::mouseButtonCallback = nullptr;
PFN_scrollCallback rhiGL::scrollCallback = nullptr;
PFN_cursorPosCallback rhiGL::cursorPosCallback = nullptr;

void rhiGL::glfwErrorCallback(int error, const char *description) {
    fputs(description, stderr);
}

void rhiGL::glfwKeyCallback(GLFWwindow* window,int key, int scancode, int action, int mods)
{
    if(rhiGL::keyCallback)
        rhiGL::keyCallback(key,action);
}

void rhiGL::glfwMouseButtonCallback(GLFWwindow* window,int button, int action, int mods)
{
    if(rhiGL::mouseButtonCallback)
        rhiGL::mouseButtonCallback(button, action);
}

void rhiGL::glfwScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    if(rhiGL::scrollCallback)
        rhiGL::scrollCallback(xoffset, yoffset);
}

void rhiGL::glfwCursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    if(rhiGL::cursorPosCallback)
        rhiGL::cursorPosCallback(xpos,ypos);
}

void rhiGL::initVAO() {
    GLfloat vertices[] = {
            -1.0f, -1.0f,
            1.0f, -1.0f,
            1.0f,  1.0f,
            -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
            1.0f, 1.0f,
            0.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}


GLuint rhiGL::initShader() {
    GLuint program = glslUtility::createDefaultProgram(attributeLocations, 2);
    GLint location;

    glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) {
        glUniform1i(location, 0);
    }

    return program;
}

void rhiGL::initTextures() {
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
                  GL_UNSIGNED_BYTE, NULL);
}

void rhiGL::initPBO() {
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);
}

void rhiGL::destroyPBO() {
    if (pbo) {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(pbo);

        glBindBuffer(GL_ARRAY_BUFFER, pbo);
        glDeleteBuffers(1, &pbo);

        pbo = (GLuint)NULL;
    }
}

void rhiGL::destroyTexture() {
    if(displayImage)
    {
        glDeleteTextures(1, &displayImage);
        displayImage = (GLuint)NULL;
    }
}

void rhiGL::init()
{
    glfwSetErrorCallback(glfwErrorCallback);
    assert(glfwInit());
}

void rhiGL::initSurface(int width, int height, bool vsync)
{
    this->width = width;
    this->height = height;
    this->vsync = vsync;
    glfwWindowHint(GLFW_DOUBLEBUFFER, vsync);
    window = glfwCreateWindow(width, height, "", NULL, NULL);
    assert(window);
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    assert(glewInit() == GLEW_OK);
}

void rhiGL::initPipeline()
{
    initVAO();
    initTextures();
    initPBO();

    GLuint passthroughProgram;
    passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);
}

void rhiGL::setCallback(PFN_cursorPosCallback newCursorPosCallback, PFN_scrollCallback newScrollCallback,
                        PFN_mouseButtonCallback newMouseButtonCallback, PFN_keyCallback newKeyCallback)
{
    rhiGL::keyCallback = newKeyCallback;
    rhiGL::mouseButtonCallback = newMouseButtonCallback;
    rhiGL::scrollCallback = newScrollCallback;
    rhiGL::cursorPosCallback = newCursorPosCallback;

    glfwSetKeyCallback(window, rhiGL::glfwKeyCallback);
    glfwSetMouseButtonCallback(window, rhiGL::glfwMouseButtonCallback);
    glfwSetScrollCallback(window, rhiGL::glfwScrollCallback);
    glfwSetCursorPosCallback(window, rhiGL::glfwCursorPosCallback);
}

void *rhiGL::mapBuffer()
{
    void* dptr;
    cudaGLMapBufferObject((void **)&dptr, pbo);
    return dptr;
}

void rhiGL::unmapBuffer()
{
    cudaGLUnmapBufferObject(pbo);
}

void rhiGL::draw(const char* title) {
    glfwPollEvents();
    glfwSetWindowTitle(window, title);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glClear(GL_COLOR_BUFFER_BIT);

    // VAO, shader program, and texture already bound
    glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
    if(vsync)
        glfwSwapBuffers(window);
    else
        glFlush();
}

void rhiGL::destroy()
{
    destroyPBO();
    destroyTexture();
    glfwDestroyWindow(window);
    glfwTerminate();
}