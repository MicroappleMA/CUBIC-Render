#include "rhiGL.h"
#include "gl/glew.h"
#include "glfw/glfw3.h"
#include "util/glslUtility.hpp"

#include <cstdio>
#include <cassert>
#include <cuda_gl_interop.h>

using namespace std;

PFN_keyCallback RHIGL::keyCallback = nullptr;
PFN_mouseButtonCallback RHIGL::mouseButtonCallback = nullptr;
PFN_scrollCallback RHIGL::scrollCallback = nullptr;
PFN_cursorPosCallback RHIGL::cursorPosCallback = nullptr;

void RHIGL::glfwErrorCallback(int error, const char *description) {
    fputs(description, stderr);
}

void RHIGL::glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if(RHIGL::keyCallback)
        RHIGL::keyCallback(key, action);
}

void RHIGL::glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if(RHIGL::mouseButtonCallback)
        RHIGL::mouseButtonCallback(button, action);
}

void RHIGL::glfwScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    if(RHIGL::scrollCallback)
        RHIGL::scrollCallback(xoffset, yoffset);
}

void RHIGL::glfwCursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    if(RHIGL::cursorPosCallback)
        RHIGL::cursorPosCallback(xpos, ypos);
}

void RHIGL::initVAO() {
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


GLuint RHIGL::initShader() {
    GLuint program = glslUtility::createDefaultProgram(attributeLocations, 2);
    GLint location;

    glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) {
        glUniform1i(location, 0);
    }

    return program;
}

void RHIGL::initTextures() {
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
                  GL_UNSIGNED_BYTE, NULL);
}

void RHIGL::initPBO() {
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

void RHIGL::destroyPBO() {
    if (pbo) {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(pbo);

        glBindBuffer(GL_ARRAY_BUFFER, pbo);
        glDeleteBuffers(1, &pbo);

        pbo = (GLuint)NULL;
    }
}

void RHIGL::destroyTexture() {
    if(displayImage)
    {
        glDeleteTextures(1, &displayImage);
        displayImage = (GLuint)NULL;
    }
}

void RHIGL::init(int width, int height, bool vsync)
{
    this->width = width;
    this->height = height;
    this->vsync = vsync;

    glfwSetErrorCallback(glfwErrorCallback);
    int glfwInitRes = glfwInit();
    assert(glfwInitRes == GLFW_TRUE);

    glfwWindowHint(GLFW_DOUBLEBUFFER, vsync);
    glfwWindowHint(GLFW_RESIZABLE, false);
    window = glfwCreateWindow(width, height, "", nullptr, nullptr);
    assert(window);
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    GLenum glewInitRes = glewInit();
    assert(glewInitRes == GLEW_OK);

    initVAO();
    initTextures();
    initPBO();

    GLuint passthroughProgram;
    passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);
}

void RHIGL::setCallback(PFN_cursorPosCallback newCursorPosCallback, PFN_scrollCallback newScrollCallback,
                        PFN_mouseButtonCallback newMouseButtonCallback, PFN_keyCallback newKeyCallback)
{
    RHIGL::keyCallback = newKeyCallback;
    RHIGL::mouseButtonCallback = newMouseButtonCallback;
    RHIGL::scrollCallback = newScrollCallback;
    RHIGL::cursorPosCallback = newCursorPosCallback;

    glfwSetKeyCallback(window, RHIGL::glfwKeyCallback);
    glfwSetMouseButtonCallback(window, RHIGL::glfwMouseButtonCallback);
    glfwSetScrollCallback(window, RHIGL::glfwScrollCallback);
    glfwSetCursorPosCallback(window, RHIGL::glfwCursorPosCallback);
}

void RHIGL::pollEvents() {
    glfwPollEvents();
}

void *RHIGL::mapBuffer()
{
    void* dptr;
    cudaGLMapBufferObject((void **)&dptr, pbo);
    return dptr;
}

void RHIGL::unmapBuffer()
{
    cudaGLUnmapBufferObject(pbo);
}

void RHIGL::draw(const char* title) {
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

void RHIGL::destroy()
{
    destroyPBO();
    destroyTexture();
    glfwDestroyWindow(window);
    glfwTerminate();
}