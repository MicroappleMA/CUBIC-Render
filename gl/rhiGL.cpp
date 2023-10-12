#include "rhiGL.h"
#include "gl/glew.h"
#include "glfw/glfw3.h"
#include "util/glslUtility.hpp"

#include <cstdio>
#include <cassert>
#include <cuda_gl_interop.h>

#include <unordered_map>

std::unordered_map<int, RHIKeyCode> RHIGL::actionCodeMap = {
        {GLFW_RELEASE,            RELEASE,            },
        {GLFW_PRESS,              PRESS,              },
};

std::unordered_map<int, RHIKeyCode> RHIGL::keyCodeMap = {
        {GLFW_MOUSE_BUTTON_LEFT,  MOUSE_BUTTON_LEFT,  },
        {GLFW_MOUSE_BUTTON_RIGHT, MOUSE_BUTTON_RIGHT, },
        {GLFW_MOUSE_BUTTON_MIDDLE,MOUSE_BUTTON_MIDDLE,},
        {GLFW_KEY_ESCAPE,         KEY_ESCAPE,         },
        {GLFW_KEY_0,              KEY_0,              },
        {GLFW_KEY_1,              KEY_1,              },
        {GLFW_KEY_2,              KEY_2,              },
        {GLFW_KEY_3,              KEY_3,              },
        {GLFW_KEY_4,              KEY_4,              },
        {GLFW_KEY_5,              KEY_5,              },
        {GLFW_KEY_6,              KEY_6,              },
        {GLFW_KEY_7,              KEY_7,              },
        {GLFW_KEY_8,              KEY_8,              },
        {GLFW_KEY_9,              KEY_9,              },
        {GLFW_KEY_A,              KEY_A,              },
        {GLFW_KEY_B,              KEY_B,              },
        {GLFW_KEY_C,              KEY_C,              },
        {GLFW_KEY_D,              KEY_D,              },
        {GLFW_KEY_E,              KEY_E,              },
        {GLFW_KEY_F,              KEY_F,              },
        {GLFW_KEY_G,              KEY_G,              },
        {GLFW_KEY_H,              KEY_H,              },
        {GLFW_KEY_I,              KEY_I,              },
        {GLFW_KEY_J,              KEY_J,              },
        {GLFW_KEY_K,              KEY_K,              },
        {GLFW_KEY_L,              KEY_L,              },
        {GLFW_KEY_M,              KEY_M,              },
        {GLFW_KEY_N,              KEY_N,              },
        {GLFW_KEY_O,              KEY_O,              },
        {GLFW_KEY_P,              KEY_P,              },
        {GLFW_KEY_Q,              KEY_Q,              },
        {GLFW_KEY_R,              KEY_R,              },
        {GLFW_KEY_S,              KEY_S,              },
        {GLFW_KEY_T,              KEY_T,              },
        {GLFW_KEY_U,              KEY_U,              },
        {GLFW_KEY_V,              KEY_V,              },
        {GLFW_KEY_W,              KEY_W,              },
        {GLFW_KEY_X,              KEY_X,              },
        {GLFW_KEY_Y,              KEY_Y,              },
        {GLFW_KEY_Z,              KEY_Z,              },
};

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
        RHIGL::keyCallback(keyCodeMap[key], actionCodeMap[action]);
}

void RHIGL::glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if(RHIGL::mouseButtonCallback)
        RHIGL::mouseButtonCallback(keyCodeMap[button], actionCodeMap[action]);
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