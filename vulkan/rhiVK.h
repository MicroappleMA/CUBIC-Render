#pragma once

#include "main/rhi.h"
#include "vulkan/vulkan.h"
#include "glfw/glfw3.h"

class RHIVK: public RHI {
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
};

