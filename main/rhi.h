#pragma once

typedef void (*PFN_cursorPosCallback)(double, double);
typedef void (*PFN_scrollCallback)(double, double);
typedef void (*PFN_mouseButtonCallback)(int, int);
typedef void (*PFN_keyCallback)(int, int);

class RHI {
public:
    virtual void init(int width, int height, bool vsync)=0;
    virtual void initPipeline()=0;
    virtual void setCallback(PFN_cursorPosCallback cursorPosCallback,
                             PFN_scrollCallback scrollCallback,
                             PFN_mouseButtonCallback mouseButtonCallback,
                             PFN_keyCallback keyCallback)=0;
    virtual void* mapBuffer()=0;
    virtual void unmapBuffer()=0;
    virtual void draw(const char* title)=0;
    virtual void destroy()=0;
};
