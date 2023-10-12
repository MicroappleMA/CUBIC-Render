#pragma once

enum RHIKeyCode{
    RELEASE,
    PRESS,
    MOUSE_BUTTON_LEFT,
    MOUSE_BUTTON_RIGHT,
    MOUSE_BUTTON_MIDDLE,
    KEY_ESCAPE,
    KEY_0,
    KEY_1,
    KEY_2,
    KEY_3,
    KEY_4,
    KEY_5,
    KEY_6,
    KEY_7,
    KEY_8,
    KEY_9,
    KEY_A,
    KEY_B,
    KEY_C,
    KEY_D,
    KEY_E,
    KEY_F,
    KEY_G,
    KEY_H,
    KEY_I,
    KEY_J,
    KEY_K,
    KEY_L,
    KEY_M,
    KEY_N,
    KEY_O,
    KEY_P,
    KEY_Q,
    KEY_R,
    KEY_S,
    KEY_T,
    KEY_U,
    KEY_V,
    KEY_W,
    KEY_X,
    KEY_Y,
    KEY_Z,
};

typedef void (*PFN_cursorPosCallback)(double, double);
typedef void (*PFN_scrollCallback)(double, double);
typedef void (*PFN_mouseButtonCallback)(RHIKeyCode, RHIKeyCode);
typedef void (*PFN_keyCallback)(RHIKeyCode, RHIKeyCode);

class RHI {
public:
    virtual void init(int width, int height, bool vsync)=0;
    virtual void setCallback(PFN_cursorPosCallback cursorPosCallback,
                             PFN_scrollCallback scrollCallback,
                             PFN_mouseButtonCallback mouseButtonCallback,
                             PFN_keyCallback keyCallback)=0;
    virtual void pollEvents()=0;
    virtual void* mapBuffer()=0;
    virtual void unmapBuffer()=0;
    virtual void draw(const char* title)=0;
    virtual void destroy()=0;
};
