#pragma once

namespace defaultShader
{
    const char* const vertex =       "#version 460\n"
                                     "layout(location = 0) in vec3 position;\n"
                                     "layout(location = 1) in vec3 color;\n"
                                     "layout(location = 0) out vec3 vertexColor;\n"
                                     "void main() {\n"
                                     "    gl_Position = vec4(position, 1.0);\n"
                                     "    vertexColor = color;\n"
                                     "}";

    const char* const fragment =     "#version 460\n"
                                     "layout(push_constant) uniform PushConstant{vec2 resolution;}pushConstant;\n"
                                     "layout(set = 0, binding = 0) uniform sampler2D sampler0;\n"
                                     "layout(location = 0) in vec3 fragColor;\n"
                                     "layout(location = 0) out vec4 outColor;\n"
                                     "void main() {\n"
                                     "    float x = 1 - gl_FragCoord.x / pushConstant.resolution.x;\n"
                                     "    float y = gl_FragCoord.y / pushConstant.resolution.y;\n"
                                     "    outColor = textureLod(sampler0, vec2(x,y), 0.0f);\n"
                                     "}";
}