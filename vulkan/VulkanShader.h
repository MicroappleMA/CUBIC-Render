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
                                     "layout(location = 0) in vec3 fragColor;\n"
                                     "layout(location = 0) out vec4 outColor;\n"
                                     "void main() {\n"
                                     "    outColor = vec4(fragColor, 1.0);\n"
                                     "}";
}