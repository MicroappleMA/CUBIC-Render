#pragma once

namespace defaultShader
{
    const char* vertex =             "#version 460\n"
                                     "\n"
                                     "layout(location = 0) out vec3 fragColor;\n"
                                     "\n"
                                     "vec2 positions[3] = vec2[](\n"
                                     "    vec2(0.0, -0.5),\n"
                                     "    vec2(0.5, 0.5),\n"
                                     "    vec2(-0.5, 0.5)\n"
                                     ");\n"
                                     "\n"
                                     "vec3 colors[3] = vec3[](\n"
                                     "    vec3(1.0, 0.0, 0.0),\n"
                                     "    vec3(0.0, 1.0, 0.0),\n"
                                     "    vec3(0.0, 0.0, 1.0)\n"
                                     ");\n"
                                     "\n"
                                     "void main() {\n"
                                     "    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);\n"
                                     "    fragColor = colors[gl_VertexIndex];\n"
                                     "}";

    const char* fragment =           "#version 460\n"
                                     "\n"
                                     "layout(location = 0) in vec3 fragColor;\n"
                                     "layout(location = 0) out vec4 outColor;\n"
                                     "void main() {\n"
                                     "    outColor = vec4(fragColor, 1.0);\n"
                                     "}";
}