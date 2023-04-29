#pragma once

#include <cmath>
#include "external/include/glm/glm.hpp"
#include "util/utilityCore.hpp"

#include "dataType.h"
#include "renderTool.h"


__device__ static
VertexOut vertexShader(const VertexIn &in, const glm::mat4 &M, const glm::mat4 &V, const glm::mat4 &P)
{
    VertexOut vertexOut;

    vertexOut.objectPos = glm::vec4(in.position, 1);
    vertexOut.worldPos = M * vertexOut.objectPos;
    vertexOut.viewPos = V * vertexOut.worldPos;
    vertexOut.clipPos = P * vertexOut.viewPos;
    vertexOut.clipPos /= vertexOut.clipPos.w;
    vertexOut.objectNor = in.normal;
    vertexOut.worldNor = glm::transpose(glm::inverse(glm::mat3(M))) * vertexOut.objectNor;
    vertexOut.viewNor = glm::transpose(glm::inverse(glm::mat3(V))) * vertexOut.worldNor;
    vertexOut.uv = in.uv;

    // windowPos,material,tex are auto generate

    // TODO: Apply vertex transformation here
    // Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
    // Then divide the pos by its w element to transform into NDC space
    // Finally transform x and y to viewport space


    return vertexOut;
}

__device__ static
glm::vec3 fragmentShader(const Fragment& frag)
{
    // Render Pass
    glm::vec3 color;
    const VertexOut &in = frag.in;

    switch (in.material) {
        case Invalid:
            break;
        case Depth:
            color = (1 - frag.depth) / 2 * glm::vec3(1,1,1);
            break;
        case Debug:
            color = frag.in.color;
            break;
        case UV:
            color = glm::vec3(in.uv,0);
            break;
        case Tex0:
            color = sampleTex(in.tex[0], in.uv);
            break;
        case Lambert:
            glm::vec3 lightNor = {0.574, 0.574, 0.574}; // Use for temp test
            color = frag.color * glm::dot(lightNor,in.objectNor);
            break;
    }

    // TODO: add your fragment shader code here

    return color;
}