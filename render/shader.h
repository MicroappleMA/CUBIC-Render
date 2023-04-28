#pragma once

#include <cmath>
#include "external/include/glm/glm.hpp"
#include "util/utilityCore.hpp"

#include "dataType.h"


__device__
VertexOut vertexShader(const VertexIn &vert, const glm::mat4 &M, const glm::mat4 &V, const glm::mat4 &P)
{
    VertexOut vertexOut;

    vertexOut.objectPos = glm::vec4(vert.pos,1);
    vertexOut.worldPos = M * vertexOut.objectPos;
    vertexOut.viewPos = V * vertexOut.worldPos;
    vertexOut.clipPos = P * vertexOut.viewPos;
    vertexOut.clipPos /= vertexOut.clipPos.w;
    vertexOut.objectNor = vert.nor;
    vertexOut.worldNor = glm::transpose(glm::inverse(glm::mat3(M))) * vertexOut.objectNor;
    vertexOut.viewNor = glm::transpose(glm::inverse(glm::mat3(V))) * vertexOut.worldNor;

    // windowPos is auto generate

    // TODO: Apply vertex transformation here
    // Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
    // Then divide the pos by its w element to transform into NDC space
    // Finally transform x and y to viewport space


    return vertexOut;
}

__device__
glm::vec3 fragmentShader(const Fragment& frag)
{
    // Render Pass
    glm::vec3 color;

    switch (frag.material) {
        case InValid:
            break;
        case Debug:
            color = (1 - frag.depth) / 2 * glm::vec3(1,1,1);
            break;
        case Direct:
            color = frag.color;
            break;
        case Lambert:
            glm::vec3 lightNor = {0.574, 0.574, 0.574}; // Use for temp test
            color = frag.color * glm::dot(lightNor,frag.objectNor);
            break;
    }

    // TODO: add your fragment shader code here

    return color;
}