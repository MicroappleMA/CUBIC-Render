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
glm::vec3 fragmentShader(const Fragment& frag, Light *light, unsigned int lightNum)
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
        case Mesh:
            color = frag.in.color;
            break;
        case UV:
            color = glm::vec3(in.uv,0);
            break;
        case Normal:
            color = in.objectNor;
            break;
        case Tex0:
            color = sampleTex(in.tex[0], in.uv);
            break;
        case PBR:
            // glm::vec3 lightNor = light[0].direction;
            // color = sampleTex(in.tex[0], in.uv) * glm::dot(lightNor,in.worldNor);
            for (int i = 0; i < lightNum; i++)
            {
                glm::vec3 LightVec = glm::normalize(light[i].direction);
                glm::vec3 ViewVec = glm::vec3(0.0f, 0.0f, 1.0f);

                glm::vec3 HalfVec = glm::normalize(ViewVec + LightVec);
                glm::vec3 NormalVec = in.worldNor; // TODO: Use normal texture

                glm::vec3 ReflectVec = -glm::reflect(ViewVec, NormalVec);


                glm::vec3 specularTerm = glm::vec3(0.0f);

                glm::vec3 diffuseTex = sampleTex(in.tex[0],in.uv);
                glm::vec3 specularTex = sampleTex(in.tex[1],in.uv);
                glm::vec3 roughnessTex = sampleTex(in.tex[3],in.uv);
                glm::vec3 emissionTex = sampleTex(in.tex[4],in.uv);

                float NoL = glm::dot(LightVec, NormalVec);


                float roughness = glm::clamp(roughnessTex.x, 0.05f, 1.0f);
                float metallic = roughnessTex.y;

                float LoH = glm::clamp(glm::dot(LightVec, HalfVec), 0.0f, 1.0f);
                float NoV = glm::clamp(glm::dot(NormalVec, ViewVec), 0.0f, 1.0f);


                float energyConservation = 1.0f - roughness;

                if (NoL > 0.0f)
                {
                    specularTerm = GGX_Spec(NormalVec, HalfVec, roughness, specularTex, LightingFunGGX_FV(LoH, roughness)) *energyConservation;
                    color += (diffuseTex + specularTerm) * NoL * glm::vec3(light[i].color) * light[i].intensity;
                }

                // TODO: Support IBL
                // glm::vec3 envColor = getEnvTextColor(thisFragment.envTexWidth, thisFragment.envTexHeight, ReflectVec, thisFragment.dev_envTex);
                color += diffuseTex * energyConservation * metallic;

                color += emissionTex;
            }
            break;
    }

    // TODO: add your fragment shader code here

    return color;
}