/**
 * @file      shader.h
 * @brief     custom shaders of the render pipeline.
 * @authors   Jiayi Chen
 * @date      2023-2023
 * @copyright Jiayi Chen
 */

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
void geometryShader(Primitive &prim)
{
    // Calc Tangent
    const glm::vec4 (&pos)[3] = {prim.v[0].worldPos,prim.v[1].worldPos,prim.v[2].worldPos};
    const glm::vec2 (&uv)[3] = {prim.v[0].uv,prim.v[1].uv,prim.v[2].uv};
    prim.v[0].tangent = getTangentAtCoordinate(uv,pos,prim.v[0].worldNor);
    prim.v[1].tangent = getTangentAtCoordinate(uv,pos,prim.v[1].worldNor);
    prim.v[2].tangent = getTangentAtCoordinate(uv,pos,prim.v[2].worldNor);

    // Set Debug Mesh Color
    prim.v[0].color = {1,0,0};
    prim.v[1].color = {0,1,0};
    prim.v[2].color = {0,0,1};
}

__device__ static
glm::vec3 envShader(const Fragment& frag)
{
    const VertexOut &in = frag.in;
    glm::mat3 TBN = getTBN(in.tangent, in.worldNor);
    glm::vec3 normalTex = sampleTex2d(in.tex[2], in.uv);
    glm::vec3 normal = glm::normalize(TBN * normalTex);
    glm::vec3 view = glm::normalize(-glm::vec3(in.worldPos));
    glm::vec3 reflect = -glm::reflect(view, normal);
    return sampleTexCubemap(in.tex[5], reflect);
}

__device__ static
glm::vec3 pbrShader(const Fragment& frag, Light *light, unsigned int lightNum)
{
    glm::vec3 color = {0,0,0};
    const VertexOut &in = frag.in;

    glm::vec3 diffuseTex = sampleTex2d(in.tex[0], in.uv);
    glm::vec3 specularTex = sampleTex2d(in.tex[1], in.uv);
    glm::vec3 normalTex = sampleTex2d(in.tex[2], in.uv);
    glm::vec3 roughnessTex = sampleTex2d(in.tex[3], in.uv);
    glm::vec3 emissionTex = sampleTex2d(in.tex[4], in.uv);
    glm::vec3 ViewVec = glm::normalize(-glm::vec3(in.worldPos));
    glm::mat3 TBN = getTBN(in.tangent, in.worldNor);

    // glm::vec3 NormalVec = in.worldNor; // Use direct normal instead of normal Texture
    normalTex *= glm::vec3{2,2,1};
    normalTex -= glm::vec3{1,1,0};

    glm::vec3 NormalVec = glm::normalize(TBN * normalTex);
    glm::vec3 ReflectVec = -glm::reflect(ViewVec, NormalVec);
    float NoV = glm::clamp(glm::dot(NormalVec, ViewVec), 0.0f, 1.0f);

    glm::vec3 environmentTex = sampleTexCubemap(in.tex[5], ReflectVec);

    float roughness = glm::clamp(roughnessTex.x, 0.05f, 1.0f);
    float metallic = roughnessTex.y;

    float energyConservation = 1.0f - roughness;

    // For-each Light Loop
    for (int i = 0; i < lightNum; i++)
    {
        glm::vec3 LightVec = {0,0,1};
        if (light[i].type == DirectionalLight)
            LightVec = glm::normalize(light[i].direction);
        else if (light[i].type == PointLight)
            LightVec = glm::normalize(glm::vec3(in.worldPos) - light[i].position);
        LightVec = -LightVec;

        glm::vec3 HalfVec = glm::normalize(ViewVec + LightVec);

        float NoL = glm::dot(LightVec, NormalVec);
        float LoH = glm::clamp(glm::dot(LightVec, HalfVec), 0.0f, 1.0f);

        if (NoL > 0.0f)
        {
            glm::vec3 specularTerm = pbrGGX_Spec(NormalVec, HalfVec, roughness, specularTex, pbrGGX_FV(LoH, roughness)) * energyConservation;
            color += (diffuseTex + specularTerm) * NoL * glm::vec3(light[i].color) * light[i].intensity;
        }
        // TODO: Support SH Lighting
    }
    color += diffuseTex * environmentTex * energyConservation * metallic + emissionTex;
    return color;
}

__device__ static
glm::vec3 fragmentShader(const Fragment& frag, Light *light, unsigned int lightNum)
{
    // Render Pass
    glm::vec3 color = {0,0,0};
    const VertexOut &in = frag.in;

    switch (in.material) {
        case Invalid:
            break;
        case Depth:
            color = (1 - frag.depth) / 2 * glm::vec3(1,1,1);
            break;
        case Mesh:
            color = in.color;
            break;
        case UV:
            color = glm::vec3(in.uv,0);
            break;
        case Normal:
            color = in.objectNor;
            break;
        case Tex0:
            color = sampleTex2d(in.tex[0], in.uv);
            break;
        case Env:
            color = envShader(frag);
            break;
        case PBR:
            color = pbrShader(frag, light, lightNum);
            break;
    }

    // TODO: add your fragment shader code here

    return color;
}

__device__ static
void inverseFragmentShader(glm::vec3 &color, Fragment &frag, Light *light, unsigned int lightNum)
{
    VertexOut &in = frag.in;

    const float learningRate = 0.05;

    //assume Tex[6] is the texture that is waiting for baking
    const int bakedTexIndex = 6;
    Tex &bakedTex = in.tex[bakedTexIndex];

    if(in.material!=Invalid && bakedTex.data)
    {
        if(in.uv.x<0 || in.uv.x>=1 || in.uv.y<0 || in.uv.y>=1)
            return; // Ensure UV is valid

        color = glm::clamp(color, 0.0f, 1.0f);

        float w = (float)bakedTex.width * in.uv.x;
        float h = (float)bakedTex.height * in.uv.y;

        int W1 = (int)(w);
        int H1 = (int)(h);

        int W2 = glm::min(W1 + 1, bakedTex.width - 1);
        int H2 = glm::min(H1 + 1, bakedTex.height - 1);

        float x = w - (float)W1;
        float y = h - (float)H1;

        glm::vec3 A = _sampleTex(bakedTex.data, W1 + H1 * bakedTex.width);
        glm::vec3 B = _sampleTex(bakedTex.data, W2 + H1 * bakedTex.width);
        glm::vec3 C = _sampleTex(bakedTex.data, W1 + H2 * bakedTex.width);
        glm::vec3 D = _sampleTex(bakedTex.data, W2 + H2 * bakedTex.width);

        glm::vec3 M = glm::mix(A, C, y);
        glm::vec3 N = glm::mix(B, D, y);

        glm::vec3 R = glm::mix(M, N, x);

        // L1 Loss Function
        glm::vec3 loss = glm::abs(R - color);

        // Calc Partial Gradient
        glm::vec3 gradLR = {R.x > color.x ? 1.0f - color.x : color.x - 1.0f,
                            R.y > color.y ? 1.0f - color.y : color.y - 1.0f,
                            R.z > color.z ? 1.0f - color.z : color.z - 1.0f};

        const glm::vec3 vecOne = {1.0f,1.0f,1.0f};

        glm::vec3 gradRM = glm::mix(vecOne,N,x);
        glm::vec3 gradRN = glm::mix(vecOne,M,1-x);

        glm::vec3 gradMA = glm::mix(vecOne,C,y);
        glm::vec3 gradMC = glm::mix(vecOne,A,1-y);
        glm::vec3 gradNB = glm::mix(vecOne,D,y);
        glm::vec3 gradND = glm::mix(vecOne,B,1-y);

        // Chain Rule
        glm::vec3 gradLA = gradLR * gradRM * gradMA;
        glm::vec3 gradLB = gradLR * gradRN * gradNB;
        glm::vec3 gradLC = gradLR * gradRM * gradMC;
        glm::vec3 gradLD = gradLR * gradRN * gradND;

        // Update Texture
        A -= gradLA * learningRate;
        B -= gradLB * learningRate;
        C -= gradLC * learningRate;
        D -= gradLD * learningRate;

        // Write Back
        _writeTex(bakedTex.data, W1 + H1 * bakedTex.width, A);
        _writeTex(bakedTex.data, W2 + H1 * bakedTex.width, B);
        _writeTex(bakedTex.data, W1 + H2 * bakedTex.width, C);
        _writeTex(bakedTex.data, W2 + H2 * bakedTex.width, D);
        // _writeTex(bakedTex.data, W1 + H1 * bakedTex.width, color);
        // _writeTex(bakedTex.data, W1 + H2 * bakedTex.width, color);
        // _writeTex(bakedTex.data, W2 + H1 * bakedTex.width, color);
        // _writeTex(bakedTex.data, W2 + H2 * bakedTex.width, color);

        color = R;
    }
}