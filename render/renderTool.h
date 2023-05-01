/**
 * @file      rasterizeTools.h
 * @brief     Tools/utility functions for rasterization.
 * @authors   Yining Karl Li
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#pragma once

#include <cmath>
#include "external/include/glm/glm.hpp"
#include "util/utilityCore.hpp"
#include "dataType.h"

// CHECKITOUT
/**
 * Finds the axis aligned bounding box for a given triangle.
 */
__host__ __device__ static
AABB getAABBForTriangle(const glm::vec3 *tri) {
    AABB aabb;
    aabb.min = glm::vec3(
            glm::min(glm::min(tri[0].x, tri[1].x), tri[2].x),
            glm::min(glm::min(tri[0].y, tri[1].y), tri[2].y),
            glm::min(glm::min(tri[0].z, tri[1].z), tri[2].z));
    aabb.max = glm::vec3(
            glm::max(glm::max(tri[0].x, tri[1].x), tri[2].x),
            glm::max(glm::max(tri[0].y, tri[1].y), tri[2].y),
            glm::max(glm::max(tri[0].z, tri[1].z), tri[2].z));
    return aabb;
}

// CHECKITOUT
/**
 * Calculate the signed area of a given triangle.
 */
__host__ __device__ static
float calculateSignedArea(const glm::vec3 *tri) {
    return 0.5f * ((tri[2].x - tri[0].x) * (tri[1].y - tri[0].y) - (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y));
}

// CHECKITOUT
/**
 * Helper function for calculating barycentric coordinates.
 */
__host__ __device__ static
float calculateBarycentricCoordinateValue(const glm::vec2 &a, const glm::vec2 &b, const glm::vec2 &c, const glm::vec3 *tri) {
    glm::vec3 baryTri[3];
    baryTri[0] = glm::vec3(a, 0);
    baryTri[1] = glm::vec3(b, 0);
    baryTri[2] = glm::vec3(c, 0);
    return calculateSignedArea(baryTri) / calculateSignedArea(tri);
}

// CHECKITOUT
/**
 * Calculate barycentric coordinates.
 */
__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(const glm::vec3 *tri, const glm::vec2 &point) {
    float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), point, glm::vec2(tri[2].x, tri[2].y), tri);
    float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), glm::vec2(tri[1].x, tri[1].y), point, tri);
    float alpha = 1.0f - beta - gamma;
    return {alpha, beta, gamma};
}

// CHECKITOUT
/**
 * Check if a barycentric coordinate is within the boundaries of a triangle.
 */
__host__ __device__ static
bool isBarycentricCoordInBounds(const glm::vec3 &barycentricCoord) {
    return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
           barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
           barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

// CHECKITOUT
/**
 * For a given barycentric coordinate, compute the corresponding z position
 * (i.e. depth) on the triangle.
 */
__host__ __device__ static
float getZAtCoordinate(const glm::vec3 &barycentricCoord, const glm::vec3 *tri) {
    return -(barycentricCoord.x * tri[0].z
           + barycentricCoord.y * tri[1].z
           + barycentricCoord.z * tri[2].z);
}

__host__ __device__ static
float getViewZAtCoordinate(const glm::vec3 &barycentricCoord, const glm::vec3 &triViewZ) {
    return 1 / (barycentricCoord.x / triViewZ.x
              + barycentricCoord.y / triViewZ.y
              + barycentricCoord.z / triViewZ.z);
}

__host__ __device__ static
glm::vec3 getInterpolationCoef(const glm::vec3 &barycentricCoord, const glm::vec3 &triViewZ)
{
    float viewZ = getViewZAtCoordinate(barycentricCoord,triViewZ);
    return {barycentricCoord.x * viewZ / triViewZ.x,
            barycentricCoord.y * viewZ / triViewZ.y,
            barycentricCoord.z * viewZ / triViewZ.z};
}

__device__ static
glm::vec3 _sampleTex(const TextureData *tex, const unsigned int &index)
{
    return {tex[index * 3] / 255.0f,
            tex[index * 3 + 1] / 255.0f,
            tex[index * 3 + 2] / 255.0f};
}

__device__ static
glm::vec3 sampleTex(const Tex &tex, const glm::vec2 &UV)
{
    if(tex.data==nullptr || UV.x<0 || UV.x>=1 || UV.y<0 || UV.y>=1)
        return {1,0,0}; // Ensure UV is valid

    float w = (float)tex.width * (UV.x - glm::floor(UV.x));
    float h = (float)tex.height * (UV.y - glm::floor(UV.y));

    int firstW = (int)(w);
    int firstH = (int)(h);

    int secondW = glm::min(firstW + 1, tex.width - 1);
	int secondH = glm::min(firstH + 1, tex.height - 1);

	float x_gap = w - (float)firstW;
	float y_gap = h - (float)firstH;

    glm::vec3 color1 = _sampleTex(tex.data,firstW + firstH * tex.width);
    glm::vec3 color2 = _sampleTex(tex.data,secondW + firstH * tex.width);
    glm::vec3 color3 = _sampleTex(tex.data,firstW + secondH * tex.width);
    glm::vec3 color4 = _sampleTex(tex.data,secondW + secondH * tex.width);

	return glm::mix(glm::mix(color1, color2, x_gap),
                    glm::mix(color3, color4, x_gap),
                    y_gap);
}

__device__ static
glm::vec2 LightingFunGGX_FV(float dotLH, float roughness)
{
    float alpha = roughness*roughness;

    //F
    float F_a, F_b;
    float dotLH5 = glm::pow(glm::clamp(1.0f - dotLH, 0.0f, 1.0f), 5.0f);
    F_a = 1.0f;
    F_b = dotLH5;

    //V
    float vis;
    float k = alpha * 0.5f;
    float k2 = k*k;
    float invK2 = 1.0f - k2;
    vis = 1.0f/(dotLH*dotLH*invK2 + k2);

    return glm::vec2((F_a - F_b)*vis, F_b*vis);
}

__device__ static
float LightingFuncGGX_D(float dotNH, float roughness)
{
    float alpha = roughness*roughness;
    float alphaSqr = alpha*alpha;
    float denom = dotNH * dotNH * (alphaSqr - 1.0f) + 1.0f;

    return alphaSqr / (PI*denom*denom);
}

__device__ static
glm::vec3 GGX_Spec(glm::vec3 Normal, glm::vec3 HalfVec, float Roughness, glm::vec3 SpecularColor, glm::vec2 paraFV)
{
    float NoH = glm::clamp(glm::dot(Normal, HalfVec), 0.0f, 1.0f);

    float D = LightingFuncGGX_D(NoH * NoH * NoH * NoH, Roughness);
    glm::vec2 FV_helper = paraFV;

    glm::vec3 F0 = SpecularColor;
    glm::vec3 FV = F0*FV_helper.x + glm::vec3(FV_helper.y, FV_helper.y, FV_helper.y);

    return D * FV;
}