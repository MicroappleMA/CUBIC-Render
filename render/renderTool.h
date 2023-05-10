/**
 * @file      rasterizeTools.h
 * @brief     Tools/utility functions for rasterization.
 * @authors   Jiayi Chen, Yining Karl Li
 * @date      2012-2023
 * @copyright Jiayi Chen, University of Pennsylvania
 */

#pragma once

#include <cmath>
#include "external/include/glm/glm.hpp"
#include "util/utilityCore.hpp"
#include "dataType.h"

/**
 * Finds the axis aligned bounding box for a given triangle.
 */
__host__ __device__ static
AABB getAABBForTriangle(const glm::vec3* __restrict__ tri) {
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

/**
 * Calculate the signed area of a given triangle.
 */
__host__ __device__ static
float calculateSignedArea(const glm::vec3* __restrict__ tri) {
    return 0.5f * ((tri[2].x - tri[0].x) * (tri[1].y - tri[0].y) - (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y));
}

/**
 * Helper function for calculating barycentric coordinates.
 */
__host__ __device__ static
float calculateBarycentricCoordinateValue(const glm::vec2& __restrict__ a, const glm::vec2& __restrict__ b, const glm::vec2& __restrict__ c, const glm::vec3* __restrict__ tri) {
    glm::vec3 baryTri[3];
    baryTri[0] = glm::vec3(a, 0);
    baryTri[1] = glm::vec3(b, 0);
    baryTri[2] = glm::vec3(c, 0);
    return calculateSignedArea(baryTri) / calculateSignedArea(tri);
}

/**
 * Calculate barycentric coordinates.
 */
__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(const glm::vec3* __restrict__ tri, const glm::vec2& __restrict__ point) {
    float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), point, glm::vec2(tri[2].x, tri[2].y), tri);
    float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), glm::vec2(tri[1].x, tri[1].y), point, tri);
    float alpha = 1.0f - beta - gamma;
    return {alpha, beta, gamma};
}

/**
 * Check if a barycentric coordinate is within the boundaries of a triangle.
 */
__host__ __device__ static
bool isBarycentricCoordInBounds(const glm::vec3& __restrict__ barycentricCoord) {
    return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
           barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
           barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

/**
 * For a given barycentric coordinate, compute the corresponding z position
 * (i.e. depth) on the triangle.
 */
__host__ __device__ static
float getZAtCoordinate(const glm::vec3& __restrict__ barycentricCoord, const glm::vec3* __restrict__ tri) {
    return -(barycentricCoord.x * tri[0].z
           + barycentricCoord.y * tri[1].z
           + barycentricCoord.z * tri[2].z);
}

__host__ __device__ static
float getViewZAtCoordinate(const glm::vec3& __restrict__ barycentricCoord, const glm::vec3& __restrict__ triViewZ) {
    return 1 / (barycentricCoord.x / triViewZ.x
              + barycentricCoord.y / triViewZ.y
              + barycentricCoord.z / triViewZ.z);
}

__host__ __device__ static
glm::vec3 getInterpolationCoef(const glm::vec3& __restrict__ barycentricCoord, const glm::vec3& __restrict__ triViewZ)
{
    float viewZ = getViewZAtCoordinate(barycentricCoord,triViewZ);
    return {barycentricCoord.x * viewZ / triViewZ.x,
            barycentricCoord.y * viewZ / triViewZ.y,
            barycentricCoord.z * viewZ / triViewZ.z};
}

__device__ static
glm::vec3 _sampleTex(const TextureData* __restrict__ tex, const unsigned int& __restrict__ index)
{
    return {tex[index * 3] / 255.0f,
            tex[index * 3 + 1] / 255.0f,
            tex[index * 3 + 2] / 255.0f};
}

__device__ static
void _writeTex(TextureData* __restrict__ tex, const unsigned int& __restrict__ index, const glm::vec3& __restrict__ value)
{
    tex[index * 3]     = glm::clamp(value.x, 0.0f, 1.0f) * 255;
    tex[index * 3 + 1] = glm::clamp(value.y, 0.0f, 1.0f) * 255;
    tex[index * 3 + 2] = glm::clamp(value.z, 0.0f, 1.0f) * 255;
}

__device__ static
glm::vec3 sampleTex2d(const Tex& __restrict__ tex, const glm::vec2& __restrict__ UV)
{
    if(tex.data==nullptr || UV.x<0 || UV.x>=1 || UV.y<0 || UV.y>=1)
        return {1,0,0}; // Ensure UV is valid

    float w = (float)tex.width * UV.x;
    float h = (float)tex.height * UV.y;

    int W1 = (int)(w);
    int H1 = (int)(h);

    int W2 = glm::min(W1 + 1, tex.width - 1);
	int H2 = glm::min(H1 + 1, tex.height - 1);

	float x = w - (float)W1;
	float y = h - (float)H1;

    glm::vec3 A = _sampleTex(tex.data, W1 + H1 * tex.width);
    glm::vec3 B = _sampleTex(tex.data, W2 + H1 * tex.width);
    glm::vec3 C = _sampleTex(tex.data, W1 + H2 * tex.width);
    glm::vec3 D = _sampleTex(tex.data, W2 + H2 * tex.width);

	return glm::mix(glm::mix(A, B, x),
                    glm::mix(C, D, x),
                    y);
}

__device__  static
float SphericalTheta(const glm::vec3& __restrict__ v)
{
    return glm::acos(glm::clamp(v.y, -1.0f, 1.0f));
}

__device__  static
float SphericalPhi(const glm::vec3& __restrict__ v)
{
    float p = (glm::atan(v.z, v.x));

    return (p < 0.0f) ? (p + TWO_PI) : p;
}

__device__ static
glm::vec3 sampleTexCubemap(const Tex& __restrict__ tex, const glm::vec3& __restrict__ dir)
{
    return sampleTex2d(tex, {SphericalPhi(dir) * INV_2PI, SphericalTheta(dir) * INV_PI});
}

__device__ static
glm::vec2 pbrGGX_FV(float dotLH, float roughness)
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

    return {(F_a - F_b)*vis, F_b*vis};
}

__device__ static
float pbrGGX_D(float dotNH, float roughness)
{
    float alpha = roughness*roughness;
    float alphaSqr = alpha*alpha;
    float denom = dotNH * dotNH * (alphaSqr - 1.0f) + 1.0f;

    return alphaSqr / (PI*denom*denom);
}

__device__ static
glm::vec3 pbrGGX_Spec(const glm::vec3& __restrict__ Normal, const glm::vec3& __restrict__ HalfVec, float Roughness, const glm::vec3& __restrict__ SpecularColor, const glm::vec2& __restrict__ paraFV)
{
    float NoH = glm::clamp(glm::dot(Normal, HalfVec), 0.0f, 1.0f);

    float D = pbrGGX_D(NoH * NoH * NoH * NoH, Roughness);
    glm::vec2 FV_helper = paraFV;

    glm::vec3 FV = SpecularColor * FV_helper.x + glm::vec3(FV_helper.y, FV_helper.y, FV_helper.y);

    return D * FV;
}

__device__ static
glm::vec3 getTangentAtCoordinate(const glm::vec2* __restrict__ uv, const glm::vec4* __restrict__ pos, const glm::vec3& __restrict__ normal)
{
    float u0 = uv[1].x - uv[0].x;
    float u1 = uv[2].x - uv[0].x;

    float v0 = uv[1].y - uv[0].y;
    float v1 = uv[2].y - uv[0].y;

    float dino = u0 * v1 - v0 * u1;

    glm::vec3 Pos1 = glm::vec3(pos[1] - pos[0]);
    glm::vec3 Pos2 = glm::vec3(pos[2] - pos[0]);
    glm::vec3 Pos3 = glm::vec3(pos[2] - pos[1]);

    glm::vec2 UV1 = uv[1] - uv[0];
    glm::vec2 UV2 = uv[2] - uv[0];

    glm::vec3 tan;
    glm::vec3 bit;
    glm::vec3 nor;// = normal;

    if (dino != 0.0f)
    {
        tan = glm::normalize( (UV2.y * Pos1 - UV1.y * Pos2) / dino );
        bit = glm::normalize( (Pos2 - UV2.x * tan) / UV2.y );

        nor = glm::normalize(glm::cross(tan, bit));
    }
    else
    {

        nor = glm::vec3(1.0f, 0.0f, 0.0f);
        bit = glm::normalize(glm::cross(nor, tan));
        tan = glm::normalize(glm::cross(bit, nor));
    }

    //U flip
    if (glm::dot(nor, glm::normalize(glm::cross(Pos1, Pos3))) < 0.0f)
    {
        tan = -(tan);
    }

    bit = glm::normalize(glm::cross(normal, tan));
    tan = glm::normalize(glm::cross(bit, normal));

    return tan;
}

__device__ static
glm::mat3 getTBN(const glm::vec3& __restrict__ tangent, const glm::vec3& __restrict__ normal)
{
    return {glm::normalize(tangent),
            glm::normalize(glm::cross(normal, tangent)),
            glm::normalize(normal)};
}
