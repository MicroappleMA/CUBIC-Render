/**
 * @file      renderFunc.h
 * @brief     All functions of CUDA-accelerated rasterization pipeline.
 * @authors   Jiayi Chen, Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2023
 * @copyright Jiayi Chen, University of Pennsylvania
 */

#pragma once

#include "util/tiny_gltf_loader.h"
#include "dataType.h"


__global__
void _vertexTransform(int numVertices,PrimitiveBuffer primitive,glm::mat4 M, glm::mat4 V, glm::mat4 P,int width, int height);

__global__
void _inverseVertexTransform(int numVertices,PrimitiveBuffer primitive,glm::mat4 M, glm::mat4 V, glm::mat4 P,int width, int height);

__global__
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* __restrict__ dev_primitives, PrimitiveBuffer primitive);

__global__
void _inversePrimitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* __restrict__ dev_primitives, PrimitiveBuffer primitive);

__global__
void _clearTileBuffer(Tile* __restrict__ dev_tileBuffer, int width, int height, int tileSize);

__global__
void _generateTileBuffer(int numPrimitives, const Primitive* __restrict__ dev_primitives, Tile* __restrict__ dev_tileBuffer, int width, int height, int tileSize);

__global__
void _rasterize(const Primitive* __restrict__ dev_primitives, const Tile* __restrict__ dev_tileBuffer, Fragment* __restrict__ dev_fragmentBuffer, int width, int height, int tileSize);

__global__
void _inverseRasterize(Primitive* __restrict__ dev_primitives, Tile* __restrict__ dev_tileBuffer, Fragment* __restrict__ dev_fragmentBuffer, int width, int height, int tileSize);

__global__
void _fragmentShading(glm::vec3* __restrict__ framebuffer, Fragment* __restrict__ fragmentBuffer, Light* __restrict__ light, unsigned int lightNum, MaterialType overrideMaterial, int w, int h);

__global__
void _inverseFragmentShading(glm::vec3* __restrict__ framebuffer, Fragment* __restrict__ fragmentBuffer, Light* __restrict__ light, unsigned int lightNum, int w, int h);

__global__
void _copyImageToPBO(uchar4* __restrict__ pbo, const glm::vec3* __restrict__ image, int w, int h, int beginW, int beginH, int bufferW, int bufferH);

__global__
void _copyTexToPBO(uchar4* __restrict__ pbo, Tex tex, int w, int h, int beginW, int beginH, int bufferW, int bufferH);

__global__
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize);

__global__
void _nodeMatrixTransform(
        int numVertices,
        glm::vec3* position,
        glm::vec3* normal,
        glm::mat4 MV, glm::mat3 MV_normal);

glm::mat4 _getMatrixFromNodeMatrixVector(const tinygltf::Node & n);

void _traverseNode (
        std::map<std::string, glm::mat4> & n2m,
        const tinygltf::Scene & scene,
        const std::string & nodeString,
        const glm::mat4 & parentMatrix);

void _initTex(const tinygltf::Scene & scene, const tinygltf::Material &mat, const std::string &keyword, Tex &texData);