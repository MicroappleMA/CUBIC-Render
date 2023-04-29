#pragma once


#include "util/tiny_gltf_loader.h"
#include "dataType.h"


__global__
void _vertexTransform(int numVertices,
        PrimitiveDevBufPointers primitive,
        glm::mat4 M, glm::mat4 V, glm::mat4 P,
        int width, int height);

__global__
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive);

__global__
void _generateTileBuffer(int numPrimitives, Primitive* dev_primitives, Tile* dev_tileBuffer, int width, int height, int tileSize);

__global__
void _rasterize(Primitive* dev_primitives, Tile* dev_tileBuffer, Fragment* dev_fragmentBuffer, int width, int height, int tileSize);

__global__
void _fragmentShading(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer);

__global__
void _copyImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image);

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