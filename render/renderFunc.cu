
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include "util/checkCUDAError.h"
#include "util/tiny_gltf_loader.h"
#include "external/include/glm/gtc/quaternion.hpp"
#include "external/include/glm/gtx/transform.hpp"

#include "shader.h"
#include "dataType.h"
#include "renderTool.h"
#include "renderFunc.h"


////////////////////////////////////////////////////////////////
///                      Render Pipeline                     ///
////////////////////////////////////////////////////////////////

__global__
void _vertexTransform(
        int numVertices,
        PrimitiveDevBufPointers primitive,
        glm::mat4 M, glm::mat4 V, glm::mat4 P,
        int width, int height) {

    // vertex id
    int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (vid < numVertices) {

        VertexIn in = {primitive.dev_position[vid],
                       primitive.dev_normal[vid]};
        VertexOut &out = primitive.dev_verticesOut[vid];

        out = vertexShader(in, M, V, P);
        out.windowPos = {(out.clipPos.x + 1.0f) * 0.5f * width,
                         (1.0f - out.clipPos.y) * 0.5f * height,
                         out.clipPos.z};
    }
}


__global__
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

    // index id
    int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (iid < numIndices) {

        // TODO: uncomment the following code for a start
        // This is primitive assembly for triangles

        int pid;	// id for cur primitives vector
        int vid;
        if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
            pid = iid / (int)primitive.primitiveType;
            vid = iid % (int)primitive.primitiveType;
            dev_primitives[pid + curPrimitiveBeginId].v[vid]
                    = primitive.dev_verticesOut[primitive.dev_indices[iid]];
            dev_primitives[pid + curPrimitiveBeginId].v[vid].color = {vid==0,vid==1,vid==2};
            dev_primitives[pid + curPrimitiveBeginId].primitiveType = primitive.primitiveType;
        }


        // TODO: other primitive types (point, line)
    }

}

__global__
void _generateTileBuffer(int numPrimitives, Primitive* dev_primitives, Tile* dev_tileBuffer, int width, int height, int tileSize)
{
    const unsigned int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if(pid>=numPrimitives) return;

    const Primitive & primitive = dev_primitives[pid];
    const glm::vec3 (&pos)[] = {primitive.v[0].windowPos,
                                primitive.v[1].windowPos,
                                primitive.v[2].windowPos};

    // Back Face Culling
    glm::vec3 nor = glm::cross(pos[0]-pos[1],pos[1]-pos[2]);
    if(nor.z > 0) return;

    const int maxTileNumX = (width + tileSize - 1)/tileSize;
    const int maxTileNumY = (height + tileSize - 1)/tileSize;

    const AABB bound = getAABBForTriangle(pos);
    const int minTileIdX = glm::max(((int)glm::round(bound.min.x))/tileSize,0);
    const int maxTileIdX = glm::min(((int)glm::round(bound.max.x))/tileSize,maxTileNumX-1);
    const int minTileIdY = glm::max(((int)glm::round(bound.min.y))/tileSize,0);
    const int maxTileIdY = glm::min(((int)glm::round(bound.max.y))/tileSize,maxTileNumY-1);

    for(int x=minTileIdX;x<=maxTileIdX;x++)
    {
        for(int y=minTileIdY;y<=maxTileIdY;y++)
        {
            const unsigned int tid = x + y * maxTileNumX;
            const unsigned int id = atomicAdd(&(dev_tileBuffer[tid].numPrimitives),1);
            if(id<maxPrimitivesPerTile)
                dev_tileBuffer[tid].primitiveId[id] = pid;
        }
    }

}

#define INTERPOLATE(out, in, coef, attri) {out.attri = in[0].attri * coef.x + in[1].attri * coef.y + in[2].attri * coef.z;}

__device__
Fragment _generateFragment(const glm::vec3 &barycentricCoord, const Primitive &primitive)
{
    glm::vec3 triViewZ = {primitive.v[0].viewPos.z,
                          primitive.v[1].viewPos.z,
                          primitive.v[2].viewPos.z};
    glm::vec3 coef = getInterpolationCoef(barycentricCoord, triViewZ);
    Fragment buffer;

    // Implement Interpolation Here
    // Need to modify if the struct of Fragment changes
    INTERPOLATE(buffer, primitive.v, coef, color);
    INTERPOLATE(buffer, primitive.v, coef, objectPos);
    INTERPOLATE(buffer, primitive.v, coef, worldPos);
    INTERPOLATE(buffer, primitive.v, coef, viewPos);
    INTERPOLATE(buffer, primitive.v, coef, clipPos);
    INTERPOLATE(buffer, primitive.v, coef, windowPos);
    INTERPOLATE(buffer, primitive.v, coef, objectNor);
    INTERPOLATE(buffer, primitive.v, coef, worldNor);
    INTERPOLATE(buffer, primitive.v, coef, viewNor);

    return buffer;
}

__global__
void _rasterize(Primitive* dev_primitives, Tile* dev_tileBuffer, Fragment* dev_fragmentBuffer, int width, int height, int tileSize)
{
    extern __shared__ Fragment tileFragment[];

    const int maxTileNumX = (width + tileSize - 1)/tileSize;
    const int maxTileNumY = (height + tileSize - 1)/tileSize;
    const int & tileIdX = blockIdx.x;
    const int & tileIdY = blockIdx.y;
    const int tileId = tileIdX + tileIdY * maxTileNumX;
    const int & tilePosX = threadIdx.x;
    const int & tilePosY = threadIdx.y;
    const int tilePos = tilePosX + tilePosY * tileSize;
    const int posX = tileIdX * tileSize + tilePosX;
    const int posY = tileIdY * tileSize + tilePosY;
    const int pos = posX + posY * width;

    if(tileIdX>=maxTileNumX||tileIdY>=maxTileNumY) return;
    if(posX>=width||posY>=height) return;


    // Init Fragment Here
    tileFragment[tilePos].depth = 1;
    tileFragment[tilePos].color = {0,0,0};
    tileFragment[tilePos].material = InValid;


    int maxPrimitiveIdIndex = glm::min(dev_tileBuffer[tileId].numPrimitives,maxPrimitivesPerTile);
    for(int primitiveIdIndex=0;primitiveIdIndex<maxPrimitiveIdIndex;primitiveIdIndex++)
    {
        const Primitive & primitive = dev_primitives[dev_tileBuffer[tileId].primitiveId[primitiveIdIndex]];
        const glm::vec3 (&primitivePos)[] = {primitive.v[0].windowPos,
                                             primitive.v[1].windowPos,
                                             primitive.v[2].windowPos};
        glm::vec3 baryCoords = calculateBarycentricCoordinate(primitivePos, glm::vec2(posX, posY));
        bool isInsideTriangle = isBarycentricCoordInBounds(baryCoords);
        if (isInsideTriangle)
        {
            float depth = -getZAtCoordinate(baryCoords, primitivePos);
            if(depth<tileFragment[tilePos].depth)
            {
                tileFragment[tilePos] = _generateFragment(baryCoords, primitive);
                tileFragment[tilePos].depth = depth;
                tileFragment[tilePos].material = Direct;
            } // No need to use atomic because no data race happen
        }
        __syncthreads(); // Ensure all threads are rasterizing the same primitive
    }

    // Copy data from shared memory to global memory
    // one thread per fragment
    dev_fragmentBuffer[pos] = tileFragment[tilePos];
}

/**
* Writes fragment colors to the framebuffer
*/
__global__
void _fragmentShading(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        framebuffer[index] = fragmentShader(fragmentBuffer[index]);
    }
}

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__
void _copyImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

////////////////////////////////////////////////////////////////
/// Functions that only be called when program start or exit ///
////////////////////////////////////////////////////////////////


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {

    // Attribute (vec3 position)
    // component (3 * float)
    // byte (4 * byte)

    // id of component
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < N) {
        int count = i / n;
        int offset = i - count * n;	// which component of the attribute

        for (int j = 0; j < componentTypeByteSize; j++) {

            dev_dst[count * componentTypeByteSize * n
                    + offset * componentTypeByteSize
                    + j]

                    =

                    dev_src[byteOffset
                            + count * (byteStride == 0 ? componentTypeByteSize * n : byteStride)
                            + offset * componentTypeByteSize
                            + j];
        }
    }
}

__global__
void _nodeMatrixTransform(
        int numVertices,
        VertexAttributePosition* position,
        VertexAttributeNormal* normal,
        glm::mat4 MV, glm::mat3 MV_normal) {

    // vertex id
    int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (vid < numVertices) {
        position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
        normal[vid] = glm::normalize(MV_normal * normal[vid]);
    }
}

glm::mat4 _getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {

    glm::mat4 curMatrix(1.0);

    const std::vector<double> &m = n.matrix;
    if (m.size() > 0) {
        // matrix, copy it

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                curMatrix[i][j] = (float)m.at(4 * i + j);
            }
        }
    } else {
        // no matrix, use rotation, scale, translation

        if (!n.translation.empty()) {
            curMatrix[3][0] = n.translation[0];
            curMatrix[3][1] = n.translation[1];
            curMatrix[3][2] = n.translation[2];
        }

        if (!n.rotation.empty()) {
            glm::mat4 R;
            glm::quat q;
            q[0] = n.rotation[0];
            q[1] = n.rotation[1];
            q[2] = n.rotation[2];

            R = glm::mat4_cast(q);
            curMatrix = curMatrix * R;
        }

        if (n.scale.size() > 0) {
            curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
        }
    }

    return curMatrix;
}

void _traverseNode (
        std::map<std::string, glm::mat4> & n2m,
        const tinygltf::Scene & scene,
        const std::string & nodeString,
        const glm::mat4 & parentMatrix
)
{
    const tinygltf::Node & n = scene.nodes.at(nodeString);
    glm::mat4 M = parentMatrix * _getMatrixFromNodeMatrixVector(n);
    n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

    auto it = n.children.begin();
    auto itEnd = n.children.end();

    for (; it != itEnd; ++it) {
        _traverseNode(n2m, scene, *it, M);
    }
}
