
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include "util/checkCUDAError.h"
#include "util/tiny_gltf.h"
#include "external/include/glm/gtc/quaternion.hpp"
#include "external/include/glm/gtx/transform.hpp"

#include "dataType.h"
#include "renderTool.h"
#include "shader.h"
#include "renderFunc.h"


////////////////////////////////////////////////////////////////
///                      Render Pipeline                     ///
////////////////////////////////////////////////////////////////

__global__
void _vertexTransform(
        int numVertices,
        PrimitiveBuffer primitive,
        glm::mat4 M, glm::mat4 V, glm::mat4 P,
        int width, int height) {

    // vertex id
    int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (vid < numVertices) {

        // Vertex Assembly
        VertexIn in = {primitive.dev_position[vid],
                       primitive.dev_normal[vid],
                       primitive.dev_uv[vid],
                       primitive.materialType,
                       primitive.dev_tex
                       };

        VertexOut &out = primitive.dev_verticesOut[vid];

        out = vertexShader(in, M, V, P);

        out.windowPos = {(out.clipPos.x + 1.0f) * 0.5f * width,
                         (1.0f - out.clipPos.y) * 0.5f * height,
                         out.clipPos.z};

        for(int i=0; i<maxTexNum; i++)
            out.tex[i] = in.tex[i];
        out.material = in.material;
    }
}


__global__
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveBuffer primitive) {

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

#define INTERPOLATE(frag, vert, coef, attri) {frag.in.attri = vert[0].attri * coef.x + vert[1].attri * coef.y + vert[2].attri * coef.z;}
#define COPY(frag, vert, attri) {frag.in.attri = vert[0].attri;}

__device__
Fragment _generateFragment(const glm::vec3 &barycentricCoord, const Primitive &primitive)
{
    glm::vec3 triViewZ = {primitive.v[0].viewPos.z,
                          primitive.v[1].viewPos.z,
                          primitive.v[2].viewPos.z};
    glm::vec3 coef = getInterpolationCoef(barycentricCoord, triViewZ);
    Fragment frag;

    // Implement Interpolation Here
    // Need to modify if the struct of Fragment changes
    INTERPOLATE(frag, primitive.v, coef, color);
    INTERPOLATE(frag, primitive.v, coef, objectPos);
    INTERPOLATE(frag, primitive.v, coef, worldPos);
    INTERPOLATE(frag, primitive.v, coef, viewPos);
    INTERPOLATE(frag, primitive.v, coef, clipPos);
    INTERPOLATE(frag, primitive.v, coef, windowPos);
    INTERPOLATE(frag, primitive.v, coef, objectNor);
    INTERPOLATE(frag, primitive.v, coef, worldNor);
    INTERPOLATE(frag, primitive.v, coef, viewNor);
    COPY(frag, primitive.v, material);
    for(int i=0; i < maxTexNum; i++)
    {
        COPY(frag, primitive.v, tex[i].data);
        COPY(frag, primitive.v, tex[i].width);
        COPY(frag, primitive.v, tex[i].height);
        INTERPOLATE(frag, primitive.v, coef, uv);
    }

    return frag;
}

__device__
void _initFragment(Fragment &frag)
{
    // Init Fragment Here
    frag.depth = 1;
    frag.color = {0,0,0};
    frag.in.material = Invalid;
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

    _initFragment(tileFragment[tilePos]);

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
void _fragmentShading(glm::vec3 *framebuffer, Fragment *fragmentBuffer, Light *light, unsigned int lightNum, MaterialType overrideMaterial, int w, int h) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        Fragment &frag = fragmentBuffer[index];
        if (overrideMaterial != Invalid && frag.in.material != Invalid)
            frag.in.material = overrideMaterial;
        framebuffer[index] = fragmentShader(frag, light, lightNum);
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
        glm::vec3* position,
        glm::vec3* normal,
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
        std::map<int, glm::mat4> & n2m,
        const tinygltf::Model & model,
        const int & nodeId,
        const glm::mat4 & parentMatrix
)
{
    const tinygltf::Node & n = model.nodes[nodeId];
    glm::mat4 M = parentMatrix * _getMatrixFromNodeMatrixVector(n);
    n2m.insert(std::pair<int, glm::mat4>(nodeId, M));

    auto it = n.children.begin();
    auto itEnd = n.children.end();

    for (; it != itEnd; ++it) {
        _traverseNode(n2m, model, *it, M);
    }
}

void _initTex(const tinygltf::Model & model, const tinygltf::Material &mat, const std::string &keyword, Tex &texData)
{
    if (mat.values.find(keyword) != mat.values.end()) {
        int texIndex = mat.values.at(keyword).TextureIndex();
        if (texIndex != -1) {
            const tinygltf::Texture &tex = model.textures[texIndex];
            int imageIndex = tex.source;
            if (imageIndex != -1) {
                const tinygltf::Image &image = model.images[imageIndex];

                size_t s = image.image.size() * sizeof(TextureData);
                cudaMalloc(&(texData.data), s);
                cudaMemcpy(texData.data, &image.image.at(0), s, cudaMemcpyHostToDevice);

                texData.width = image.width;
                texData.height = image.height;

                checkCUDAError("Set Texture Image data");
                printf("%s texture = %s\n", keyword.c_str(), image.name.c_str());
            }
        }
    }
}