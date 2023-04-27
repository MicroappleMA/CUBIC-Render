/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "data.h"
#include "rasterizeTools.h"
#include "rasterize.h"
#include "shader.h"


////////////////////////////////////////////////////////////////
///                      Render Pipeline                     ///
////////////////////////////////////////////////////////////////

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
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

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        framebuffer[index] = fragmentShader(fragmentBuffer[index]);
    }
}

__global__ 
void _vertexTransformAndAssembly(
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
                tileFragment[tilePos].material = Lambert;
            } // No need to use atomic because no data race happen
        }
        __syncthreads(); // Ensure all threads are rasterizing the same primitive
    }

    // Copy data from shared memory to global memory
    // one thread per fragment
    dev_fragmentBuffer[pos] = tileFragment[tilePos];
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & M, const glm::mat4 & V, const glm::mat4 & P) {
    dim3 blockSize2d(tileSize, tileSize);
    dim3 blockCount2d((width - 1) / tileSize + 1,(height - 1) / tileSize + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
        int curPrimitiveBeginId = 0; // change static to non-static
		dim3 numThreadsPerBlock(defaultThreadPerBlock);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly<<<numBlocksForVertices, numThreadsPerBlock>>>
                (p->numVertices,
                 *p,
                 M, V, P,
                 width,
                 height);
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly<<<numBlocksForIndices, numThreadsPerBlock>>>
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}

    cudaMemset(dev_tileBuffer, 0, ((width + tileSize - 1) / tileSize) * ((height + tileSize - 1) / tileSize) * sizeof(Tile));
    checkCUDAError("_clearBuffer");

    // TODO: rasterize
    {
        dim3 numThreadsPerBlock = defaultThreadPerBlock;
        dim3 numBlocks = (sceneInfo.numPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x;
        _generateTileBuffer<<<numBlocks,numThreadsPerBlock>>>
                (sceneInfo.numPrimitives,
                 dev_primitives,
                 dev_tileBuffer,
                 width,
                 height,
                 tileSize);
        checkCUDAError("_generateTileBuffer");

        _rasterize<<<blockCount2d,blockSize2d,tileSize*tileSize*sizeof(Fragment)>>>
                (dev_primitives,
                 dev_tileBuffer,
                 dev_fragmentBuffer,
                 width,
                 height,
                 tileSize);
        checkCUDAError("_rasterize");

    }


    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

    // Copy depthbuffer colors into framebuffer
	render<<<blockCount2d, blockSize2d>>>(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");

    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
}



////////////////////////////////////////////////////////////////
/// Functions that only be called when program start or exit ///
////////////////////////////////////////////////////////////////


/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
    cudaFree(dev_fragmentBuffer);
    cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
    cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_tileBuffer);
    cudaMalloc(&dev_tileBuffer, ((width + tileSize - 1) / tileSize) * ((height + tileSize - 1) / tileSize) * sizeof(Tile));
    cudaMemset(dev_tileBuffer, 0, ((width + tileSize - 1) / tileSize) * ((height + tileSize - 1) / tileSize) * sizeof(Tile));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    checkCUDAError("rasterizeInit");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

    auto it(mesh2PrimitivesMap.begin());
    auto itEnd(mesh2PrimitivesMap.end());
    for (; it != itEnd; ++it) {
        for (auto p = it->second.begin(); p != it->second.end(); ++p) {
            cudaFree(p->dev_indices);
            cudaFree(p->dev_position);
            cudaFree(p->dev_normal);
            cudaFree(p->dev_texcoord0);
            cudaFree(p->dev_diffuseTex);

            cudaFree(p->dev_verticesOut);


            //TODO: release other attributes and materials
        }
    }

    ////////////

    cudaFree(dev_primitives);
    dev_primitives = nullptr;

    cudaFree(dev_fragmentBuffer);
    dev_fragmentBuffer = nullptr;

    cudaFree(dev_tileBuffer);
    dev_tileBuffer = nullptr;

    cudaFree(dev_framebuffer);
    dev_framebuffer = nullptr;

    checkCUDAError("rasterize Free");
}


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

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {

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

void traverseNode (
        std::map<std::string, glm::mat4> & n2m,
        const tinygltf::Scene & scene,
        const std::string & nodeString,
        const glm::mat4 & parentMatrix
)
{
    const tinygltf::Node & n = scene.nodes.at(nodeString);
    glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
    n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

    auto it = n.children.begin();
    auto itEnd = n.children.end();

    for (; it != itEnd; ++it) {
        traverseNode(n2m, scene, *it, M);
    }
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

    int totalNumPrimitives = 0; // change static to non-static

    std::map<std::string, BufferByte*> bufferViewDevPointers;

    // 1. copy all `bufferViews` to device memory
    {
        auto it(scene.bufferViews.begin());
        auto itEnd(scene.bufferViews.end());

        for (; it != itEnd; it++) {
            const std::string key = it->first;
            const tinygltf::BufferView &bufferView = it->second;
            if (bufferView.target == 0) {
                continue; // Unsupported bufferView.
            }

            const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

            BufferByte* dev_bufferView;
            cudaMalloc(&dev_bufferView, bufferView.byteLength);
            cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

            checkCUDAError("Set BufferView Device Mem");

            bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

        }
    }



    // 2. for each mesh:
    //		for each primitive:
    //			build device buffer of indices, materail, and each attributes
    //			and store these pointers in a map
    {
        std::map<std::string, glm::mat4> nodeString2Matrix;
        auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

        {
            auto it = rootNodeNamesList.begin();
            auto itEnd = rootNodeNamesList.end();
            for (; it != itEnd; ++it) {
                traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
            }
        }


        // parse through node to access mesh

        auto itNode = nodeString2Matrix.begin();
        auto itEndNode = nodeString2Matrix.end();
        for (; itNode != itEndNode; ++itNode) {

            const tinygltf::Node & N = scene.nodes.at(itNode->first);
            const glm::mat4 & matrix = itNode->second;
            const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

            auto itMeshName = N.meshes.begin();
            auto itEndMeshName = N.meshes.end();

            for (; itMeshName != itEndMeshName; ++itMeshName) {

                const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

                auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
                std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

                // for each primitive
                for (size_t i = 0; i < mesh.primitives.size(); i++) {
                    const tinygltf::Primitive &primitive = mesh.primitives[i];

                    if (primitive.indices.empty())
                        return;

                    // TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
                    VertexIndex* dev_indices = nullptr;
                    VertexAttributePosition* dev_position = nullptr;
                    VertexAttributeNormal* dev_normal = nullptr;
                    VertexAttributeTexcoord* dev_texcoord0 = nullptr;

                    // ----------Indices-------------

                    const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
                    const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
                    BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

                    // assume type is SCALAR for indices
                    int n = 1;
                    int numIndices = indexAccessor.count;
                    int componentTypeByteSize = sizeof(VertexIndex);
                    int byteLength = numIndices * n * componentTypeByteSize;

                    dim3 numThreadsPerBlock(defaultThreadPerBlock);
                    dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
                    cudaMalloc(&dev_indices, byteLength);
                    _deviceBufferCopy<<<numBlocks, numThreadsPerBlock>>>(
                            numIndices,
                            (BufferByte*)dev_indices,
                            dev_bufferView,
                            n,
                            indexAccessor.byteStride,
                            indexAccessor.byteOffset,
                            componentTypeByteSize);


                    checkCUDAError("Set Index Buffer");


                    // ---------Primitive Info-------

                    // Warning: LINE_STRIP is not supported in tinygltfloader
                    int numPrimitives;
                    PrimitiveType primitiveType;
                    switch (primitive.mode) {
                        case TINYGLTF_MODE_TRIANGLES:
                            primitiveType = PrimitiveType::Triangle;
                            numPrimitives = numIndices / 3;
                            break;
                        case TINYGLTF_MODE_TRIANGLE_STRIP:
                            primitiveType = PrimitiveType::Triangle;
                            numPrimitives = numIndices - 2;
                            break;
                        case TINYGLTF_MODE_TRIANGLE_FAN:
                            primitiveType = PrimitiveType::Triangle;
                            numPrimitives = numIndices - 2;
                            break;
                        case TINYGLTF_MODE_LINE:
                            primitiveType = PrimitiveType::Line;
                            numPrimitives = numIndices / 2;
                            break;
                        case TINYGLTF_MODE_LINE_LOOP:
                            primitiveType = PrimitiveType::Line;
                            numPrimitives = numIndices + 1;
                            break;
                        case TINYGLTF_MODE_POINTS:
                            primitiveType = PrimitiveType::Point;
                            numPrimitives = numIndices;
                            break;
                        default:
                            // output error
                            break;
                    }


                    // ----------Attributes-------------

                    auto it(primitive.attributes.begin());
                    auto itEnd(primitive.attributes.end());

                    int numVertices = 0;
                    // for each attribute
                    for (; it != itEnd; it++) {
                        const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
                        const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

                        int n = 1;
                        if (accessor.type == TINYGLTF_TYPE_SCALAR) {
                            n = 1;
                        }
                        else if (accessor.type == TINYGLTF_TYPE_VEC2) {
                            n = 2;
                        }
                        else if (accessor.type == TINYGLTF_TYPE_VEC3) {
                            n = 3;
                        }
                        else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                            n = 4;
                        }

                        BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
                        BufferByte ** dev_attribute = nullptr;

                        numVertices = accessor.count;
                        int componentTypeByteSize;

                        // Note: since the type of our attribute array (dev_position) is static (float32)
                        // We assume the glTF model attribute type are 5126(FLOAT) here

                        if (it->first.compare("POSITION") == 0) {
                            componentTypeByteSize = sizeof(VertexAttributePosition) / n;
                            dev_attribute = (BufferByte**)&dev_position;
                        }
                        else if (it->first.compare("NORMAL") == 0) {
                            componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
                            dev_attribute = (BufferByte**)&dev_normal;
                        }
                        else if (it->first.compare("TEXCOORD_0") == 0) {
                            componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
                            dev_attribute = (BufferByte**)&dev_texcoord0;
                        }

                        std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

                        dim3 numThreadsPerBlock(defaultThreadPerBlock);
                        dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
                        int byteLength = numVertices * n * componentTypeByteSize;
                        cudaMalloc(dev_attribute, byteLength);

                        _deviceBufferCopy<<<numBlocks, numThreadsPerBlock>>>(
                                n * numVertices,
                                *dev_attribute,
                                dev_bufferView,
                                n,
                                accessor.byteStride,
                                accessor.byteOffset,
                                componentTypeByteSize);

                        std::string msg = "Set Attribute Buffer: " + it->first;
                        checkCUDAError(msg.c_str());
                    }

                    // malloc for VertexOut
                    VertexOut* dev_vertexOut;
                    cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
                    checkCUDAError("Malloc VertexOut Buffer");

                    // ----------Materials-------------

                    // You can only worry about this part once you started to
                    // implement textures for your rasterizer
                    TextureData* dev_diffuseTex = nullptr;
                    int diffuseTexWidth = 0;
                    int diffuseTexHeight = 0;
                    if (!primitive.material.empty()) {
                        const tinygltf::Material &mat = scene.materials.at(primitive.material);
                        printf("material.name = %s\n", mat.name.c_str());

                        if (mat.values.find("diffuse") != mat.values.end()) {
                            std::string diffuseTexName = mat.values.at("diffuse").string_value;
                            if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
                                const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
                                if (scene.images.find(tex.source) != scene.images.end()) {
                                    const tinygltf::Image &image = scene.images.at(tex.source);

                                    size_t s = image.image.size() * sizeof(TextureData);
                                    cudaMalloc(&dev_diffuseTex, s);
                                    cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);

                                    diffuseTexWidth = image.width;
                                    diffuseTexHeight = image.height;

                                    checkCUDAError("Set Texture Image data");
                                }
                            }
                        }

                        // TODO: write your code for other materails
                        // You may have to take a look at tinygltfloader
                        // You can also use the above code loading diffuse material as a start point
                    }


                    // ---------Node hierarchy transform--------
                    cudaDeviceSynchronize();

                    dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
                    _nodeMatrixTransform<<<numBlocksNodeTransform, numThreadsPerBlock>>>(
                            numVertices,
                            dev_position,
                            dev_normal,
                            matrix,
                            matrixNormal);

                    checkCUDAError("Node hierarchy transformation");

                    // at the end of the for loop of primitive
                    // push dev pointers to map
                    primitiveVector.push_back(PrimitiveDevBufPointers{
                            primitive.mode,
                            primitiveType,
                            numPrimitives,
                            numIndices,
                            numVertices,

                            dev_indices,
                            dev_position,
                            dev_normal,
                            dev_texcoord0,

                            dev_diffuseTex,
                            diffuseTexWidth,
                            diffuseTexHeight,

                            dev_vertexOut	//VertexOut
                    });

                    totalNumPrimitives += numPrimitives;

                } // for each primitive

            } // for each mesh

        } // for each node

    }


    // 3. Malloc for dev_primitives
    {
        sceneInfo.numPrimitives = totalNumPrimitives;
        cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
    }


    // Finally, cudaFree raw dev_bufferViews
    {

        std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
        std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());

        //bufferViewDevPointers

        for (; it != itEnd; it++) {
            cudaFree(it->second);
        }

        checkCUDAError("Free BufferView Device Mem");
    }

}

