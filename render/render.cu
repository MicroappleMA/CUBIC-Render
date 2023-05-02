/**
 * @file      render.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include "util/checkCUDAError.h"
#include "external/include/glm/gtc/quaternion.hpp"
#include "external/include/glm/gtx/transform.hpp"

#include "util/tiny_gltf.h"
#include "dataType.h"
#include "renderTool.h"
#include "renderFunc.h"
#include "render.h"


////////////////////////////////////////////////////////////////
///                      Render Pipeline                     ///
////////////////////////////////////////////////////////////////

void Render::render(uchar4 *pbo, const glm::mat4 & M, const glm::mat4 & V, const glm::mat4 & P) {
    dim3 blockSize2d(tileSize, tileSize);
    dim3 blockCount2d((width - 1) / tileSize + 1,(height - 1) / tileSize + 1);

    // Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

    // Vertex Process & primitive assembly
    {
        int curPrimitiveBeginId = 0; // change static to non-static
        dim3 numThreadsPerBlock(defaultThreadPerBlock);

        auto it = sceneInfo.mesh2PrimitivesMap.begin();
        auto itEnd = sceneInfo.mesh2PrimitivesMap.end();

        for (; it != itEnd; ++it) {
            auto p = (it->second).begin();	// each primitive
            auto pEnd = (it->second).end();
            for (; p != pEnd; ++p) {
                dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
                dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

                _vertexTransform<<<numBlocksForVertices, numThreadsPerBlock>>>
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

    // TODO: render
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
    _fragmentShading<<<blockCount2d, blockSize2d>>>(dev_framebuffer,
                                                    dev_fragmentBuffer,
                                                    dev_lights,
                                                    sceneInfo.numLights,
                                                    overrideMaterial,
                                                    width,
                                                    height);
    checkCUDAError("fragment shader");

    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    _copyImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
}



////////////////////////////////////////////////////////////////
/// Functions that only be called when program start or exit ///
////////////////////////////////////////////////////////////////


/**
 * Called once at the end of the program to free CUDA memory.
 */
void Render::free() {

    // deconstruct primitives attribute/indices device buffer

    auto it(sceneInfo.mesh2PrimitivesMap.begin());
    auto itEnd(sceneInfo.mesh2PrimitivesMap.end());
    for (; it != itEnd; ++it) {
        for (auto p = it->second.begin(); p != it->second.end(); ++p) {
            cudaFree(p->dev_indices);
            cudaFree(p->dev_position);
            cudaFree(p->dev_normal);
            cudaFree(p->dev_uv);
            for(int i=0; i<maxTexNum; i++)
            {
                cudaFree(p->dev_tex[i].data);
            }

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

    cudaFree(dev_lights);
    dev_lights = nullptr;

    checkCUDAError("render Free");
}

void Render::init(const tinygltf::Model & model, const std::vector<Light> &light, const int &w, const int &h) {

    // 0. Init some buffers that are not related to the model
    {
        width = w;
        height = h;

        cudaFree(dev_fragmentBuffer);
        cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
        cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
        cudaFree(dev_tileBuffer);
        cudaMalloc(&dev_tileBuffer,
                   ((width + tileSize - 1) / tileSize) * ((height + tileSize - 1) / tileSize) * sizeof(Tile));
        cudaMemset(dev_tileBuffer, 0,
                   ((width + tileSize - 1) / tileSize) * ((height + tileSize - 1) / tileSize) * sizeof(Tile));
        cudaFree(dev_framebuffer);
        cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
        cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));


        sceneInfo.numLights = light.size();
        cudaFree(dev_lights);
        cudaMalloc(&dev_lights,sceneInfo.numLights * sizeof(Light));
        for(int i=0; i<sceneInfo.numLights; i++)
        {
            cudaMemcpy(dev_lights+i, &(light[i]), sizeof(Light), cudaMemcpyHostToDevice);
        }

        checkCUDAError("init");
    }

    int totalNumPrimitives = 0; // change static to non-static

    std::vector<BufferByte*> bufferViewDevPointers;

    // 1. copy all `bufferViews` to device memory
    {
        auto it(model.bufferViews.begin());
        auto itEnd(model.bufferViews.end());

        for (; it != itEnd; it++) {
            const std::string key = it->name;
            const tinygltf::BufferView &bufferView = *it;
            if (bufferView.target == 0) {
                continue; // Unsupported bufferView.
            }

            const tinygltf::Buffer &buffer = model.buffers.at(bufferView.buffer);

            BufferByte* dev_bufferView;
            cudaMalloc(&dev_bufferView, bufferView.byteLength);
            cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

            checkCUDAError("Set BufferView Device Mem");

            bufferViewDevPointers.push_back(dev_bufferView);

        }
    }



    // 2. for each mesh:
    //		for each primitive:
    //			build device buffer of indices, materail, and each attributes
    //			and store these pointers in a map
    {
        std::map<int, glm::mat4> nodeId2Matrix;
        auto rootNodeList = model.scenes.at(model.defaultScene).nodes;

        {
            auto it = rootNodeList.begin();
            auto itEnd = rootNodeList.end();
            for (; it != itEnd; ++it) {
                _traverseNode(nodeId2Matrix, model, *it, glm::mat4(1.0f));
            }
        }


        // parse through node to access mesh

        auto itNode = nodeId2Matrix.begin();
        auto itEndNode = nodeId2Matrix.end();
        for (; itNode != itEndNode; ++itNode) {

            const tinygltf::Node & N = model.nodes.at(itNode->first);
            const glm::mat4 & matrix = itNode->second;
            const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));


            const tinygltf::Mesh & mesh = model.meshes[N.mesh];

            auto res = sceneInfo.mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveBuffer>>(mesh.name, std::vector<PrimitiveBuffer>()));
            std::vector<PrimitiveBuffer> & primitiveVector = (res.first)->second;

            // for each primitive
            for (size_t i = 0; i < mesh.primitives.size(); i++) {
                const tinygltf::Primitive &primitive = mesh.primitives[i];

                if (primitive.indices == 0)
                    return;

                // TODO: add new attributes for your PrimitiveBuffer when you add new attributes
                VertexIndex* dev_indices = nullptr;
                glm::vec3* dev_position = nullptr;
                glm::vec3* dev_normal = nullptr;
                glm::vec2* dev_texcoord0 = nullptr;

                // ----------Indices-------------

                const tinygltf::Accessor &indexAccessor = model.accessors.at(primitive.indices);
                const tinygltf::BufferView &bufferView = model.bufferViews.at(indexAccessor.bufferView);
                BufferByte* dev_bufferView = bufferViewDevPointers[indexAccessor.bufferView];

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
                        indexAccessor.ByteStride(bufferView),
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
                    const tinygltf::Accessor &accessor = model.accessors.at(it->second);
                    const tinygltf::BufferView &bufferView = model.bufferViews.at(accessor.bufferView);

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
                        componentTypeByteSize = sizeof(glm::vec3) / n;
                        dev_attribute = (BufferByte**)&dev_position;
                    }
                    else if (it->first.compare("NORMAL") == 0) {
                        componentTypeByteSize = sizeof(glm::vec3) / n;
                        dev_attribute = (BufferByte**)&dev_normal;
                    }
                    else if (it->first.compare("TEXCOORD_0") == 0) {
                        componentTypeByteSize = sizeof(glm::vec2) / n;
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
                            accessor.ByteStride(bufferView),
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
                MaterialType materialType = Invalid;

                Tex diffuseTex{nullptr,0,0};
                Tex specularTex{nullptr,0,0};
                Tex normalTex{nullptr,0,0};
                Tex roughnessTex{nullptr,0,0};
                Tex emissionTex{nullptr,0,0};
                Tex environmentTex{nullptr,0,0};


                if (primitive.material != -1) {
                    const tinygltf::Material &mat = model.materials[primitive.material];
                    printf("material.name = %s\n", mat.name.c_str());

                    _initTex(model, mat, "diffuse", diffuseTex);
                    _initTex(model, mat, "specular", specularTex);
                    _initTex(model, mat, "normal", normalTex);
                    _initTex(model, mat, "roughness", roughnessTex);
                    _initTex(model, mat, "emission", emissionTex);
                    _initTex(model, mat, "environment", environmentTex);
                }

                // Generate material info according to texture;
                if (diffuseTex.data && specularTex.data && normalTex.data && roughnessTex.data)
                    materialType = PBR;
                else if (diffuseTex.data)
                    materialType = Tex0;
                else
                    materialType = Mesh;


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
                primitiveVector.push_back(PrimitiveBuffer{
                        primitive.mode,
                        primitiveType,
                        materialType,
                        numPrimitives,
                        numIndices,
                        numVertices,

                        dev_indices,
                        dev_position,
                        dev_normal,

                        dev_texcoord0,
                        {diffuseTex,
                         specularTex,
                         normalTex,
                         roughnessTex,
                         emissionTex,
                         environmentTex},

                        dev_vertexOut	//VertexOut
                });

                totalNumPrimitives += numPrimitives;

            } // for each primitive


        } // for each node

    }


    // 3. Malloc for dev_primitives
    {
        sceneInfo.numPrimitives = totalNumPrimitives;
        cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
    }


    // Finally, cudaFree raw dev_bufferViews
    {

        auto it(bufferViewDevPointers.begin());
        auto itEnd(bufferViewDevPointers.end());

        //bufferViewDevPointers

        for (; it != itEnd; it++) {
            cudaFree(*it);
        }

        checkCUDAError("Free BufferView Device Mem");
    }

}

