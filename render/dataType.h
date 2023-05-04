/**
 * @file      dataType.h
 * @brief     Define constant and data struct.
 * @authors   Jiayi Chen, Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2023
 * @copyright Jiayi Chen, University of Pennsylvania
 */

#pragma once

#include "external/include/glm/glm.hpp"

static const unsigned int tileSize = 8;
static const unsigned int maxPrimitivesPerTile = 1024;
static const unsigned int defaultThreadPerBlock = 128;
static const unsigned int maxTexNum = 6;

typedef unsigned short VertexIndex;
typedef unsigned char TextureData;
typedef unsigned char BufferByte;

enum PrimitiveType{
    Point = 1,
    Line = 2,
    Triangle = 3
};

enum MaterialType{
    Invalid = 0,
    Depth = 1,
    Mesh = 2,
    UV = 3,
    Normal = 4,
    Tex0 = 5,
    PBR = 6
};

enum LightType{
    DirectionalLight,
    PointLight
};

struct Light{
    LightType type;
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 color;
    float intensity;
};

struct Tex
{
    TextureData *data;
    int width;
    int height;
};

struct VertexIn {
    glm::vec3 &position;
    glm::vec3 &normal;
    glm::vec2 &uv;
    MaterialType &material;
    Tex *tex;
};

struct VertexOut {
    glm::vec4 objectPos;
    glm::vec4 worldPos;
    glm::vec4 viewPos;
    glm::vec4 clipPos;
    glm::vec3 windowPos;
    glm::vec3 objectNor;
    glm::vec3 worldNor;
    glm::vec3 viewNor;
    glm::vec3 color;
    glm::vec3 tangent;
    // TODO: add new attributes to your VertexOut
    // The attributes listed below might be useful,
    // but always feel free to modify on your own

    glm::vec2 uv;
    Tex tex[maxTexNum];
    MaterialType material;
};

struct Primitive {
    PrimitiveType primitiveType = Triangle;	// C++ 11 init
    VertexOut v[3];
};

struct Fragment {
    glm::vec3 color;
    float depth;
    VertexOut in;

    // TODO: add new attributes to your Fragment
};

struct Tile {
    unsigned int numPrimitives;
    unsigned int primitiveId[maxPrimitivesPerTile];
};

struct PrimitiveBuffer {
    int primitiveMode;	//from tinygltfloader macro
    PrimitiveType primitiveType;
    MaterialType materialType;
    int numPrimitives;
    int numIndices;
    int numVertices;

    // Vertex In, const after loaded
    VertexIndex *dev_indices;
    glm::vec3 *dev_position;
    glm::vec3 *dev_normal;
    glm::vec2 *dev_uv;

    // Materials, add more attributes when needed
    Tex dev_tex[maxTexNum];

    // Vertex Out, vertex used for rasterization, this is changing every frame
    VertexOut *dev_verticesOut;

    // TODO: add more attributes when needed
};

struct SceneInfo {
    unsigned int numPrimitives;
    std::map<std::string, std::vector<PrimitiveBuffer>> mesh2PrimitivesMap;
    unsigned int numLights;
};

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};
