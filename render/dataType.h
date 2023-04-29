#pragma once

#include "external/include/glm/glm.hpp"

static const unsigned int tileSize = 8;
static const unsigned int maxPrimitivesPerTile = 1024;
static const unsigned int defaultThreadPerBlock = 128;
static const unsigned int maxTexNum = 5;

typedef unsigned short VertexIndex;
typedef unsigned char TextureData;
typedef unsigned char BufferByte;

enum PrimitiveType{
    Point = 1,
    Line = 2,
    Triangle = 3
};

enum MaterialType{
    Invalid,
    Depth,
    Debug,
    UV,
    Tex0,
    Lambert
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
    // The attributes listed below might be useful,
    // but always feel free to modify on your own

    // glm::vec3 eyePos;	// eye space position used for shading
    // glm::vec3 eyeNor;
    // VertexAttributeTexcoord texcoord0;
    // TextureData* dev_diffuseTex;
    // ...
};

struct Tile {
    unsigned int numPrimitives;
    unsigned int primitiveId[maxPrimitivesPerTile];
};

struct SceneInfo {
    unsigned int numPrimitives;
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

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};
