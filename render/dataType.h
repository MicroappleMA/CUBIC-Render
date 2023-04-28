#pragma once

#include "external/include/glm/glm.hpp"

static const unsigned int tileSize = 8;
static const unsigned int maxPrimitivesPerTile = 1024;
static const unsigned int defaultThreadPerBlock = 128;


typedef unsigned short VertexIndex;
typedef glm::vec3 VertexAttributePosition;
typedef glm::vec3 VertexAttributeNormal;
typedef glm::vec2 VertexAttributeTexcoord;
typedef unsigned char TextureData;

typedef unsigned char BufferByte;

enum PrimitiveType{
    Point = 1,
    Line = 2,
    Triangle = 3
};

enum MaterialType{
    InValid = 0,
    Debug = 1,
    Direct = 2,
    Lambert = 3,
};

struct VertexIn {
    glm::vec3 pos;
    glm::vec3 nor;
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


    glm::vec2 texcoord0;
    TextureData* dev_diffuseTex = nullptr;
    // int texWidth, texHeight;
    // ...
};

struct Primitive {
    PrimitiveType primitiveType = Triangle;	// C++ 11 init
    VertexOut v[3];
};

struct Fragment {
    float depth;
    MaterialType material;

    glm::vec3 color;
    glm::vec4 objectPos;
    glm::vec4 worldPos;
    glm::vec4 viewPos;
    glm::vec4 clipPos;
    glm::vec3 windowPos;
    glm::vec3 objectNor;
    glm::vec3 worldNor;
    glm::vec3 viewNor;

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

struct PrimitiveDevBufPointers {
    int primitiveMode;	//from tinygltfloader macro
    PrimitiveType primitiveType;
    int numPrimitives;
    int numIndices;
    int numVertices;

    // Vertex In, const after loaded
    VertexIndex* dev_indices;
    VertexAttributePosition* dev_position;
    VertexAttributeNormal* dev_normal;
    VertexAttributeTexcoord* dev_texcoord0;

    // Materials, add more attributes when needed
    TextureData* dev_diffuseTex;
    int diffuseTexWidth;
    int diffuseTexHeight;
    // TextureData* dev_specularTex;
    // TextureData* dev_normalTex;
    // ...

    // Vertex Out, vertex used for rasterization, this is changing every frame
    VertexOut* dev_verticesOut;

    // TODO: add more attributes when needed
};

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};
