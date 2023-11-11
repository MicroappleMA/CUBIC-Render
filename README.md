CUBIC Render
===============
**CU**DA-**b**ased **I**ndependent and **C**ustomizable Render

[![Build](https://github.com/MicroappleMA/CUBIC-Render/actions/workflows/build.yml/badge.svg)](https://github.com/MicroappleMA/CUBIC-Render/actions/workflows/build.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/c27c2583221b410289df5842eeee6923)](https://app.codacy.com/gh/MicroappleMA/CUBIC-Render/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![GitHub license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MicroappleMA/CUBIC-Render/blob/master/LICENSE)

![CUBIC Render Logo](./img/logo.png)

## Download

[Download](https://nightly.link/MicroappleMA/CUBIC-Render/workflows/build/master/CUBIC-Render.zip) latest build from CICD pipeline.

## Prerequisites

To build **CUBIC Render**, you must set up following environment.

- Windows 10/11
- Visual Studio 2019 (or more recent)
- CUDA Toolkit 12.0 (or more recent)
- CMake 3.24 (or more recent)

## Build

You may execute the **build.bat** or **build_*.bat**. This batch file will generate the projects, and build the **Release**, **RelWithDebInfo**, **Debug** config of **CUBIC Render** automatically. After successful build, you can find the Visual Studio project and binary file at the **build** directory.

## Run

You need to pass the path of configuration file as a parameter to CUBIC Render for running.

```shell
./CUBIC-Render.exe ./config/defaultConfig.json
```

## Feature

- **Deferred Shading**
- **Tiled Based Rendering**
- **Physical Based Shading**
- **Inverse Rendering**

## Structure
- `config`: example configuration files of CUBIC Render
- `external`: third party libraries that used by CUBIC Render
- `gltf`: gltf sample models from Khronos
- `img`: images that used by this readme file
- `main`: main function that mainly deal with user input and frame output
- `render`: render pipeline and shaders
- `util`: utility functions and head-only libraries

## Example

![Example 0](./img/example0.png)
![Example 1](./img/example1.png)

## Credits

- [Code Framework](https://github.com/CIS565-Fall-2018/Project4-CUDA-Rasterizer) by [University of Pennsylvania, CIS 565](https://cis565-fall-2022.github.io/)
- [glTF Sample Models](https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/1.0) by  [Khronos Group](https://github.com/KhronosGroup)
- [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
- [json](https://github.com/nlohmann/json) by [@nlohmann](https://github.com/nlohmann)
