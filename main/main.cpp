/**
 * @file      main.cpp
 * @brief     Main file for CUDA rasterizer. Handles CUDA-GL interop for display.
 * @authors   Jiayi Chen, Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2023
 * @copyright Jiayi Chen, University of Pennsylvania
 */

#include "main.hpp"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char **argv) {
#ifdef DEBUG
    cout << "CUDA Rasterizer :: Debug Version\n";
#else
    cout << "CUDA Rasterizer :: Release Version\n";
#endif
    if (argc != 2) {
        cout << "Usage: [Config Path]. Press Enter to exit\n";
        getchar();
        return 0;
    }

    std::ifstream ifs(argv[1]);
    nlohmann::json config = nlohmann::json::parse(ifs);

    width = config["width"];
    height = config["height"];
    string modelPath = config["model"];

    cout<<"Width = "<<width<<", Height = "<<height<<"\nModel = "<<modelPath<<"\n";

    // Load light info from json
    vector<Light> light;
    for (auto &&l:config["lights"])
    {
        string typeString = l["type"];
        LightType type = DirectionalLight;
        if(typeString == "directional") type = DirectionalLight;
        else if(typeString == "point") type = PointLight;

        light.push_back({
            type,
            {l["position"]["x"],l["position"]["y"],l["position"]["z"]},
            glm::normalize(glm::vec3{l["direction"]["x"],l["direction"]["y"],l["direction"]["z"]}),
            {l["color"]["r"],l["color"]["g"],l["color"]["b"]},
            l["intensity"]
        });
    }


    // Load scene from disk
	tinygltf::Scene scene;
	tinygltf::TinyGLTFLoader loader;
	std::string err;
	std::string ext = getFilePathExtension(modelPath);

	bool ret = false;
	if (ext == "glb") {
		// assume binary glTF.
		ret = loader.LoadBinaryFromFile(&scene, &err, modelPath);
	} else {
		// assume ascii glTF.
		ret = loader.LoadASCIIFromFile(&scene, &err, modelPath);
	}

	if (!err.empty()) {
        cout<<"Err: "+err;
	}

	if (!ret) {
        cout<<"Failed to parse glTF\n";
        getchar();
		return -1;
	}


    frame = 0;
    seconds = time(NULL);
    fpstracker = 0;

    // Launch CUDA/GL
    if (init(scene, light)) {
        // GLFW main loop
        mainLoop();
    }

    return 0;
}

void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        runCuda();

        time_t seconds2 = time (NULL);

        if (seconds2 - seconds >= 1) {

            fps = fpstracker / (seconds2 - seconds);
            fpstracker = 0;
            seconds = seconds2;
        }

        string title = "CUDA Rasterizer | " + utilityCore::convertIntToString((int)fps) + " FPS";
        glfwSetWindowTitle(window, title.c_str());

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------
float scale = 1.0f;
float x_trans = 0.0f, y_trans = 0.0f, z_trans = 10.0f;
float x_angle = 0.0f, y_angle = (float)PI;
void runCuda() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
    dptr = NULL;

    glm::mat4 P = glm::frustum<float>(-scale * ((float)width) / ((float)height),
                                      scale * ((float)width / (float)height),
                                      -scale, scale, 1.0, 1000.0);

    glm::mat4 V = glm::rotate((float)PI, glm::vec3{0.0f, 1.0f, 0.0f});

    glm::mat4 M =
            glm::translate(glm::vec3(x_trans, y_trans, z_trans))
            * glm::rotate(x_angle, glm::vec3{1.0f, 0.0f, 0.0f})
            * glm::rotate(y_angle, glm::vec3{0.0f, 1.0f, 0.0f});

    glm::mat3 MV_normal = glm::transpose(glm::inverse(glm::mat3(V) * glm::mat3(M)));
    glm::mat4 MV = V * M;
    glm::mat4 MVP = P * MV;

    cudaGLMapBufferObject((void **)&dptr, pbo);
    Render::getInstance().render(dptr, M, V, P);
    cudaGLUnmapBufferObject(pbo);

    frame++;
    fpstracker++;
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

bool init(const tinygltf::Scene & scene, const vector<Light> & light) {
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        return false;
    }

    window = glfwCreateWindow(width, height, "", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();

    // Mouse Control Callbacks
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, mouseMotionCallback);
    glfwSetScrollCallback(window, mouseWheelCallback);

    {
        auto it(scene.scenes.begin());
        auto itEnd(scene.scenes.end());

        for (; it != itEnd; it++) {
            for (size_t i = 0; i < it->second.size(); i++) {
                std::cout << it->second[i]
                          << ((i != (it->second.size() - 1)) ? ", " : "");
            }
            std::cout << " ] \n";
        }
    }


    Render::getInstance().init(scene, light, width, height);

    GLuint passthroughProgram;
    passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void initPBO() {
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);
}

void initCuda() {
    // Use device with highest Gflops/s
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initTextures() {
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
                  GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
    GLfloat vertices[] = {
            -1.0f, -1.0f,
            1.0f, -1.0f,
            1.0f,  1.0f,
            -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
            1.0f, 1.0f,
            0.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}


GLuint initShader() {
    GLuint program = glslUtility::createDefaultProgram(attributeLocations, 2);
    GLint location;

    glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) {
        glUniform1i(location, 0);
    }

    return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda() {
    if (pbo) {
        deletePBO(&pbo);
    }
    if (displayImage) {
        deleteTexture(&displayImage);
    }
}

void deletePBO(GLuint *pbo) {
    if (pbo) {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint *tex) {
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void shut_down(int return_code) {
    Render::getInstance().free();
    cudaDeviceReset();
    exit(return_code);
}

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------

void errorCallback(int error, const char *description) {
    fputs(description, stderr);
}

// Index must in range[0,9]
#define BIND_MATERIAL_KEY(index) \
{if (key == GLFW_KEY_0 + (index)) {Render::getInstance().overrideMaterial = (MaterialType)(index); }}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
        BIND_MATERIAL_KEY(0);
        BIND_MATERIAL_KEY(1);
        BIND_MATERIAL_KEY(2);
        BIND_MATERIAL_KEY(3);
        BIND_MATERIAL_KEY(4);
        BIND_MATERIAL_KEY(5);
        BIND_MATERIAL_KEY(6);
    }
}

//----------------------------
//----- util -----------------
//----------------------------
static std::string getFilePathExtension(const std::string &FileName) {
    if (FileName.find_last_of('.') != std::string::npos)
        return FileName.substr(FileName.find_last_of('.') + 1);
    return "";
}



//-----------------------------
//---- Mouse control ----------
//-----------------------------

enum ControlState { NONE = 0, ROTATE, TRANSLATE };
ControlState mouseState = NONE;
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            mouseState = ROTATE;
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            mouseState = TRANSLATE;
        }

    }
    else if (action == GLFW_RELEASE)
    {
        mouseState = NONE;
    }
}

double lastx = (double)width / 2;
double lasty = (double)height / 2;
void mouseMotionCallback(GLFWwindow* window, double xpos, double ypos)
{
    const double s_r = 0.01;
    const double s_t = 0.01;

    double diffx = xpos - lastx;
    double diffy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    if (mouseState == ROTATE)
    {
        //rotate
        x_angle -= (float)s_r * diffy;
        y_angle += (float)s_r * diffx;
    }
    else if (mouseState == TRANSLATE)
    {
        //translate
        x_trans += (float)(s_t * diffx);
        y_trans += (float)(-s_t * diffy);
    }
}

void mouseWheelCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    const double sensitivity = 0.3;
    z_trans -= (float)(sensitivity * yoffset);
}
