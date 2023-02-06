#include <RayTracer/host/Camera.h>
#include <RayTracer/host/BruteForce.h>
#include <RayTracer/host/BVH.h>
#include <RayTracer/host/KDTree.h>
#include <RayTracer/host/PPMTarget.h>
#include <RayTracer/host/Scene.h>

#ifdef RT_CUDA_ENABLED
    #include <RayTracer/device/Camera.cuh>
    #include <RayTracer/device/PPMTarget.cuh>
    #include <RayTracer/device/BruteForce.cuh>
    #include <RayTracer/device/BVH.cuh>
    #include <RayTracer/device/KDTree.cuh>
    #include <RayTracer/device/Scene.cuh>
#endif

#define TINYOBJLOADER_IMPLEMENTATION 
#include "../RayTracer/tiny_obj_loader.h"

#pragma warning(push, 0)
#include <nlohmann/json.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#pragma warning(pop)

#include <iostream>
#include <string>
#include <assert.h>

//-----------------------------------------------------------------------------------
// Config
struct RayGeneratorConfig {
    std::string name;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 up;
    float fov;
    float aspectRatio;
    float aperture;
    float focusDistance;
};

struct RenderTargetConfig {
    std::string name;
    unsigned int width;
    unsigned int height;
};

struct AccelerationStructureConfig {
    std::string name;
    int depth;
};

struct Config {
    std::string scene;
    std::string metricsOutput;
    std::string outputFile;
    unsigned int samplesPerPixel;
    unsigned int maxDepth;
    bool cuda;
    rt::Color missColor;
    RayGeneratorConfig rayGenerator;
    AccelerationStructureConfig accelerationStructure;
    RenderTargetConfig renderTarget;
};

//-----------------------------------------------------------------------------------
// Json
namespace glm {

void from_json(const nlohmann::json& j, glm::vec3& v) {
    j.at("x").get_to(v.x);
    j.at("y").get_to(v.y);
    j.at("z").get_to(v.z);
}

}

namespace rt {

void to_json(nlohmann::json& j, const rt::Metrics::Result& result) {
    j = nlohmann::json{
        {"space_partitioning_time", result.m_spacePartitioningTime},
        {"frame_time", result.m_frameTime},
        {"save_buffer_time", result.m_saveBufferTime},
        {"rays_created", result.m_rayCreations},
        {"volumes_tested", result.m_volumeTests},
        {"triangles_tested", result.m_triangleTests},
        {"triangles_intersections", result.m_triangleIntersections}
    };
}

}

std::ostream& operator<<(std::ostream& os, const glm::vec3 vec) {
    os << vec.x << ", " << vec.y << ", " << vec.z;
    return os;
}

template<class T>
T OptionalJsonValue(const nlohmann::json& json, const std::string& key, T fallback) {
    if(json.find(key) == json.end() ) {
        std::cerr<< "[Warning] Property \"" << key << "\" not found, used fallback \"" << fallback << "\" instead.\n";
    }

    return json.value(key, fallback);
}

void from_json(const nlohmann::json& json, Config& config) {
    json.at("scene").get_to(config.scene);
    config.metricsOutput = OptionalJsonValue(json, "output_metrics", std::string{});
    json.at("output_file").get_to(config.outputFile);
    json.at("samples_per_pixel").get_to(config.samplesPerPixel);
    json.at("max_depth").get_to(config.maxDepth);
    config.maxDepth = OptionalJsonValue(json, "cuda", false);
    json.at("miss_color").get_to(config.missColor);

    const auto& rayGenerator = json.at("ray_generator");
    rayGenerator.at("name").get_to(config.rayGenerator.name);
    rayGenerator.at("position").get_to(config.rayGenerator.position);
    rayGenerator.at("look_at").get_to(config.rayGenerator.lookAt);
    config.rayGenerator.up = OptionalJsonValue(rayGenerator, "up", glm::vec3{ 1.0f, 0.0f, 0.0f} );
    rayGenerator.at("fov").get_to(config.rayGenerator.fov);
    config.rayGenerator.aspectRatio = OptionalJsonValue(rayGenerator, "aspect_ratio", 16.0f / 9.0f );
    config.rayGenerator.aperture = OptionalJsonValue(rayGenerator, "asperture", 0.1f);
    config.rayGenerator.focusDistance = OptionalJsonValue(rayGenerator, "focus_distance", 10.0f);

    const auto& accelerationStructure = json.at("acceleration_structure");
    accelerationStructure.at("name").get_to(config.accelerationStructure.name);
    accelerationStructure.at("depth").get_to(config.accelerationStructure.depth);

    const auto& renderTarget = json.at("render_target");
    renderTarget.at("name").get_to(config.renderTarget.name);
    renderTarget.at("width").get_to(config.renderTarget.width);
    renderTarget.at("height").get_to(config.renderTarget.height);
}

//-----------------------------------------------------------------------------------
//
void HostMain(const Config& config) {
    std::unique_ptr<rt::IRayGenerator> rayGenerator;
    if(config.rayGenerator.name == "Camera") {
        rayGenerator = std::make_unique<rt::Camera>(
            config.rayGenerator.position,
            config.rayGenerator.lookAt,
            config.rayGenerator.up,
            config.rayGenerator.fov,
            config.rayGenerator.aspectRatio,
            config.rayGenerator.aperture,
            config.rayGenerator.focusDistance
        );
    }

    std::unique_ptr<rt::IAccelerationStructure> accelerationStructure;
    if(config.accelerationStructure.name == "BruteForce") {
        accelerationStructure = std::make_unique<rt::BruteForce>();
    } else if(config.accelerationStructure.name == "BVH") {
        accelerationStructure = std::make_unique<rt::BVH>();
    } else if(config.accelerationStructure.name == "KDTree") {
        accelerationStructure = std::make_unique<rt::KDTree>(config.accelerationStructure.depth);
    } else {
        std::cerr << "[Error] No valid acceleration structure provided\n";
        return;
    }

    std::unique_ptr<rt::IRenderTarget> renderTarget;
    if(config.renderTarget.name == "PPMTarget") {
        renderTarget = std::make_unique<rt::PPMTarget>(
            config.renderTarget.width,
            config.renderTarget.height
        );
    } else {
        std::cerr << "[Error] No valid redner target provided\n";
        return;
    }

    rt::Scene scene(
        rayGenerator.get(), 
        accelerationStructure.get(), 
        renderTarget.get()
    );

    scene.LoadScene(config.scene);
    auto result = scene.GenerateFrame(config.samplesPerPixel, config.maxDepth, config.missColor);

    if(!config.metricsOutput.empty()) {
        std::ofstream outputStream(config.metricsOutput);
        outputStream << std::setw(4) << nlohmann::json{result} << std::endl;
    }
}

#ifdef RT_CUDA_ENABLED
void DeviceMain(const Config& config) {
    std::unique_ptr<rt::device::IRayGenerator> rayGenerator;
    if(config.rayGenerator.name == "Camera") {
        rayGenerator = std::make_unique<rt::device::Camera>(
            config.rayGenerator.position,
            config.rayGenerator.lookAt,
            config.rayGenerator.up,
            config.rayGenerator.fov,
            config.rayGenerator.aspectRatio,
            config.rayGenerator.aperture,
            config.rayGenerator.focusDistance
        );
    }

    std::unique_ptr<rt::device::IAccelerationStructure> accelerationStructure;
    if(config.accelerationStructure.name == "BruteForce") {
        accelerationStructure = std::make_unique<rt::device::BruteForce>();
    } else if(config.accelerationStructure.name == "BVH") {
        accelerationStructure = std::make_unique<rt::device::BVH>();
    } else if(config.accelerationStructure.name == "KDTree") {
        accelerationStructure = std::make_unique<rt::device::KDTree>(config.accelerationStructure.depth);
    } else {
        std::cerr << "[Error] No valid acceleration structure provided\n";
        return;
    }

    std::unique_ptr<rt::device::IRenderTarget> renderTarget;
    if(config.renderTarget.name == "PPMTarget") {
        renderTarget = std::make_unique<rt::device::PPMTarget>(
            config.renderTarget.width,
            config.renderTarget.height
        );
    } else {
        std::cerr << "[Error] No valid redner target provided\n";
        return;
    }

    rt::device::Scene scene(
        rayGenerator.get(),
        accelerationStructure.get(),
        renderTarget.get()
    );

    scene.LoadScene(config.scene);
    auto result = scene.GenerateFrame(config.samplesPerPixel, config.maxDepth, config.missColor, 8, 8);

    if(!config.metricsOutput.empty()) {
        std::ofstream outputStream(config.metricsOutput);
        outputStream << std::setw(4) << nlohmann::json{ result } << std::endl;
    }
}
#endif

int main(int argc, char* argv[]) {
    std::cerr << "RayTracer alpha\n";

    if(argc < 2) {
        std::cerr << "[Error] No config file provided\n";
        return EXIT_FAILURE;
    }

    std::ifstream jsonFile(argv[1]);
    nlohmann::json configJson;
    jsonFile >> configJson;
    jsonFile.close();

    Config config{};
    try {
        std::cerr << "[Info] Reading config file: " << argv[1] << '\n';
        config = configJson.get<Config>();
    } catch(const std::exception& error) {
        std::cerr << "[Error] Failed to read json config\n" << error.what();
        return EXIT_FAILURE;
    }
    
    std::ofstream out(config.outputFile);
    std::streambuf* coutBuffer = std::cout.rdbuf();
    std::cout.rdbuf(out.rdbuf());

    if(!config.cuda) {
        HostMain(config);
    } else {
#ifdef RT_CUDA_ENABLED
        DeviceMain(config);
#else
        std::cerr << "CUDA config not supported\n";
#endif
    }

    return EXIT_SUCCESS;
}
