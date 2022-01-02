#include <RayTracer/host/Camera.h>
#include <RayTracer/host/BruteForce.h>
#include <RayTracer/host/BVH.h>
#include <RayTracer/host/KDTree.h>
#include <RayTracer/host/PPMTarget.h>
#include <RayTracer/host/Scene.h>

#include <RayTracer/device/Camera.cuh>
#include <RayTracer/device/PPMTarget.cuh>
#include <RayTracer/device/BruteForce.cuh>
#include <RayTracer/device/BVH.cuh>
#include <RayTracer/device/KDTree.cuh>
#include <RayTracer/device/Scene.cuh>

#define TINYOBJLOADER_IMPLEMENTATION 
#include "../RayTracer/tiny_obj_loader.h"

#pragma warning(push, 0)
#include <nlohmann/json.hpp>
#pragma warning(pop)

#include <iostream>
#include <string>

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
        {"time_elapsed", result.m_time},
        {"rays_created", result.m_rayCreations},
        {"volumes_tested", result.m_volumeTests},
        {"triangles_tested", result.m_triangleTests},
        {"triangles_intersections", result.m_triangleIntersections}
    };
}

}

struct RayGeneratorConfig {
    std::string name;
    glm::vec3 position;
    glm::vec3 lookAt;
    float fov;
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

void from_json(const nlohmann::json& json, Config& config) {
    json.at("scene").get_to(config.scene);
    json.at("output_metrics").get_to(config.metricsOutput);
    json.at("output_file").get_to(config.outputFile);
    json.at("samples_per_pixel").get_to(config.samplesPerPixel);
    json.at("max_depth").get_to(config.maxDepth);
    json.at("cuda").get_to(config.cuda);
    json.at("miss_color").get_to(config.missColor);

    const auto& rayGenerator = json.at("ray_generator");
    rayGenerator.at("name").get_to(config.rayGenerator.name);
    rayGenerator.at("position").get_to(config.rayGenerator.position);
    rayGenerator.at("look_at").get_to(config.rayGenerator.lookAt);
    rayGenerator.at("fov").get_to(config.rayGenerator.fov);

    const auto& accelerationStructure = json.at("acceleration_structure");
    accelerationStructure.at("name").get_to(config.accelerationStructure.name);
    accelerationStructure.at("depth").get_to(config.accelerationStructure.depth);

    const auto& renderTarget = json.at("render_target");
    renderTarget.at("name").get_to(config.renderTarget.name);
    renderTarget.at("width").get_to(config.renderTarget.width);
    renderTarget.at("width").get_to(config.renderTarget.height);
}

void HostMain(const Config& config) {
    std::unique_ptr<rt::IRayGenerator> rayGenerator;
    if(config.rayGenerator.name == "Camera") {
        rayGenerator = std::make_unique<rt::Camera>(
            config.rayGenerator.position,
            config.rayGenerator.lookAt,
            rt::Vector3(0.0f, 1.0f, 0.0f),      // up
            config.rayGenerator.fov,
            16.0f / 9.0f,                       // aspect ratio
            0.1f,                               // aperture
            10.0f                               // focus_distance
        );
    }

    std::unique_ptr<rt::IAccelerationStructure> accelerationStructure;
    if(config.accelerationStructure.name == "BruteForce") {
        accelerationStructure = std::make_unique<rt::BruteForce>();
    } else if(config.accelerationStructure.name == "BVH") {
        accelerationStructure = std::make_unique<rt::BVH>();
    } else if(config.accelerationStructure.name == "KDTree") {
        accelerationStructure = std::make_unique<rt::KDTree>(config.accelerationStructure.depth);
    }

    std::unique_ptr<rt::IRenderTarget> renderTarget;
    if(config.renderTarget.name == "PPMTarget") {
        renderTarget = std::make_unique<rt::PPMTarget>(
            config.renderTarget.width,
            config.renderTarget.height
        );
    }

    rt::Scene scene(
        rayGenerator.get(), 
        accelerationStructure.get(), 
        renderTarget.get()
    );

    scene.LoadScene(config.scene);
    auto result = scene.GenerateFrame(config.samplesPerPixel, config.maxDepth, config.missColor);

    nlohmann::json jsonResult = result;
    std::ofstream outputStream(config.metricsOutput);
    outputStream << std::setw(4) << jsonResult << std::endl;
}

void DeviceMain(const Config& config) {
    std::unique_ptr<rt::device::IRayGenerator> rayGenerator;
    if(config.rayGenerator.name == "Camera") {
        rayGenerator = std::make_unique<rt::device::Camera>(
            config.rayGenerator.position,
            config.rayGenerator.lookAt,
            rt::Vector3(0.0f, 1.0f, 0.0f),      // up
            config.rayGenerator.fov,
            16.0f / 9.0f,                       // aspect ratio
            0.1f,                               // aperture
            10.0f                               // focus_distance
        );
    }

    std::unique_ptr<rt::device::IAccelerationStructure> accelerationStructure;
    if(config.accelerationStructure.name == "BruteForce") {
        accelerationStructure = std::make_unique<rt::device::BruteForce>();
    } else if(config.accelerationStructure.name == "BVH") {
        accelerationStructure = std::make_unique<rt::device::BVH>();
    } else if(config.accelerationStructure.name == "KDTree") {
        accelerationStructure = std::make_unique<rt::device::KDTree>(config.accelerationStructure.depth);
    }

    std::unique_ptr<rt::device::IRenderTarget> renderTarget;
    if(config.renderTarget.name == "PPMTarget") {
        renderTarget = std::make_unique<rt::device::PPMTarget>(
            config.renderTarget.width,
            config.renderTarget.height
        );
    }

    rt::device::Scene scene(
        rayGenerator.get(),
        accelerationStructure.get(),
        renderTarget.get()
    );

    scene.LoadScene(config.scene);
    auto result = scene.GenerateFrame(config.samplesPerPixel, config.maxDepth, config.missColor, 8, 8);

    nlohmann::json jsonResult = result;
    std::ofstream outputStream(config.metricsOutput);
    outputStream << std::setw(4) << jsonResult << std::endl;
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        std::cerr << "No config file provided\n";
        return EXIT_FAILURE;
    }

    std::ifstream jsonFile(argv[1]);
    nlohmann::json configJson;
    jsonFile >> configJson;
    jsonFile.close();

    Config config{};
    try {
         config = configJson.get<Config>();
    } catch(const std::exception& error) {
        std::cerr << "Failed to read json config\n" << error.what();
        return EXIT_FAILURE;
    }
    
    std::ofstream out(config.outputFile);
    std::streambuf* coutBuffer = std::cout.rdbuf();
    std::cout.rdbuf(out.rdbuf());

    if(!config.cuda) {
        HostMain(config);
    } else {
        DeviceMain(config);
    }

    return EXIT_SUCCESS;
}
