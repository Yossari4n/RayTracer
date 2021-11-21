#include <RayTracer/host/Camera.h>
#include <RayTracer/host/BruteForce.h>
#include <RayTracer/host/BVH.h>
#include <RayTracer/host/KDTree.h>
#include <RayTracer/host/PPMTarget.h>
#include <RayTracer/host/Scene.h>

#include <RayTracer/device/Camera.cuh>
#include <RayTracer/device/PPMTarget.cuh>
#include <RayTracer/device/BruteForce.cuh>
#include <RayTracer/device/Scene.cuh>

#define TINYOBJLOADER_IMPLEMENTATION 
#include "../RayTracer/tiny_obj_loader.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

struct Config {
    std::string scene;
    unsigned int samplesPerPixel;
    unsigned int maxDepth;
    bool cuda;
};

void from_json(const nlohmann::json& json, Config& config) {
    json.at("scene").get_to(config.scene);
    json.at("samples_per_pixel").get_to(config.samplesPerPixel);
    json.at("max_depth").get_to(config.maxDepth);
    json.at("cuda").get_to(config.cuda);
}

void HostMain(const Config& config) {
    rt::Camera camera(
        rt::Point3(0.0f, 0.0f, 25.0f),      // look from
        rt::Point3(0.0f, 2.0f, 0.0f),       // look at
        rt::Vector3(0.0f, 1.0f, 0.0f),      // up
        20.0f,                              // vfov
        16.0f / 9.0f,                       // aspect ratio
        0.1f,                               // aperture
        10.0f                               // focus_distance
    );
    rt::BruteForce bf;
    rt::BVH bvh;
    rt::KDTree kdTree;

    rt::PPMTarget target(400, 300);
    rt::Scene scene(&camera, &bf, &target);

    scene.LoadScene(config.scene);
    scene.GenerateFrame(config.samplesPerPixel, config.maxDepth);
}

void DeviceMain(const Config& config) {
    rt::device::Camera camera(
        rt::Point3(0.0f, 0.0f, 25.0f),      // look from
        rt::Point3(0.0f, 2.0f, 0.0f),       // look at
        rt::Vector3(0.0f, 1.0f, 0.0f),      // up
        20.0f,                              // vfov
        16.0f / 9.0f,                       // aspect ratio
        0.1f,                               // aperture
        10.0f                               // focus_distance
    );
    rt::device::PPMTarget target(400, 300);
    rt::device::BruteForce bf;

    rt::device::Scene scene(&camera, &bf, &target);
    scene.LoadScene(config.scene);
    scene.GenerateFrame(config.samplesPerPixel, config.maxDepth, 8, 8);
}

int main(int argc, char* argv[]) {
    std::ifstream jsonFile(argv[1]);
    nlohmann::json configJson;
    jsonFile >> configJson;
    Config config = configJson.get<Config>();

    if(config.cuda) {
        DeviceMain(config);
    } else {
        HostMain(config);
    }

    return 0;
}
