#include <RayTracer/host/Camera.h>
#include <RayTracer/host/BruteForce.h>
#include <RayTracer/host/BVH.h>
#include <RayTracer/host/KDTree.h>
#include <RayTracer/host/PPMTarget.h>
#include <RayTracer/host/Scene.h>

#include <RayTracer/device/Camera.cuh>
#include <RayTracer/device/Scene.cuh>

#define TINYOBJLOADER_IMPLEMENTATION 
#include "../RayTracer/tiny_obj_loader.h"

#include <iostream>
#include <string>

void HostMain(const std::string& path) {
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

    scene.LoadScene(path);
    scene.GenerateFrame(25, 10);
}

void DeviceMain(const std::string& path) {
    rt::device::Camera camera(
        rt::Point3(0.0f, 0.0f, 25.0f),      // look from
        rt::Point3(0.0f, 2.0f, 0.0f),       // look at
        rt::Vector3(0.0f, 1.0f, 0.0f),      // up
        20.0f,                              // vfov
        16.0f / 9.0f,                       // aspect ratio
        0.1f,                               // aperture
        10.0f                               // focus_distance
    );

    rt::device::Scene scene(&camera, nullptr, nullptr);
    scene.LoadScene(path);
    scene.GenerateFrame(1, 10, 8, 8);
}

int main(int argc, char* argv[]) {
    const std::string path = argv[1];
    HostMain(path);

    return 0;
}
