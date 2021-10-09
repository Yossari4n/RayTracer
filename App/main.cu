#include <RayTracer/Camera.h>

#include <RayTracer/host/Scene.h>
#include <RayTracer/host/BruteForce.h>
#include <RayTracer/host/BVH.h>
#include <RayTracer/host/KDTree.h>
#include <RayTracer/host/PPMTarget.h>
#include <RayTracer/DebugMaterial.h>

#define TINYOBJLOADER_IMPLEMENTATION 
#include "../RayTracer/tiny_obj_loader.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if(argc != 2) {
        std::cerr << "Wrong arguments\n";
        return -1;
    }

    const std::string path = argv[1];

    rt::Camera camera(
        rt::Point3(0.0f, 0.0f, 25.0f),       // look from
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
    rt::PPMTarget target(800, 600);
    rt::Scene scene(&camera, &kdTree, &target);

    scene.LoadScene(path);
    scene.GenerateFrame(25, 50);

    return 0;
}