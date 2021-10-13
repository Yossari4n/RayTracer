#pragma warning(push, 0)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>
#pragma warning(pop)

#include <RayTracer/Camera.h>
#include <RayTracer/DebugMaterial.h>
#include <RayTracer/host/Scene.h>
#include <RayTracer/host/BruteForce.h>
#include <RayTracer/host/BVH.h>
#include <RayTracer/host/KDTree.h>
#include <RayTracer/host/PPMTarget.h>
#include <RayTracer/device/Scene.cuh>

#define TINYOBJLOADER_IMPLEMENTATION 
#include "../RayTracer/tiny_obj_loader.h"

#include <iostream>
#include <string>

#include <RayTracer/AABB.h>
#include <RayTracer/Color.h>
#include <RayTracer/Math.h>
#include <RayTracer/Ray.h>
#include <RayTracer/Triangle.h>
__global__ void Kernel() {
    rt::Point3 p1(1.0f);
    rt::Vector3 v1(1.0f);
    rt::AABB v(p1, p1);
}

int main(int argc, char* argv[]) {
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
    rt::PPMTarget target(400, 300);
    rt::Scene scene(&camera, &bf, &target);

    scene.LoadScene(path);
    scene.GenerateFrame(25, 10);

    return 0;
}