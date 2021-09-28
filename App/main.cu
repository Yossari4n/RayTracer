#include <RayTracer/host/StaticScene.h>
#include <RayTracer/host/Camera.h>
#include <RayTracer/host/BruteForce.h>
#include <RayTracer/host/PPMTarget.h>


#include <iostream>

int main() {
    rt::Camera camera(
        rt::Point3(-15.0f, -10.0f, 0.0f),   // look from
        rt::Point3(-10.0f, -10.0f, 0.0f),   // look at
        rt::Vector3(0.0f, 1.0f, 0.0f),      // up
        20.0f,                              // vfov
        16.0f / 9.0f,                       // aspect ratio
        0.1f,                               // aperture
        10.0f                               // focus_distance
    );
    rt::BruteForce bf;
    rt::PPMTarget target(800, 600);
    rt::StaticScene scene(&camera, &bf, &target);

    return 0;
}