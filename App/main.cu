#include <RayTracer/host/StaticScene.h>
#include <RayTracer/host/PPMTarget.h>

#include <iostream>

int main() {
    rt::PPMTarget target(800, 600);
    rt::StaticScene scene(nullptr, nullptr, &target);

    return 0;
}