#include <RayTracer/Ray.h>

#include <iostream>

int main() {
    rt::Ray ray;
    rt::Ray ray2(rt::Point3(1.0f), rt::Vector3(1.0f));

    rt::Ray ray3(ray);

    return 0;
}