#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>

#include <RayTracer/test.h>

#include <iostream>

int main() {
    cuda_test<<<1,1>>>();
    test();
    return 0;
}