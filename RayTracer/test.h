#ifndef test_h
#define test_h

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>

void test();

__global__ void cuda_test();

#endif
