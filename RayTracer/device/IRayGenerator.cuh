#ifndef IRayGenerator_cuh
#define IRayGenerator_cuh

#pragma warning(push, 0)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>
#pragma warning(pop)

#include "../Ray.h"

namespace rt::device {

class IRayGenerator {
public:
    virtual ~IRayGenerator() = default;

    __device__ virtual Ray GenerateRay(float s, float t, curandState* localRandState) const = 0;
};

}

#endif
