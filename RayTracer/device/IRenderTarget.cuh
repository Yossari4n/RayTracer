#ifndef IRenderTarget_cuh
#define IRenderTarget_cuh

#pragma warning(push, 0)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>
#pragma warning(pop)

#include "../Color.h"

namespace rt::device {

class IRenderTarget {
    virtual ~IRenderTarget() = default;

    __device__ virtual void WriteColor(size_t x, size_t y, const Color& color, unsigned int samplesPerPixel) = 0;
    virtual void SaveBuffer() = 0;

    virtual size_t Width() const  = 0;
    virtual size_t Height() const = 0;
};

}

#endif
