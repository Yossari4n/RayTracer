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
public:
    class IDevice {
    public:
        __device__ virtual ~IDevice() {}

        __device__ virtual void WriteColor(size_t x, size_t y, const Color& color, unsigned int samplesPerPixel) = 0;
        __device__ virtual size_t Width() const = 0;
        __device__ virtual size_t Height() const = 0;
        __device__ virtual Color* FrameBuffer() = 0;
    };

    using DevicePtr = IRenderTarget::IDevice**;

    virtual ~IRenderTarget() = default;

    virtual void SaveBuffer() = 0;
    virtual size_t Width() const  = 0;
    virtual size_t Height() const = 0;

    virtual DevicePtr ToDevice() const = 0;
};

}

#endif
