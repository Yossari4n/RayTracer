#ifndef IAccelerationStructure_cuh
#define IAccelerationStructure_cuh

#pragma warning(push, 0)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>
#pragma warning(pop)

#include "../Color.h"
#include "../Mesh.h"

namespace rt::device {

class IAccelerationStructure {
public:
    class IDevice {
    public:
        __device__ virtual Color Traverse(const Ray& ray, unsigned int depth, const Color& missColor, curandState* randState) const = 0;
    };
    using DevicePtr = IAccelerationStructure::IDevice**;
    using MeshList = std::vector<Mesh>;

    virtual ~IAccelerationStructure() = default;

    virtual void PartitionSpace(const MeshList& raytracables) = 0;

    virtual DevicePtr ToDevice() const = 0;
};

}

#endif
