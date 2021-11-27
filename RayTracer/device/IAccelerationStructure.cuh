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
        __device__ virtual ~IDevice() {}

        __device__ Color Traverse(const Ray& ray, unsigned int depth, const Color& missColor, curandState* randState) const {
            Ray currentRay = ray;
            Color currentColor(1.0f);

            for(int i = 0; i < depth; i++) {
                Mesh::RayTraceRecord record{};
                const Mesh::RayTraceResult result = FindClosestHit(currentRay, 0.01f, FLT_MAX, randState, record);
                switch(result) {
                case Mesh::RayTraceResult::Scattered:
                    //printf("Scattered\n");
                    currentRay = record.m_scattered;
                    currentColor *= (record.m_attenuation + record.m_emitted);
                    break;

                case Mesh::RayTraceResult::Emitted:
                    //printf("Emitted\n");
                    return currentColor * record.m_emitted;
                    break;

                case Mesh::RayTraceResult::Missed:
                    return currentColor * missColor;
                    break;
                }
            }

            return currentColor;
        }

    protected:
        __device__ virtual Mesh::RayTraceResult FindClosestHit(const Ray& ray, float minTime, float maxTime, curandState* randState, Mesh::RayTraceRecord& record) const = 0;
    };

    using DevicePtr = IAccelerationStructure::IDevice**;
    using MeshList = std::vector<Mesh>;

    virtual ~IAccelerationStructure() = default;

    virtual void PartitionSpace(const MeshList& raytracables) = 0;

    virtual DevicePtr ToDevice() const = 0;
};

}

#endif
