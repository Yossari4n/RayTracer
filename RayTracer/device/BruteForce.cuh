#ifndef BruteForce_cuh
#define BruteForce_cuh

#pragma warning(push, 0)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>
#pragma warning(pop)

#include "IAccelerationStructure.cuh"
#include "../Debug.h"

namespace rt::device {

class BruteForce : public IAccelerationStructure {
public:
    class BruteForceDevice : public IDevice {
    public:
        __device__ BruteForceDevice(Mesh* raytracables, size_t raytracablesCount)
            : m_raytracablesCount(raytracablesCount)
            , m_raytracables(raytracables) {}

        __device__ Mesh::RayTraceResult FindClosestHit(const Ray& ray, float minTime, float maxTime, curandState* randState, Mesh::RayTraceRecord& record) const override {
            Mesh::RayTraceResult result = Mesh::RayTraceResult::Missed;
            record.m_time = maxTime;
            for(int i = 0; i < m_raytracablesCount; i++) {
                if(const auto currResult = m_raytracables[i].RayTrace(ray, minTime, record.m_time, randState, record); currResult != Mesh::RayTraceResult::Missed) {
                    result = currResult;
                }
            }

            return result;
        }

    private:
        size_t m_raytracablesCount;
        Mesh* m_raytracables;
    };

    ~BruteForce();

    void PartitionSpace(const MeshList& raytracables) override;

    DevicePtr ToDevice() const {
        return d_bruteForce;
    }

private:
    DevicePtr d_bruteForce{nullptr};
};

namespace {

__global__ void CreateBruteForceDeviceObject(IAccelerationStructure::DevicePtr bruteForcePtr, Mesh* raytracables, size_t raytracableCount) {
    (*bruteForcePtr) = new BruteForce::BruteForceDevice(raytracables, raytracableCount);
}

__global__ void DeleteCameraDeviceObject(IAccelerationStructure::DevicePtr bruteForcePtr) {
    delete (*bruteForcePtr);
}

}

void BruteForce::PartitionSpace(const MeshList& raytracables) {
    Mesh* d_meshList;
    CHECK_CUDA( cudaMalloc(&d_meshList, sizeof(Mesh) * raytracables.size()) );

    for(int i = 0; i < raytracables.size(); i++) {
        Mesh mesh(raytracables[i]);

        CHECK_CUDA( cudaMalloc((void**)&mesh.m_triangles, sizeof(Triangle) * raytracables[i].m_triangleCount) );
        CHECK_CUDA( cudaMemcpy(mesh.m_triangles, raytracables[i].m_triangles, sizeof(Triangle) * raytracables[i].m_triangleCount, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(&d_meshList[i], &mesh, sizeof(Mesh), cudaMemcpyHostToDevice));

        mesh.m_triangles = nullptr;
    }

    CHECK_CUDA( cudaMalloc((void**)&d_bruteForce, sizeof(IAccelerationStructure::IDevice)) );
    CreateBruteForceDeviceObject<<<1, 1>>>(d_bruteForce, d_meshList, raytracables.size());
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );
}

BruteForce::~BruteForce() {
    DeleteCameraDeviceObject<<<1, 1>>>(d_bruteForce);
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );
}

}

#endif
