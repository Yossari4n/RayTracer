#ifndef BruteForce_cuh
#define BruteForce_cuh

#include "IAccelerationStructure.cuh"

namespace rt::device {

class BruteForce : public IAccelerationStructure {
public:
    class BruteForceDevice : public IDevice {
        __device__ Color Traverse(const Ray& ray, unsigned int depth, const Color& missColor, curandState* randState) const override {
            return Color(1.0f, 0.0f, 0.0f);
        }
    };

    BruteForce();
    ~BruteForce();

    void PartitionSpace(const MeshList& raytracables) {

    }

    DevicePtr ToDevice() const {
        return d_bruteForce;
    }

private:
    DevicePtr d_bruteForce{nullptr};
};

namespace {

__global__ void CreateBruteForceDeviceObject(IAccelerationStructure::DevicePtr bruteForcePtr) {
    (*bruteForcePtr) = new BruteForce::BruteForceDevice();
}

__global__ void DeleteCameraDeviceObject(IAccelerationStructure::DevicePtr bruteForcePtr) {
    delete bruteForcePtr;
}

}

BruteForce::BruteForce() {
    CHECK_CUDA(cudaMalloc((void**)&d_bruteForce, sizeof(IAccelerationStructure)));
    CreateBruteForceDeviceObject<<<1, 1>>>(d_bruteForce);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

BruteForce::~BruteForce() {
    DeleteCameraDeviceObject<<<1, 1>>>(d_bruteForce);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

}

#endif
