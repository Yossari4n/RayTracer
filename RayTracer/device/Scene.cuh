#ifndef Scene_cuh
#define Scene_cuh

#pragma warning(push, 0)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>
#pragma warning(pop)

#include "IRayGenerator.cuh"
#include "IRenderTarget.cuh"
#include "IAccelerationStructure.cuh"
#include "../Debug.h"
#include "../Mesh.h"

namespace rt {

namespace device {

namespace {

__global__ void InitCurandState(unsigned int maxX, unsigned int maxY, curandState* randState) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= maxX) || (j >= maxY))
        return;

    unsigned int pixelIndex = j * maxX + i;
    curand_init(1984U + pixelIndex, 0U, 0U, &randState[pixelIndex]);
}

__global__ void GenerateFrameKernel(unsigned int samplesPerPixel, unsigned int maxDepth, Color missColor, IRayGenerator::DevicePtr d_rayGenerator, IAccelerationStructure::DevicePtr d_partitioner, IRenderTarget::DevicePtr d_target, curandState* randState) {
    const unsigned int maxX = 1;//static_cast<unsigned int>((*d_target)->Width());
    const unsigned int maxY = 1;//static_cast<unsigned int>((*d_target)->Height());

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= maxX) || (j >= maxY))
        return;

    const int pixelIndex = j * maxX + i;
    curandState localRandState = randState[pixelIndex];
    glm::vec3 color(0, 0, 0);
    for(int s = 0; s < samplesPerPixel; s++) {
        float u = float(i + curand_uniform(&localRandState)) / float(maxX);
        float v = float(j + curand_uniform(&localRandState)) / float(maxY);
        Ray ray = (*d_rayGenerator)->GenerateRay(u, v, &localRandState);
        //color += (*d_partitioner)->Traverse(ray, maxDepth, missColor, &localRandState);
    }

    randState[pixelIndex] = localRandState;
    //(*d_target)->WriteColor(i, j, color, samplesPerPixel);
}

}

class Scene {
public:
    __host__ Scene(IRayGenerator* rayGenerator, IAccelerationStructure* spacePartitioner, IRenderTarget* renderTarget)
        : m_rayGenerator(rayGenerator)
        , m_spacePartitioner(spacePartitioner)
        , m_renderTarget(renderTarget) {}

    __host__ void LoadScene(const std::string& path) {}

    __host__ void GenerateFrame(unsigned int samplesPerPixel, unsigned int maxDepth, unsigned int tx, unsigned int ty) const {
        const unsigned int width = 1; //static_cast<unsigned int>(m_renderTarget->Width());
        const unsigned int height = 1; //static_cast<unsigned int>(m_renderTarget->Height());
        const unsigned int pixelsCount = width * height;
        const Color missColor(0.0f);

        const dim3 blocks(width / tx + 1, height / ty + 1);
        const dim3 threads(tx, ty);

        // Init CUDA random
        curandState* state = nullptr;
        CHECK_CUDA(cudaMalloc((void**)&state, width * height * sizeof(curandState)));
        InitCurandState<<<blocks, threads>>>(width, height, state);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        GenerateFrameKernel<<<blocks, threads>>>(samplesPerPixel, maxDepth, missColor, m_rayGenerator->ToDevice(), nullptr, nullptr, state);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

private:
    IRayGenerator* m_rayGenerator;
    IAccelerationStructure* m_spacePartitioner;
    IRenderTarget* m_renderTarget;
};

}

}

#endif
