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
#include "../Metrics.h"

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
    const unsigned int maxX = static_cast<unsigned int>((*d_target)->Width());
    const unsigned int maxY = static_cast<unsigned int>((*d_target)->Height());

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= maxX) || (j >= maxY))
        return;

    const int pixelIndex = j * maxX + i;
    curandState localRandState = randState[pixelIndex];
    Color color(0.0f);
    for(int s = 0; s < samplesPerPixel; s++) {
        float u = float(i + curand_uniform(&localRandState)) / float(maxX);
        float v = float(j + curand_uniform(&localRandState)) / float(maxY);
        Ray ray = (*d_rayGenerator)->GenerateRay(u, v, &localRandState);
        color += (*d_partitioner)->Traverse(ray, maxDepth, missColor, &localRandState);
    }

    randState[pixelIndex] = localRandState;
    (*d_target)->WriteColor(i, j, color, samplesPerPixel);
}

}

class Scene {
public:
    Scene(IRayGenerator* rayGenerator, IAccelerationStructure* accelerationStructure, IRenderTarget* renderTarget)
        : m_rayGenerator(rayGenerator)
        , m_accelerationStructure(accelerationStructure)
        , m_renderTarget(renderTarget) {}

    void LoadScene(const std::string& path) {
        tinyobj::ObjReaderConfig config;

        LOG_INFO("Loading scene %s\n", path.c_str());
        tinyobj::ObjReader reader;
        if(!reader.ParseFromFile(path, config)) {
            if(!reader.Error().empty()) {
                LOG_ERROR("Failed to load scene\n");
            }
            return;
        }

        if(!reader.Warning().empty()) {
            LOG_WARNING("%s\n", reader.Warning().c_str());
        }

        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();
        auto& materials = reader.GetMaterials();

        tinyobj::material_t defaultMaterial;
        defaultMaterial.diffuse[0] = 1.0f;
        defaultMaterial.diffuse[1] = 1.0f;
        defaultMaterial.diffuse[2] = 1.0f;

        std::vector<Mesh> meshes;
        meshes.reserve(shapes.size());
        for(const auto& shape : shapes) {
            if(materials.empty()) {
                meshes.emplace_back(attrib, shape, defaultMaterial);
            } else {
                meshes.emplace_back(attrib, shape, materials[shape.mesh.material_ids[0]]);
            }
        }

        LOG_INFO("Partitioning space...");
        Metrics::Instance().BeginSpacePartitioning();
        m_accelerationStructure->PartitionSpace(meshes);
        float spacePartitionTime = Metrics::Instance().EndSpaceParitioning();
        LOG(" Done (%.2fms)\n", spacePartitionTime);
    }

    Metrics::Result GenerateFrame(unsigned int samplesPerPixel, unsigned int maxDepth, const Color& missColor, unsigned int tx, unsigned int ty) const {
        const unsigned int width = static_cast<unsigned int>(m_renderTarget->Width());
        const unsigned int height = static_cast<unsigned int>(m_renderTarget->Height());

        LOG_INFO("Generating %d x %d frame...", width, height);

        const dim3 blocks(width / tx + 1, height / ty + 1);
        const dim3 threads(tx, ty);

        // Init CUDA random
        curandState* state = nullptr;
        CHECK_CUDA( cudaMalloc((void**)&state, width * height * sizeof(curandState)) );
        InitCurandState<<<blocks, threads>>>(width, height, state);
        CHECK_CUDA( cudaGetLastError() );
        CHECK_CUDA( cudaDeviceSynchronize() );

        Metrics::Instance().BeginFrame();
        GenerateFrameKernel<<<blocks, threads>>>(samplesPerPixel, maxDepth, missColor, m_rayGenerator->ToDevice(), m_accelerationStructure->ToDevice(), m_renderTarget->ToDevice(), state);
        CHECK_CUDA( cudaGetLastError() );
        CHECK_CUDA( cudaDeviceSynchronize() );
        const auto frameTime = Metrics::Instance().EndFrame();
        LOG(" Done (%.2fms)\n", frameTime);

        LOG_INFO("Saving buffer... ");
        Metrics::Instance().BeginSaveBuffer();
        m_renderTarget->SaveBuffer();
        const auto saveBufferTime = Metrics::Instance().EndSaveBuffer();
        LOG(" Done (%.2fms)\n", saveBufferTime);

        return Metrics::Instance().Value();
    }

private:
    IRayGenerator* m_rayGenerator;
    IAccelerationStructure* m_accelerationStructure;
    IRenderTarget* m_renderTarget;
};

}

}

#endif
