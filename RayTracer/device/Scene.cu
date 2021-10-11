#include "Scene.cuh"

namespace rt::device {

namespace {

__global__ void LoadSceneKernel() {
    printf("LoadSceneKernel\n");
}

__global__ void GenerateFrameKernel() {
    printf("GenerateFrameKernel\n");
}

}

Scene::Scene(IRayGenerator* rayGenerator, ISpacePartitioner* spacePartitioner, IRenderTarget* renderTarget)
    : m_rayGenerator(rayGenerator)
    , m_spacePartitioner(spacePartitioner)
    , m_renderTarget(renderTarget) {}

void Scene::LoadScene(const std::string& path) {
    LoadSceneKernel<<<1, 1>>>();
}

void Scene::GenerateFrame(unsigned int samplesPerPixe, unsigned int maxDepth) const {
    GenerateFrameKernel<<<1, 1>>>();
}

}
