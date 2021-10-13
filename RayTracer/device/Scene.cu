#include "Scene.cuh"

#pragma warning(push, 0)
#include "../tiny_obj_loader.h"
#pragma warning(pop)

#include "IRayGenerator.cuh"
#include "ISpacePartitioner.cuh"
#include "IRenderTarget.cuh"

#include "../Debug.h"
#include "../Mesh.h"

#include <vector>

namespace rt::device {

namespace {

__global__ void GenerateFrameKernel() {
    printf("GenerateFrameKernel\n");
}

}

Scene::Scene(IRayGenerator* rayGenerator, ISpacePartitioner* spacePartitioner, IRenderTarget* renderTarget)
    : m_rayGenerator(rayGenerator)
    , m_spacePartitioner(spacePartitioner)
    , m_renderTarget(renderTarget) {}

void Scene::LoadScene(const std::string& path) {
    tinyobj::ObjReaderConfig config;

    tinyobj::ObjReader reader;
    if(!reader.ParseFromFile(path, config)) {
        if(!reader.Error().empty()) {
            LOG_ERROR("Failed to load scene " << path << reader.Error());
        }
        return;
    }

    if(!reader.Warning().empty()) {
        LOG_WARNING(reader.Warning());
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    std::vector<Mesh> meshes;
    meshes.reserve(shapes.size());
    for(const auto& shape : shapes) {
        meshes.emplace_back(attrib, shape, materials[shape.mesh.material_ids[0]]);
    }

    m_spacePartitioner->PartitionSpace(meshes);
}

void Scene::GenerateFrame(unsigned int samplesPerPixe, unsigned int maxDepth) const {
    GenerateFrameKernel<<<1, 1>>>();
}

}
