#include "Scene.h"

#pragma warning(push, 0)
#include "../tiny_obj_loader.h"
#pragma warning(pop)

#include "IRayGenerator.h"
#include "ISpacePartitioner.h"
#include "IRenderTarget.h"
#include "Mesh.h"

#include <iostream>

namespace rt {

Scene::Scene(IRayGenerator* ray_generator, ISpacePartitioner* space_partitioner, IRenderTarget* render_target)
    : m_RayGenerator(ray_generator)
    , m_SpacePartitioner(space_partitioner)
    , m_RenderTarget(render_target) {}

void Scene::LoadScene(const std::string& path) {
    tinyobj::ObjReaderConfig config;

    tinyobj::ObjReader reader;
    if(!reader.ParseFromFile(path, config)) {
        if(!reader.Error().empty()) {
            std::cerr << "[LoadScene][Error] Failed to load scene " << path << '\n' << reader.Error();
        }
        return;
    }

    if(!reader.Warning().empty()) {
        std::cerr << "[LoadScene][Warning] " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    std::vector<Mesh> meshes;
    meshes.reserve(shapes.size());
    for(const auto& shape : shapes) {
        meshes.emplace_back(attrib, shape);
    }
}

void Scene::GenerateFrame(unsigned int samples_per_pixel, unsigned int max_depth) const {
    
}

}
