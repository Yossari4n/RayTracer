#include "Scene.h"

#include "../tiny_obj_loader.h"

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

    m_SpacePartitioner->PartitionSpace(std::move(meshes));
}

void Scene::GenerateFrame(unsigned int samples_per_pixel, unsigned int max_depth) const {
    const size_t width = m_RenderTarget->Width();
    const size_t height = m_RenderTarget->Height();

    for(int j = height - 1; j >= 0; j--) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for(int i = 0; i < width; i++) {
            Color color(0.0f);

            // multisampling
            for(int s = 0; s < samples_per_pixel; s++) {
                const float u = static_cast<float>(i + RandomFloat()) / (static_cast<float>(width) - 1);
                const float v = static_cast<float>(j + RandomFloat()) / (static_cast<float>(height) - 1);

                const Ray& ray = m_RayGenerator->GenerateRay(u, v);
                color += m_SpacePartitioner->Traverse(ray, max_depth, Color(0.0f));
            }

            m_RenderTarget->WriteColor(i, height - j - 1, color, samples_per_pixel);
        }
    }

    m_RenderTarget->SaveBuffer();
}

}
