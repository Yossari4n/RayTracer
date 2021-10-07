#include "Scene.h"

#pragma warning(push, 0)
#include "../tiny_obj_loader.h"
#pragma warning(pop)

#include "IRayGenerator.h"
#include "ISpacePartitioner.h"
#include "IRenderTarget.h"
#include "Mesh.h"

#include "../Debug.h"

#include <iostream>

namespace rt {

Scene::Scene(IRayGenerator* ray_generator, ISpacePartitioner* space_partitioner, IRenderTarget* render_target)
    : m_rayGenerator(ray_generator)
    , m_spacePartitioner(space_partitioner)
    , m_renderTarget(render_target) {}

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
        meshes.emplace_back(attrib, shape);
    }

    m_spacePartitioner->PartitionSpace(std::move(meshes));
}

void Scene::GenerateFrame(unsigned int samplesPerPixel, unsigned int maxDepth) const {
    const int width = static_cast<int>(m_renderTarget->Width());
    const int height = static_cast<int>(m_renderTarget->Height());

    for(int j = height - 1; j >= 0; j--) {
        LOG_INFO("Scanlines remaining: " << j << ' ' << std::flush);

        for(int i = 0; i < width; i++) {
            Color color(0.0f);

            // multisampling
            for(unsigned int s = 0; s < samplesPerPixel; s++) {
                const float u = static_cast<float>(i + RandomFloat()) / (static_cast<float>(width) - 1);
                const float v = static_cast<float>(j + RandomFloat()) / (static_cast<float>(height) - 1);

                const Ray& ray = m_rayGenerator->GenerateRay(u, v);
                color += m_spacePartitioner->Traverse(ray, maxDepth, Color(0.0f));
            }

            m_renderTarget->WriteColor(i, height - j - 1, color, samplesPerPixel);
        }
    }

    LOG_INFO('\n');
    LOG_INFO("Saving buffer\n");
    m_renderTarget->SaveBuffer();
}

}
