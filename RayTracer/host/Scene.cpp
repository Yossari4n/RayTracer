#include "Scene.h"

#pragma warning(push, 0)
#include "../tiny_obj_loader.h"
#pragma warning(pop)

#include "IRayGenerator.h"
#include "ISpacePartitioner.h"
#include "IRenderTarget.h"
#include "Mesh.h"

#include "../Debug.h"
#include "../DebugMaterial.h"

#include <iostream>
#include <memory>

namespace rt {

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

void Scene::GenerateFrame(unsigned int samplesPerPixel, unsigned int maxDepth) const {
    const int width = static_cast<int>(m_renderTarget->Width());
    const int height = static_cast<int>(m_renderTarget->Height());
    const Color missColor(0.0f);

    for(int j = height - 1; j >= 0; j--) {
        LOG_INFO("Scanlines remaining: " << j << ' ' << std::flush);

        for(int i = 0; i < width; i++) {
            Color color(0.0f);

            // multisampling
            for(unsigned int s = 0; s < samplesPerPixel; s++) {
                const float u = static_cast<float>(i + Random<float>()) / (static_cast<float>(width) - 1);
                const float v = static_cast<float>(j + Random<float>()) / (static_cast<float>(height) - 1);

                const Ray& ray = m_rayGenerator->GenerateRay(u, v);
                color += m_spacePartitioner->Traverse(ray, maxDepth, missColor);
            }

            m_renderTarget->WriteColor(i, height - j - 1, color, samplesPerPixel);
        }
    }

    LOG_INFO('\n');
    m_renderTarget->SaveBuffer();
}

}
