#include "Scene.h"

#pragma warning(push, 0)
#include "../tiny_obj_loader.h"
#pragma warning(pop)

#include "../Mesh.h"
#include "../Debug.h"

#include <iostream>
#include <memory>

namespace rt {

Scene::Scene(IRayGenerator* rayGenerator, IAccelerationStructure* accelerationStructure, IRenderTarget* renderTarget)
    : m_rayGenerator(rayGenerator)
    , m_accelerationStructure(accelerationStructure)
    , m_renderTarget(renderTarget) {}

void Scene::LoadScene(const std::string& path) {
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

Metrics::Result Scene::GenerateFrame(unsigned int samplesPerPixel, unsigned int maxDepth, const Color& missColor) const {
    const int width = static_cast<int>(m_renderTarget->Width());
    const int height = static_cast<int>(m_renderTarget->Height());

    LOG_INFO("Generating %d x %d frame...", width, height);

    Metrics::Instance().BeginFrame();
    for(int j = height - 1; j >= 0; j--) {
        for(int i = 0; i < width; i++) {
            Color color(0.0f);

            // multisampling
            for(unsigned int s = 0; s < samplesPerPixel; s++) {
                const float u = static_cast<float>(i + Random<float>()) / (static_cast<float>(width) - 1);
                const float v = static_cast<float>(j + Random<float>()) / (static_cast<float>(height) - 1);

                const Ray& ray = m_rayGenerator->GenerateRay(u, v);
                color += m_accelerationStructure->Traverse(ray, maxDepth, missColor);
            }

            m_renderTarget->WriteColor(i, height - j - 1U, color, samplesPerPixel);
        }
    }
    const auto frameTime = Metrics::Instance().EndFrame();
    LOG(" Done (%.2fms)\n", frameTime);

    LOG_INFO("Saving buffer... ");
    Metrics::Instance().BeginSaveBuffer();
    m_renderTarget->SaveBuffer();
    const auto saveBufferTime = Metrics::Instance().EndSaveBuffer();
    LOG(" Done (%.2fms)\n", saveBufferTime);

    return Metrics::Instance().Value();
}

}
