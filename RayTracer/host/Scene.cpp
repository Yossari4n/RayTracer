#include "Scene.h"

#pragma warning(push, 0)
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
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
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path.c_str(), aiProcess_Triangulate | aiProcess_FlipUVs);
    if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "Failed to load model\n";
        return;
    }

    std::vector<Mesh> meshes;
    meshes.reserve(scene->mNumMeshes);
    for(unsigned int i = 0; i < scene->mNumMeshes; i++) {
        meshes.emplace_back(scene->mMeshes[i]);
    }
}

void Scene::GenerateFrame(unsigned int samples_per_pixel, unsigned int max_depth) const {
    
}

}
