#include "Scene.h"

#include "IRayGenerator.h"
#include "ISpacePartitioner.h"
#include "IRenderTarget.h"

namespace rt {

Scene::Scene(IRayGenerator* ray_generator, ISpacePartitioner* space_partitioner, IRenderTarget* render_target)
    : m_RayGenerator(ray_generator)
    , m_SpacePartitioner(space_partitioner)
    , m_RenderTarget(render_target) {}

void Scene::GenerateFrame(unsigned int samples_per_pixel, unsigned int max_depth) const {
    
}

}
