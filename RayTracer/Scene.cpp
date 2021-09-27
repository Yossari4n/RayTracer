#include "Scene.h"

namespace rt {

Scene::Scene(IRayGenerator* ray_generator, ISpacePartitioner* space_partitioner, IRenderTarget* render_target)
    : m_RayGenerator(ray_generator)
    , m_SpacePartitioner(space_partitioner)
    , m_RenderTarget(render_target) {}

}