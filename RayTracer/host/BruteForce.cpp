#include "BruteForce.h"

namespace rt {

void BruteForce::PartitionSpace(const std::vector<Mesh>& raytracables) {
    m_rayTracables = raytracables;
}

Color BruteForce::Traverse(const Ray& ray, unsigned int depth, const Color& missColor) const {
    Mesh::RayTraceRecord record{};
    for(const auto& raytracable : m_rayTracables) {
        const Mesh::RayTraceResult result = raytracable.RayTrace(ray, 0.001f, FLT_MAX, record);
        switch(result)
        {
        case Mesh::RayTraceResult::Scattered:
            return record.m_attenuation;
            break;

        case Mesh::RayTraceResult::Emitted:
            return record.m_emitted;
            break;

        default:
            break;
        }
    }

    return missColor;
}

}