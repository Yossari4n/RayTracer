#include "BruteForce.h"

namespace rt {

void BruteForce::PartitionSpace(const std::vector<Mesh>& raytracables) {
    m_RayTracables = raytracables;
}

Color BruteForce::Traverse(const Ray& ray, unsigned int depth, const Color& missColor) const {
    Mesh::RayTraceResult result{};
    for(const auto& raytracable : m_RayTracables) {
        if(raytracable.RayTrace(ray, 0.001f, FLT_MAX, result)) {
            return result.m_Attenuation;
        }
    }

    return missColor;
}

}