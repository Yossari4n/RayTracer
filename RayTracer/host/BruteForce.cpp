#include "BruteForce.h"

namespace rt {

void BruteForce::PartitionSpace(const MeshList& raytracables) {
    m_rayTracables = raytracables;
}

Color BruteForce::Traverse(const Ray& ray, unsigned int depth, const Color& missColor) const {
    if(depth == 0) {
        return Color(0.0f);
    }

    Mesh::RayTraceRecord record{};
    const Mesh::RayTraceResult result = FindClosestHit(ray, 0.001f, FLT_MAX, record);
    switch(result)
    {
    case Mesh::RayTraceResult::Scattered:
        return record.m_emitted + record.m_attenuation * Traverse(record.m_scattered, depth - 1, missColor);
        break;

    case Mesh::RayTraceResult::Emitted:
        return record.m_emitted;
        break;

    case Mesh::RayTraceResult::Missed:
    default:
        return missColor;
        break;
    }
}

Mesh::RayTraceResult BruteForce::FindClosestHit(const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const {
    Mesh::RayTraceResult result = Mesh::RayTraceResult::Missed;

    record.m_time = maxTime;
    for(const auto& raytracable : m_rayTracables) {
        if(const auto currResult = raytracable.RayTrace(ray, minTime, record.m_time, record); currResult != Mesh::RayTraceResult::Missed) {
            result = currResult;
        }
    }

    return result;
}

}