#include "BruteForce.h"
#include "../Debug.h"

namespace rt {

void BruteForce::PartitionSpace(const MeshList& raytracables) {
    m_rayTracables = raytracables;
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