#include "BVH.h"

namespace rt {

void BVH::PartitionSpace(const MeshList& raytracables) {
    m_root = std::make_unique<BVHNode>(raytracables);
}

Mesh::RayTraceResult BVH::FindClosestHit(const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const {
    return InnerFindClosestHit(*m_root, ray, minTime, maxTime, record);
}

Mesh::RayTraceResult BVH::InnerFindClosestHit(const BVHNode& node, const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const {
    if(!node.m_volume.Hit(ray, minTime, maxTime)) {
        return Mesh::RayTraceResult::Missed;
    }

    if(node.m_raytracable) {
        auto test = node.m_raytracable->RayTrace(ray, minTime, maxTime, record);
        return test;
    }

    Mesh::RayTraceResult result = Mesh::RayTraceResult::Missed;
    if(node.m_right) {
        result = InnerFindClosestHit(*node.m_right, ray, minTime, maxTime, record);
    }

    if(node.m_left) {
        maxTime = result != Mesh::RayTraceResult::Missed ? record.m_time : maxTime;
        if(auto leftResult = InnerFindClosestHit(*node.m_left, ray, minTime, maxTime, record); leftResult != Mesh::RayTraceResult::Missed) {
            result = leftResult;
        }
    }

    return result;
}

}
