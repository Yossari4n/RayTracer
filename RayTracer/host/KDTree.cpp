#include "KDTree.h"

namespace rt {
KDTree::KDTree(unsigned int maxDepth)
    : m_maxDepth(maxDepth) {}

void KDTree::PartitionSpace(const MeshList& raytracables) {
    m_root = std::make_unique<KDTreeNode>(raytracables, m_maxDepth);
}

Mesh::RayTraceResult KDTree::FindClosestHit(const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const {
    return InnerFindClosestHit(*m_root, ray, minTime, maxTime, record);
}

Mesh::RayTraceResult KDTree::InnerFindClosestHit(const KDTreeNode& node, const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const {
    Metrics::Instance().VolumeTested();
    if(!node.m_volume.Hit(ray, minTime, maxTime)) {
        return Mesh::RayTraceResult::Missed;
    }

    if(!node.m_raytracables.empty()) {
        record.m_time = maxTime;

        Mesh::RayTraceResult result = Mesh::RayTraceResult::Missed;
        for(const auto& raytracable : node.m_raytracables) {
            if(const auto currResult = raytracable.RayTrace(ray, minTime, record.m_time, record); currResult != Mesh::RayTraceResult::Missed) {
                result = currResult;
            }
        }

        return result;
    }

    Mesh::RayTraceResult result = Mesh::RayTraceResult::Missed;
    if(node.m_left) {
        result = InnerFindClosestHit(*node.m_left, ray, minTime, maxTime, record);
    }

    if(node.m_right) {
        maxTime = result != Mesh::RayTraceResult::Missed ? record.m_time : maxTime;
        if(auto leftResult = InnerFindClosestHit(*node.m_right, ray, minTime, maxTime, record); leftResult != Mesh::RayTraceResult::Missed) {
            result = leftResult;
        }
    }

    return result;
}

}
