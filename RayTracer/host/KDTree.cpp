#include "KDTree.h"

namespace rt {

namespace {

Mesh::SplitAxis RandomSplitAxis() {
    switch(Random(0, 3))
    {
    case 0:
        return Mesh::SplitAxis::X;

    case 1:
        return Mesh::SplitAxis::Y;

    default:
        return Mesh::SplitAxis::Z;
    }
}

}

void KDTree::PartitionSpace(const MeshList& raytracables) {
    m_root = std::make_unique<KDTree::Node>();

    InnerPartitionSpace(*m_root, raytracables, 1);
}

void KDTree::InnerPartitionSpace(Node& curr, const std::vector<Mesh>& raytracables, int depth) {
    for(const auto& raytracable : raytracables) {
        curr.m_volume = AABB(curr.m_volume, raytracable.Volume());
    }

    if(depth == 0) {
        curr.m_raytracables = raytracables;
        return;
    }

    const Mesh::SplitAxis splitAxis = RandomSplitAxis();
    const Point3 midPoint = curr.m_volume.Centroid();

    std::vector<Mesh> closers;
    std::vector<Mesh> furthers;
    for(const auto raytracable : raytracables) {
        auto splitted = raytracable.Split(midPoint, splitAxis);
        
        if(splitted.first.has_value()) {
            closers.push_back(splitted.first.value());
        }

        if(splitted.second.has_value()) {
            furthers.push_back(splitted.second.value());
        }
    }

    if(!closers.empty()) {
        curr.m_left = std::make_unique<Node>();
        InnerPartitionSpace(*curr.m_left, closers, depth - 1);
    }

    if(!furthers.empty()) {
        curr.m_right = std::make_unique<Node>();
        InnerPartitionSpace(*curr.m_right, furthers, depth - 1);
    }
}

Mesh::RayTraceResult KDTree::FindClosestHit(const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const {
    return InnerFindClosestHit(*m_root, ray, minTime, maxTime, record);
}

Mesh::RayTraceResult KDTree::InnerFindClosestHit(const KDTree::Node& node, const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const {
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
