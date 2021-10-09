#include "KDTree.h"

namespace rt {

namespace {

Mesh::SplitAxis RandomSplitAxis() {
    switch(RandomInt(0, 3))
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

Color KDTree::Traverse(const Ray& ray, unsigned int depth, const Color& missColor) const {
    if(depth == 0) {
        return Color(0.0f);
    }

    Mesh::RayTraceRecord record{};
    const Mesh::RayTraceResult result = FindClosestHit(*m_root, ray, 0.001f, FLT_MAX, record);
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

void KDTree::InnerPartitionSpace(Node& curr, const std::vector<Mesh>& raytracables, int depth) {
    for(const auto raytracable : raytracables) {
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

Mesh::RayTraceResult KDTree::FindClosestHit(const KDTree::Node& node, const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const {
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
    if(node.m_right) {
        result = FindClosestHit(*node.m_right, ray, minTime, maxTime, record);
    }

    if(node.m_left) {
        if(auto leftResult = FindClosestHit(*node.m_left, ray, minTime, record.m_time, record); leftResult != Mesh::RayTraceResult::Missed) {
            result = leftResult;
        }
    }

    return result;
}

}
