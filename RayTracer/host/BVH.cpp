#include "BVH.h"

namespace rt {

void BVH::PartitionSpace(const MeshList& raytracables) {
    m_root = std::make_unique<BVH::Node>();

    std::vector<const Mesh*> raytracablePtrs;
    raytracablePtrs.reserve(raytracables.size());
    for(const auto& raytracable : raytracables) {
        raytracablePtrs.push_back(&raytracable);
    }

    InnerParitionSpace(*m_root, raytracablePtrs, 0, raytracablePtrs.size());
}

void BVH::InnerParitionSpace(Node& curr, std::vector<const Mesh*>& raytracablePtrs, size_t start, size_t end) {
    const size_t span = end - start;
    if(span == 1) {
        curr.m_raytracable = std::make_unique<Mesh>(*raytracablePtrs[start]);
        curr.m_volume = curr.m_raytracable->Volume();
    } else {
        switch(Random(0, 2))
        {
        case 0:
            std::sort(raytracablePtrs.begin() + start, raytracablePtrs.begin() + end, AxisXComparator());
            break;

        case 1:
            std::sort(raytracablePtrs.begin() + start, raytracablePtrs.begin() + end, AxisYComparator());
            break;

        default:
            std::sort(raytracablePtrs.begin() + start, raytracablePtrs.begin() + end, AxisZComparator());
            break;
        }

        const size_t mid = start + span / 2;
        curr.m_left = std::make_unique<Node>();
        InnerParitionSpace(*curr.m_left, raytracablePtrs, start, mid);

        curr.m_right = std::make_unique<Node>();
        InnerParitionSpace(*curr.m_right, raytracablePtrs, mid, end);

        curr.m_volume = AABB(curr.m_left->m_volume, curr.m_right->m_volume);
    }
}

Mesh::RayTraceResult BVH::FindClosestHit(const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const {
    return InnerFindClosestHit(*m_root, ray, minTime, maxTime, record);
}

Mesh::RayTraceResult BVH::InnerFindClosestHit(const BVH::Node& node, const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const {
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
