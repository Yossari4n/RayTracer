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

Color BVH::Traverse(const Ray& ray, unsigned int depth, const Color& missColor) const {
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

void BVH::InnerParitionSpace(Node& curr, std::vector<const Mesh*>& raytracablePtrs, size_t start, size_t end) {
    const size_t span = end - start;
    if(span == 1) {
        curr.m_raytracable = std::make_unique<Mesh>(*raytracablePtrs[start]);
        curr.m_volume = curr.m_raytracable->Volume();
    } else {
        switch(RandomInt(0, 2))
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

Mesh::RayTraceResult BVH::FindClosestHit(const BVH::Node& node, const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const {
    if(!node.m_volume.Hit(ray, minTime, maxTime)) {
        return Mesh::RayTraceResult::Missed;
    }

    if(node.m_raytracable) {
        return node.m_raytracable->RayTrace(ray, minTime, maxTime, record);
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
