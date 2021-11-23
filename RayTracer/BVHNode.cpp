#include "BVHNode.h"

namespace rt {

BVHNode::BVHNode(std::vector<Mesh> raytracables)
    : BVHNode(raytracables, 0, raytracables.size()) {}

BVHNode::BVHNode(std::vector<Mesh>& raytracables, size_t start, size_t end) {
    const size_t span = end - start;
    if(span == 1) {
        m_raytracable = std::make_unique<Mesh>(raytracables[start]);
        m_volume = m_raytracable->Volume();
        m_depth = 1U;
    } else {
        switch(Random(0, 2))
        {
        case 0:
            std::sort(raytracables.begin() + start, raytracables.begin() + end, AxisXComparator());
            break;

        case 1:
            std::sort(raytracables.begin() + start, raytracables.begin() + end, AxisYComparator());
            break;

        default:
            std::sort(raytracables.begin() + start, raytracables.begin() + end, AxisZComparator());
            break;
        }

        const size_t mid = start + span / 2;

        m_left = std::move(std::unique_ptr<BVHNode>(new BVHNode(raytracables, start, mid)));
        m_right = std::move(std::unique_ptr<BVHNode>(new BVHNode(raytracables, mid, end)));
        m_volume = AABB(m_left->m_volume, m_right->m_volume);
        m_depth = std::max(m_left->m_depth, m_right->m_depth) + 1U;
    }
}

}