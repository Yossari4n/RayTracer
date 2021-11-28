#include "KDTreeNode.h"

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

KDTreeNode::KDTreeNode(const std::vector<Mesh>& raytracables, int depth) {
    for(const auto& raytracable : raytracables) {
        m_volume = AABB(m_volume, raytracable.Volume());
    }

    if(depth == 0) {
        m_raytracables = raytracables;
        m_depth = 1;
        return;
    }

    const Mesh::SplitAxis splitAxis = RandomSplitAxis();
    const Point3 midPoint = m_volume.Centroid();

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

    m_left = std::make_unique<KDTreeNode>(closers, depth - 1);
    m_right = std::make_unique<KDTreeNode>(furthers, depth - 1);
    m_depth = std::max(m_left->m_depth, m_right->m_depth) + 1U;
}

}
