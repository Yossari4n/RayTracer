#ifndef KDTreeNode_h
#define KDTreeNode_h

#include "Mesh.h"
#include "AABB.h"

namespace rt {

class KDTreeNode {
public:
    explicit KDTreeNode(const std::vector<Mesh>& raytracables, int depth);

    std::vector<Mesh> m_raytracables;
    AABB m_volume;
    std::unique_ptr<KDTreeNode> m_left;
    std::unique_ptr<KDTreeNode> m_right;
    unsigned int m_depth;
};

}

#endif
