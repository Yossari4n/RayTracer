#ifndef KDTree_h
#define KDTree_h

#include "IAccelerationStructure.h"

namespace rt {

class KDTree : public IAccelerationStructure {
    struct Node {
        MeshList m_raytracables;
        AABB m_volume;

        std::unique_ptr<Node> m_left;
        std::unique_ptr<Node> m_right;
    };

public:
    void PartitionSpace(const MeshList& raytracables) override;
    Color Traverse(const Ray& ray, unsigned int depth, const Color& missColor = Color(0.0f)) const override;

private:
    void InnerPartitionSpace(Node& curr, const std::vector<Mesh>& raytracables, int depth);
    Mesh::RayTraceResult FindClosestHit(const KDTree::Node& node, const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const;

    std::unique_ptr<Node> m_root;
};

}

#endif
