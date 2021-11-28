#ifndef KDTree_h
#define KDTree_h

#include "IAccelerationStructure.h"
#include "../KDTreeNode.h"

namespace rt {

class KDTree : public IAccelerationStructure {
public:
    explicit KDTree(unsigned int maxDepth);

    void PartitionSpace(const MeshList& raytracables) override;

private:
    Mesh::RayTraceResult FindClosestHit(const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const override;
    Mesh::RayTraceResult InnerFindClosestHit(const KDTreeNode& node, const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const;

    unsigned int m_maxDepth;
    std::unique_ptr<KDTreeNode> m_root;
};

}

#endif
