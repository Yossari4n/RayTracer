#ifndef BVH_h
#define BHV_h

#include "IAccelerationStructure.h"

#include "../BVHNode.h"
#include "../AABB.h"

namespace rt {

class BVH : public IAccelerationStructure {
    struct Node {
        std::unique_ptr<Mesh> m_raytracable;

        AABB m_volume;
        std::unique_ptr<Node> m_left;
        std::unique_ptr<Node> m_right;
    };

    struct AxisXComparator {
        bool operator()(const Mesh* lhs, const Mesh* rhs) {
            return lhs->Volume().Min().x < rhs->Volume().Min().x;
        }
    };

    struct AxisYComparator {
        bool operator()(const Mesh* lhs, const Mesh* rhs) {
            return lhs->Volume().Min().y < rhs->Volume().Min().y;
        }
    };

    struct AxisZComparator {
        bool operator()(const Mesh* lhs, const Mesh* rhs) {
            return lhs->Volume().Min().z < rhs->Volume().Min().z;
        }
    };

public:
    void PartitionSpace(const MeshList& raytracables) override;

private:
    Mesh::RayTraceResult FindClosestHit(const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const override;
    Mesh::RayTraceResult InnerFindClosestHit(const BVHNode& node, const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const;

    std::unique_ptr<BVHNode> m_root;
};

}

#endif
