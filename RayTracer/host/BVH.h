#ifndef BVH_h
#define BHV_h

#include "ISpacePartitioner.h"

#include "../AABB.h"

namespace rt {

class BVH : public ISpacePartitioner {
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
    Color Traverse(const Ray& ray, unsigned int depth, const Color& missColor = Color(0.0f)) const override;

private:
    void InnerParitionSpace(Node& curr, std::vector<const Mesh*>& raytracablePtrs, size_t start, size_t end);
    Mesh::RayTraceResult FindClosestHit(const BVH::Node& node, const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const;

    std::unique_ptr<Node> m_root;
};

}

#endif
