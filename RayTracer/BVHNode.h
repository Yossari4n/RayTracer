#ifndef BVHNode_h
#define BVHNode_h

#include "Mesh.h"
#include "AABB.h"

#include <memory>
#include <vector>

namespace rt {

class BVHNode {
    struct AxisXComparator {
        bool operator()(const Mesh& lhs, const Mesh& rhs) {
            return lhs.Volume().Min().x < rhs.Volume().Min().x;
        }
    };

    struct AxisYComparator {
        bool operator()(const Mesh& lhs, const Mesh& rhs) {
            return lhs.Volume().Min().y < rhs.Volume().Min().y;
        }
    };

    struct AxisZComparator {
        bool operator()(const Mesh& lhs, const Mesh& rhs) {
            return lhs.Volume().Min().z < rhs.Volume().Min().z;
        }
    };

public:
    BVHNode(std::vector<Mesh> raytracables);

    std::unique_ptr<Mesh> m_raytracable;
    AABB m_volume;
    std::unique_ptr<BVHNode> m_left;
    std::unique_ptr<BVHNode> m_right;
    unsigned int m_depth;

private:
    BVHNode(std::vector<Mesh>& raytracables, size_t start, size_t end);
};

}

#endif
