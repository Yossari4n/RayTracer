#ifndef BVH_h
#define BHV_h

#include "ISpacePartitioner.h"

namespace rt {

class BVH : public ISpacePartitioner {
    struct Node {
        std::unique_ptr<Mesh> m_raytracable;
        std::unique_ptr<Node> m_left;
        std::unique_ptr<Node> m_right;
    };

public:

    void PartitionSpace(const std::vector<Mesh>& raytracables) override;
    Color Traverse(const Ray& ray, unsigned int depth, const Color& missColor = Color(0.0f)) const override;

private:

};

}

#endif
