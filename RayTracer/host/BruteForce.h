#ifndef BruteForce_h
#define BruteForce_h

#include "IAccelerationStructure.h"

#include <optional>

namespace rt {

class BruteForce : public IAccelerationStructure {
public:
    void PartitionSpace(const MeshList& raytracables) override;
    Color Traverse(const Ray& ray, unsigned int depth, const Color& missColor = Color(0.0f)) const override;

private:
    Mesh::RayTraceResult FindClosestHit(const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const;

    MeshList m_rayTracables;
};

}

#endif
