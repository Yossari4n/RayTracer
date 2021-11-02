#ifndef BruteForce_h
#define BruteForce_h

#include "IAccelerationStructure.h"

#include <optional>

namespace rt {

class BruteForce : public IAccelerationStructure {
public:
    void PartitionSpace(const MeshList& raytracables) override;

private:
    Mesh::RayTraceResult FindClosestHit(const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const override;

    MeshList m_rayTracables;
};

}

#endif
