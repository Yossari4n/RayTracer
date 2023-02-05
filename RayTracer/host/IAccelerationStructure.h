#ifndef IAccelerationStructure_h
#define IAccelerationStructure_h

#include "../Color.h"
#include "../Ray.h"
#include "../Mesh.h"

namespace rt {

class IAccelerationStructure {
public:
    using MeshList = std::vector<Mesh>;

    virtual ~IAccelerationStructure() = default;

    virtual void PartitionSpace(const MeshList& raytracables) = 0;
    
    Color Traverse(const Ray& ray, unsigned int depth, const Color& missColor = Color(0.0f)) const;

protected:
    virtual Mesh::RayTraceResult FindClosestHit(const Ray& ray, float minTime, float maxTime, Mesh::RayTraceRecord& record) const = 0;
};

}

#endif
