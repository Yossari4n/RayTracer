#ifndef AABB_h
#define AABB_h

#include "Math.h"
#include "Ray.h"

namespace rt {

class AABB {
public:
    AABB() = default;
    AABB(const Point3& min, const Point3& max);
    AABB(const AABB& lhs, const AABB& rhs);

    bool Hit(const Ray& ray, float minTime, float maxTime) const;

    Point3 Min() const;
    Point3 Max() const;
    Point3 Centroid() const;

private:
    Point3 m_min{FLT_MAX};
    Point3 m_max{FLT_MIN};
};

}

#endif
