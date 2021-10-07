#ifndef Triangle_h
#define Triangle_h

#include "Math.h"
#include "Ray.h"

namespace rt {

class Triangle {
public:
    struct HitRecord {
        Point3 m_point;
        Vector3 m_normal;
        Vector2 m_coordinates;
        float m_time;
    };

    Triangle(const Point3& v1, const Point3& v2, const Point3& v3);

    bool Hit(const Ray& ray, float minTime, float maxTime, HitRecord& record) const;

    Point3 V0() const;
    Point3 V1() const;
    Point3 V2() const;
    Point3 MidPoint() const;

private:
    Point3 m_v0{0.0f};
    Point3 m_v1{0.0f};
    Point3 m_v2{0.0f};
};

}

#endif
