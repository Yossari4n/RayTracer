#ifndef Triangle_h
#define Triangle_h

#include "Math.h"
#include "Ray.h"

namespace rt {

class Triangle {
public:
    struct HitResult {
        Point3 m_Point;
        Vector3 m_Normal;
        Vector2 m_Coordinates;
        float m_Time;
    };

    bool Hit(const Ray& ray, float minTime, float maxTime, HitResult& result) const;

    Point3 V0() const;
    Point3 V1() const;
    Point3 V2() const;
    Point3 MidPoint() const;

private:
    Point3 m_V0{0.0f};
    Point3 m_V1{0.0f};
    Point3 m_V2{0.0f};
};

}

#endif
