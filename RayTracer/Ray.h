#ifndef Ray_h
#define Ray_h

#include "Math.h"

namespace rt {

class Ray {
public:
    Ray() = default;
    Ray(const Point3& origin, const Vector3& direction);

    Vector3 At(float time) const;

    Point3 Origin() const;
    Vector3 Direction() const;

private:
    Point3 m_Origin{0.0f};
    Vector3 m_Direction{0.0f};
};

}

#endif
