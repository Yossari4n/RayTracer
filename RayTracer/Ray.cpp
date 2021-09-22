#include "Ray.h"

namespace rt {

rt::Ray::Ray(const Point3& origin, const Vector3& direction)
    : m_Origin(origin)
    , m_Direction(direction) {}

Vector3 rt::Ray::At(float time) const {
    return m_Origin + m_Direction * time;
}

Point3 rt::Ray::Origin() const {
    return m_Origin;
}

Vector3 rt::Ray::Direction() const
{
    return m_Direction;
}

}