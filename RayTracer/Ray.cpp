#include "Ray.h"

namespace rt {

rt::Ray::Ray(const Point3& origin, const Vector3& direction)
    : m_origin(origin)
    , m_direction(direction) {}

Vector3 rt::Ray::At(float time) const {
    return m_origin + m_direction * time;
}

Point3 rt::Ray::Origin() const {
    return m_origin;
}

Vector3 rt::Ray::Direction() const
{
    return m_direction;
}

}