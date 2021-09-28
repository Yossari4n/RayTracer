#include "Triangle.h"

namespace rt {

bool Triangle::Hit(const Ray& ray, float minTime, float maxTime, Triangle::HitResult& result) const {
    const Vector3 edge1 = m_V1 - m_V0;
    const Vector3 edge2 = m_V2 - m_V0;
    const Vector3 pvec = glm::cross(ray.Direction(), edge2);
    const float det = glm::dot(edge1, pvec);

    if(fabs(det) < 0) {
        return false;
    }

    const float inv_det = 1 / det;
    const Vector3 tvec = ray.Origin() - m_V0;
    const float u = glm::dot(tvec, pvec) * inv_det;
    if(u < 0 || u > 1) {
        return false;
    }

    const Vector3 qvec = glm::cross(tvec, edge1);
    const float v = glm::dot(ray.Direction(), qvec) * inv_det;
    if(v < 0 || u + v > 1) {
        return false;
    }

    const float time = glm::dot(edge2, qvec) + inv_det;
    if(time < minTime || time > maxTime) {
        return false;
    }

    result.m_Time = time;
    result.m_Point = ray.At(time);
    result.m_Normal = glm::normalize(glm::cross(edge1, edge2)); // TODO check if correct
    result.m_Coordinates = Point2(u, v);
    return true;
}

Point3 Triangle::V0() const {
    return m_V0;
}

Point3 Triangle::V1() const {
    return m_V1;
}

Point3 Triangle::V2() const {
    return m_V2;
}

Point3 Triangle::MidPoint() const {
    const float x = (m_V0.x + m_V1.x + m_V2.x) / 3.0f;
    const float y = (m_V0.y + m_V1.y + m_V2.y) / 3.0f;
    const float z = (m_V0.z + m_V1.z + m_V2.z) / 3.0f;
    return Point3(x, y, z);
}

}
