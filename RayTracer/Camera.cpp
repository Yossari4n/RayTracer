#include "Camera.h"

namespace rt{

Camera::Camera(const Point3& lookFrom, const Point3& lookAt, const Vector3& up, float fov, float aspectRatio, float aperture, float focusDistance)
    : m_lensRadius(aperture / 2) {
    const float theta = glm::radians(fov);
    const float h = tan(theta / 2);
    const float viewport_height = 2.0f * h;
    const float viewport_width = aspectRatio * viewport_height;

    m_w = glm::normalize(lookFrom - lookAt);
    m_u = glm::normalize(glm::cross(up, m_w));
    m_v = glm::cross(m_w, m_u);

    m_origin = lookFrom;
    m_horizontal = focusDistance * viewport_width * m_u;
    m_vertical = focusDistance * viewport_height * m_v;
    m_lowerLeft = m_origin - m_horizontal * 0.5f - m_vertical * 0.5f - focusDistance * m_w;
}

Ray Camera::GenerateRay(float s, float t) const {
    const Vector3 rd = m_lensRadius * RandomInUnitDisk();
    const Vector3 offset = m_u * rd.x + m_v * rd.y;

    return Ray(
        m_origin + offset, 
        m_lowerLeft + s * m_horizontal + t * m_vertical - m_origin - offset
    );
}

}
