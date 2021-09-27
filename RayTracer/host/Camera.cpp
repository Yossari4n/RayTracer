#include "Camera.h"

namespace rt{

Camera::Camera(const Point3& look_from, const Point3& look_at, const Vector3& up, float fov, float aspect_ratio, float aperture, float focus_distance)
    : m_LensRadius(aperture / 2) {
    const float theta = glm::radians(fov);
    const float h = tan(theta / 2);
    const float viewport_height = 2.0f * h;
    const float viewport_width = aspect_ratio * viewport_height;

    m_W = glm::normalize(look_from - look_at);
    m_U = glm::normalize(glm::cross(up, m_W));
    m_V = glm::cross(m_W, m_U);

    m_Origin = look_from;
    m_Horizontal = focus_distance * viewport_width * m_U;
    m_Vertical = focus_distance * viewport_height * m_V;
    m_LowerLeft = m_Origin - m_Horizontal * 0.5f - m_Vertical * 0.5f - focus_distance * m_W;
}

Ray Camera::GenerateRay(float s, float t) const {
    const Vector3 rd = m_LensRadius * RandomInUnitDisk();
    const Vector3 offset = m_U * rd.x + m_V * rd.y;

    return Ray(
        m_Origin + offset, 
        m_LowerLeft + s * m_Horizontal + t * m_Vertical - m_Origin - offset
    );
}

}
