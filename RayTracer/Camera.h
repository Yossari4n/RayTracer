#ifndef Camera_h
#define Camera_h

#pragma warning(push, 0)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>
#pragma warning(pop)

#include "host/IRayGenerator.h"

#include "Math.h"

namespace rt {

class Camera : public IRayGenerator {
public:
    __host__ __device__ Camera(const Point3& lookFrom, const Point3& lookAt, const Vector3& up, float fov, float aspectRatio, float aperture, float focusDistance)
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

    __host__ Ray GenerateRay(float s, float t) const override {
        const Vector3 rd = m_lensRadius * RandomInUnitDisk();
        const Vector3 offset = m_u * rd.x + m_v * rd.y;

        return Ray(
            m_origin + offset,
            m_lowerLeft + s * m_horizontal + t * m_vertical - m_origin - offset
        );
    }

private:
    Point3 m_origin;
    Point3 m_lowerLeft;
    Vector3 m_horizontal;
    Vector3 m_vertical;

    // Orthonomal basis
    Vector3 m_u, m_v, m_w;
    float m_lensRadius;
};

}

#endif
