#ifndef AABB_h
#define AABB_h

#include "Build.h"

#pragma warning(push, 0)
#ifdef RT_CUDA_ENABLED
    #include "cuda_runtime.h"
    #include "device_launch_parameters.h"
    #include "curand_kernel.h"
    #include <curand_kernel.h>
#endif
#pragma warning(pop)

#include "Math.h"
#include "Ray.h"

namespace rt {

class AABB {
public:
    AABB() = default;
    RT_DEVICE RT_HOST AABB(const Point3& min, const Point3& max)
        : m_min(min)
        , m_max(max) {}

    RT_DEVICE RT_HOST AABB(const AABB& lhs, const AABB& rhs)
        : m_min(glm::min(lhs.Min(), rhs.Min()))
        , m_max(glm::max(lhs.Max(), rhs.Max())) {}

    RT_DEVICE RT_HOST bool Hit(const Ray& ray, float minTime, float maxTime) const {
        auto axis_test = [&](float min, float max, float origin, float direction) -> bool {
            const float inv_d = 1.0f / direction;
            float t0 = (min - origin) * inv_d;
            float t1 = (max - origin) * inv_d;

            if(inv_d < 0.0) {
                Swap(t0, t1);
            }

            minTime = t0 > minTime ? t0 : minTime;
            maxTime = t1 < maxTime ? t1 : maxTime;
            return maxTime >= minTime;
        };

        return axis_test(m_min.x, m_max.x, ray.Origin().x, ray.Direction().x)
            && axis_test(m_min.y, m_max.y, ray.Origin().y, ray.Direction().y)
            && axis_test(m_min.z, m_max.z, ray.Origin().z, ray.Direction().z);
    }

    RT_DEVICE RT_HOST Point3 Centroid() const {
        const float x = (m_min.x + m_max.x) / 2.0f;
        const float y = (m_min.y + m_max.y) / 2.0f;
        const float z = (m_min.z + m_max.z) / 2.0f;
        return Point3(x, y, z);
    }

    RT_DEVICE RT_HOST Point3 Min() const { return m_min; }
    RT_DEVICE RT_HOST Point3 Max() const { return m_max; }

private:
    Point3 m_min{FLT_MAX};
    Point3 m_max{FLT_MIN};
};

}

#endif
