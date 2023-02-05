#ifndef Ray_h
#define Ray_h

#pragma warning(push, 0)
#ifdef RT_CUDA_ENABLED
    #include "cuda_runtime.h"
    #include "device_launch_parameters.h"
    #include "curand_kernel.h"
    #include <curand_kernel.h>
#endif
#pragma warning(pop)

#include "Build.h"
#include "Math.h"

namespace rt {

class Ray {
public:
    Ray() = default;

    RT_DEVICE RT_HOST Ray(const Point3& origin, const Vector3& direction)
        : m_origin(origin)
        , m_direction(direction) {}

    RT_DEVICE RT_HOST Vector3 At(float time) const { return m_origin + m_direction * time; }
    RT_DEVICE RT_HOST Point3 Origin() const { return m_origin; }
    RT_DEVICE RT_HOST Vector3 Direction() const { return m_direction; }

private:
    Point3 m_origin{0.0f};
    Vector3 m_direction{0.0f};
};

}

#endif
