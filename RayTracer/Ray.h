#ifndef Ray_h
#define Ray_h

#pragma warning(push, 0)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>
#pragma warning(pop)

#include "Math.h"

namespace rt {

class Ray {
public:
    Ray() = default;

    __device__ __host__ Ray(const Point3& origin, const Vector3& direction)
        : m_origin(origin)
        , m_direction(direction) {}

    __device__ __host__ Vector3 At(float time) const { return m_origin + m_direction * time; }
    __device__ __host__ Point3 Origin() const { return m_origin; }
    __device__ __host__ Vector3 Direction() const { return m_direction; }

private:
    Point3 m_origin{0.0f};
    Vector3 m_direction{0.0f};
};

}

#endif
