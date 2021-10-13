#ifndef Triangle_h
#define Triangle_h

#pragma warning(push, 0)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>
#pragma warning(pop)

#include "Math.h"
#include "Ray.h"

namespace rt {

class Triangle {
public:
    struct HitRecord {
        Point3 m_point;
        Vector3 m_normal;
        Vector2 m_coordinates;
        float m_time;
    };

    __host__ __device__ Triangle(const Point3& v1, const Point3& v2, const Point3& v3)
        : m_v0(v1)
        , m_v1(v2)
        , m_v2(v3) {}

    __host__ __device__ bool Hit(const Ray& ray, float minTime, float maxTime, HitRecord& record) const {
        const Vector3 edge1 = m_v1 - m_v0;
        const Vector3 edge2 = m_v2 - m_v0;
        const Vector3 pvec = glm::cross(ray.Direction(), edge2);
        const float det = glm::dot(edge1, pvec);

        if(fabs(det) < 0)
            return false;

        const float inv_det = 1 / det;

        const Vector3 tvec = ray.Origin() - m_v0;
        const float u = glm::dot(tvec, pvec) * inv_det;
        if(u < 0 || u > 1)
            return false;

        const Vector3 qvec = glm::cross(tvec, edge1);
        const float v = glm::dot(ray.Direction(), qvec) * inv_det;
        if(v < 0 || u + v > 1)
            return false;

        const float time = glm::dot(edge2, qvec) * inv_det;
        if(time < minTime || time > maxTime)
            return false;

        record.m_time = time;
        record.m_point = ray.At(time);
        record.m_normal = glm::normalize(glm::cross(edge1, edge2));
        record.m_coordinates = Point2(u, v);
        return true;
    }

    __host__ __device__ Point3 MidPoint() const {
        const float x = (m_v0.x + m_v1.x + m_v2.x) / 3.0f;
        const float y = (m_v0.y + m_v1.y + m_v2.y) / 3.0f;
        const float z = (m_v0.z + m_v1.z + m_v2.z) / 3.0f;
        return Point3(x, y, z);
    }

    __host__ __device__ Point3 V0() const { return m_v0; }
    __host__ __device__ Point3 V1() const { return m_v0; }
    __host__ __device__ Point3 V2() const { return m_v0; }

private:
    Point3 m_v0{0.0f};
    Point3 m_v1{0.0f};
    Point3 m_v2{0.0f};
};

}

#endif
