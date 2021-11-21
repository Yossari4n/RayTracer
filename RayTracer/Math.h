#ifndef Math_h
#define Math_h

#pragma warning(push, 0)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>
#pragma warning(pop)

#pragma warning(push, 0)
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#pragma warning(pop)

#include <random>

namespace rt {

using Point2 = glm::vec2;
using Point3 = glm::vec3;
using Vector2 = glm::vec2;
using Vector3 = glm::vec3;

#define RANDvec3 glm::vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

template <typename T>
__device__ __host__ inline void Swap(T& a, T& b) {
    T c = a;
    a = b;
    b = c;
}

template <typename T>
__device__ __host__ inline bool Near(const T& a, const T& b) {
    const float diff = a - b;
    return diff >= FLT_EPSILON && diff <= FLT_EPSILON;
}

template <typename T>
__device__ __host__ inline bool NearZero(const T& a) {
    return Near(a, 0);
}

template <>
__device__ __host__ inline bool NearZero<Vector3>(const Vector3& vec) {
    return (fabs(vec.x) < FLT_EPSILON) && (fabs(vec.y) < FLT_EPSILON) && (fabs(vec.z) < FLT_EPSILON);
}

template <class T>
inline T Random();

template <>
inline float Random<float>() {
    static std::uniform_real_distribution<float> unif(0.0, 1.0);
    static std::default_random_engine rng;
    return unif(rng);
}

template <>
inline int Random<int>() {
    static std::uniform_int_distribution<int> uni(0, 1);
    static std::default_random_engine rng;
    return uni(rng);
}

template <class T>
inline T Random(T min, T max) {
    return min + (max - min) * Random<T>();
}

inline Vector3 RandomInSphere() {
    while(true) {
        const Vector3 p = Vector3(Random(-1.0f, 1.0f), Random(-1.0f, 1.0f), Random(-1.0f, 1.0f));
        if(glm::length2(p) >= 1.0)
            continue;
        return p;
    }
}

inline Vector3 RandomUnit() {
    return glm::normalize(RandomInSphere());
}

inline Vector3 RandomInUnitDisk() {
    while (true) {
        const Vector3 p = Vector3(Random(-1.0f, 1.0f), Random(-1.0f, 1.0f), 0);
        if (glm::length2(p) >= 1.0)
            continue;
        return p;
    }
}

__device__ inline glm::vec3 RandomInUnitSphere(curandState* local_rand_state) {
    glm::vec3 p;
    do {
        p = 2.0f * RANDvec3 - glm::vec3(1, 1, 1);
    } while(glm::length(p) >= 1.0f);
    return p;
}

__device__ inline Vector3 RandomInUnitDisk(curandState* rand) {
    glm::vec3 p;
    do {
        p = 2.0f * glm::vec3(curand_uniform(rand), curand_uniform(rand), 0) - glm::vec3(1, 1, 0);
    } while(dot(p, p) >= 1.0f);
    return p;
}

}

#endif
