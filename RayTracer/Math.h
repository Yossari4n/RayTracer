#ifndef Math_h
#define Math_h

#pragma warning(push, 0)
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#pragma warning(pop)

#include <random>

namespace rt {

using Point2 = glm::vec2;
using Point3 = glm::vec3;
using Vector2 = glm::vec2;
using Vector3 = glm::vec3;

inline bool Near(float a, float b) {
    const float diff = a - b;
    return diff >= FLT_EPSILON && diff <= FLT_EPSILON;
}

inline bool NearZero(const Vector3& vec) {
    return (fabs(vec.x) < FLT_EPSILON) && (fabs(vec.y) < FLT_EPSILON) && (fabs(vec.z) < FLT_EPSILON);
}

inline float RandomFloat() {
    static std::uniform_real_distribution<float> unif(0.0, 1.0);
    static std::default_random_engine rng;
    return unif(rng);
}

inline float RandomFloat(float min, float max) {
    return min + (max - min) * RandomFloat();
}

inline Vector3 RandomInSphere() {
    while(true) {
        const Vector3 p = Vector3(RandomFloat(-1.0, 1.0), RandomFloat(-1.0, 1.0), RandomFloat(-1.0, 1.0));
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
        const Vector3 p = Vector3(RandomFloat(-1.0, 1.0), RandomFloat(-1.0, 1.0), 0);
        if (glm::length2(p) >= 1.0)
            continue;
        return p;
    }
}

}

#endif
