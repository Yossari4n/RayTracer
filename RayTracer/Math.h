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

inline float RandomFloat() {
    static std::uniform_real_distribution<float> unif(0.0, 1.0);
    static std::default_random_engine rng;
    return unif(rng);
}

inline float RandomFloat(float min, float max) {
    return min + (max - min) * RandomFloat();
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
