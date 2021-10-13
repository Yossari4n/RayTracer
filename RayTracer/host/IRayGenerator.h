#ifndef IRayGenerator_h
#define IRayGenerator_h

#include "../Ray.h"

namespace rt {

class IRayGenerator {
public:
    virtual ~IRayGenerator() = default;

    __host__ virtual Ray GenerateRay(float s, float t) const = 0;
};

}

#endif