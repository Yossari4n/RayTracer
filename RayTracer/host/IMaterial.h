#ifndef IMaterial_h
#define IMaterial_h

#include "../Color.h"
#include "../Math.h"
#include "../Ray.h"
#include "../Triangle.h"

namespace rt {

class IMaterial {
public:
    struct ScatterResult {
        Ray m_Scattered;
        Color m_Attenuation;
    };

    virtual ~IMaterial() = default;

    virtual bool Scatter(const Ray& ray, const Triangle::HitResult& record, ScatterResult& result) const = 0;
    virtual Color Emit(const Triangle::HitResult& record) const = 0;
};

}

#endif