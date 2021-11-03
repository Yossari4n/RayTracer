#ifndef IMaterial_h
#define IMaterial_h

#include "../Color.h"
#include "../Math.h"
#include "../Ray.h"
#include "../Triangle.h"

#include <memory>

namespace rt {

class IMaterial {
public:
    struct ScatterRecord {
        Ray m_scattered;
        Color m_attenuation;
    };

    virtual ~IMaterial() = default;

    virtual bool Scatter(const Ray& ray, const Triangle::HitRecord& hitRecord, ScatterRecord& scatterRecord) const = 0;
    virtual Color Emit(const Triangle::HitRecord& record) const = 0;
};

}

#endif