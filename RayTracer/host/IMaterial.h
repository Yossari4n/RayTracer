#ifndef IMaterial_h
#define IMaterial_h

#include "../Color.h"
#include "../Math.h"
#include "../Ray.h"
#include "../Triangle.h"

namespace rt {

class IMaterial {
public:
    struct ScatterRecord {
        Ray m_Scattered;
        Color m_Attenuation;
    };

    virtual ~IMaterial() = default;

    virtual bool Scatter(const Ray& ray, const Triangle::HitRecord& record, ScatterRecord& result) const = 0;
    virtual Color Emit(const Triangle::HitRecord& record) const = 0;
};

}

#endif