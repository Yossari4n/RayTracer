#ifndef DebugMaterial_h
#define DebugMaterial_h

#include "IMaterial.h"

namespace rt {

class DebugMaterial : public IMaterial {
public:
    bool Scatter(const Ray& ray, const Triangle::HitRecord& hitRecord, ScatterRecord& scatterRecord) const override;
    Color Emit(const Triangle::HitRecord& record) const override;
};

}

#endif
