#ifndef DiffuseLight_h
#define DiffuseLight_h

#include "IMaterial.h"

namespace rt {

class DiffuseLight : public IMaterial {
public:
    explicit DiffuseLight(const Color& emitted);

    bool Scatter(const Ray& ray, const Triangle::HitRecord& hitRecord, IMaterial::ScatterRecord& scatterRecord) const override;
    Color Emit(const Triangle::HitRecord& record) const override;

private:
    Color m_emitted;
};

}

#endif
