#ifndef DiffuseLight_h
#define DiffuseLight_h

#include "host/IMaterial.h"

namespace rt {

class DiffuseLight : public IMaterial {
public:
    explicit DiffuseLight(const Color& emitted);

    bool Scatter(const Ray& ray, const Triangle::HitRecord& record, IMaterial::ScatterRecord& result) const override;
    Color Emit(const Triangle::HitRecord& record) const override;

private:
    Color m_emitted;
};

}

#endif
