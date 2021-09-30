#ifndef DiffuseLight_h
#define DiffuseLight_h

#include "IMaterial.h"

namespace rt {

class DiffuseLight : public IMaterial {
public:
    explicit DiffuseLight(const Color& emitted);

    bool Scatter(const Ray& ray, const Triangle::HitResult& record, IMaterial::ScatterResult& result) const override;
    Color Emit(const Triangle::HitResult& record) const override;

private:
    Color m_Emitted;
};

}

#endif
