#ifndef Lambertian_h
#define Lambertian_h

#include "IMaterial.h"

namespace rt {

class Lambertian : public IMaterial {
public:
    explicit Lambertian(const Color& albedo);

    bool Scatter(const Ray& ray, const Triangle::HitRecord& hitRecord, IMaterial::ScatterRecord& scatterRecord) const override;
    Color Emit(const Triangle::HitRecord& record) const override;

private:
    Color m_albedo;
};

}

#endif
