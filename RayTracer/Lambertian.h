#ifndef Lambertian_h
#define Lambertian_h

#include "host/IMaterial.h"

namespace rt {

class Lambertian : public IMaterial {
public:
    explicit Lambertian(const Color& albedo);

    bool Scatter(const Ray& ray, const Triangle::HitRecord& record, IMaterial::ScatterRecord& result) const override;
    Color Emit(const Triangle::HitRecord& record) const override;

private:
    Color m_albedo;
};

}

#endif
