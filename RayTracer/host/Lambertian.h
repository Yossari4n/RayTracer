#ifndef Lambertian_h
#define Lambertian_h

#include "IMaterial.h"

namespace rt {

class Lambertian : public IMaterial {
public:
    explicit Lambertian(const Color& albedo);

    bool Scatter(const Ray& ray, const Triangle::HitResult& record, IMaterial::ScatterResult& result) const override;
    Color Emit(const Triangle::HitResult& record) const override;

private:
    Color m_Albedo;
};

}

#endif // !Lambertian_h
