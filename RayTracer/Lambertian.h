#ifndef Lambertian_h
#define Lambertian_h

#include "host/IMaterial.h"

namespace rt {

class Lambertian : public IMaterial {
public:
    explicit Lambertian(const Color& albedo)
        : m_albedo(albedo) {}

    std::unique_ptr<IMaterial> Clone() const override {
        return std::make_unique<Lambertian>(m_albedo);
    }

    bool Scatter(const Ray& ray, const Triangle::HitRecord& hitRecord, IMaterial::ScatterRecord& scatterRecord) const override {
        Vector3 scatterDirection = hitRecord.m_normal + RandomUnit();

        if(NearZero(scatterDirection)) {
            scatterDirection = hitRecord.m_normal;
        }

        scatterRecord.m_scattered = Ray(hitRecord.m_point, scatterDirection);
        scatterRecord.m_attenuation = m_albedo;
        return true;
    }

    Color Emit(const Triangle::HitRecord& record) const override {
        return Color(0.0f);
    }

private:
    Color m_albedo;
};

}

#endif
