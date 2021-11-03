#include "Lambertian.h"

namespace rt {

Lambertian::Lambertian(const Color& albedo)
    : m_albedo(albedo) {}

bool Lambertian::Scatter(const Ray& ray, const Triangle::HitRecord& hitRecord, IMaterial::ScatterRecord& scatterRecord) const {
    Vector3 scatterDirection = hitRecord.m_normal + RandomUnit();

    if(NearZero(scatterDirection)) {
        scatterDirection = hitRecord.m_normal;
    }

    scatterRecord.m_scattered = Ray(hitRecord.m_point, scatterDirection);
    scatterRecord.m_attenuation = m_albedo;
    return true;
}

Color Lambertian::Emit(const Triangle::HitRecord& record) const {
    return Color(0.0f);
}

}
