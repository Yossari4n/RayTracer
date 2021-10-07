#include "Lambertian.h"

namespace rt {

rt::Lambertian::Lambertian(const Color& albedo) 
    : m_albedo(albedo) {}

bool Lambertian::Scatter(const Ray& ray, const Triangle::HitRecord& record, IMaterial::ScatterRecord& result) const {
    Vector3 scatterDirection = record.m_normal + RandomUnit();

    if(NearZero(scatterDirection)) {
        scatterDirection = record.m_normal;
    }

    result.m_Scattered = Ray(record.m_point, scatterDirection);
    result.m_Attenuation = m_albedo;
    return true;
}

Color Lambertian::Emit(const Triangle::HitRecord& record) const {
    return Color(0.0f);
}

}
