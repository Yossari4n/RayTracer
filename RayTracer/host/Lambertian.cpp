#include "Lambertian.h"

namespace rt {

rt::Lambertian::Lambertian(const Color& albedo) 
    : m_Albedo(albedo) {}

bool Lambertian::Scatter(const Ray& ray, const Triangle::HitResult& record, IMaterial::ScatterResult& result) const {
    Vector3 scatterDirection = record.m_Normal + RandomUnit();

    if(NearZero(scatterDirection)) {
        scatterDirection = record.m_Normal;
    }

    result.m_Scattered = Ray(record.m_Point, scatterDirection);
    result.m_Attenuation = m_Albedo;
    return true;
}

Color Lambertian::Emit(const Triangle::HitResult& record) const {
    return Color(0.0f);
}

}
