#include "DiffuseLight.h"

namespace rt {

DiffuseLight::DiffuseLight(const Color& emitted)
    : m_emitted(emitted) {}

bool DiffuseLight::Scatter(const Ray& ray, const Triangle::HitRecord& hitRecord, IMaterial::ScatterRecord& scatterRecord) const {
    return false;
}

Color DiffuseLight::Emit(const Triangle::HitRecord& record) const {
    return m_emitted;
}

}
