#include "DiffuseLight.h"

namespace rt {

DiffuseLight::DiffuseLight(const Color& emitted)
    : m_emitted(emitted) {}

std::unique_ptr<IMaterial> DiffuseLight::Clone() const {
    return std::make_unique<DiffuseLight>(m_emitted);
}

bool DiffuseLight::Scatter(const Ray& ray, const Triangle::HitRecord& hitRecord, IMaterial::ScatterRecord& scatterRecord) const {
    return false;
}

Color DiffuseLight::Emit(const Triangle::HitRecord& record) const {
    return m_emitted;
}

}