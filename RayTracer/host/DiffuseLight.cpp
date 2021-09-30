#include "DiffuseLight.h"

namespace rt {

DiffuseLight::DiffuseLight(const Color& emitted)
    : m_Emitted(emitted) {}

bool DiffuseLight::Scatter(const Ray& ray, const Triangle::HitResult& record, IMaterial::ScatterResult& result) const {
    return false;
}

Color DiffuseLight::Emit(const Triangle::HitResult& record) const {
    return m_Emitted;
}

}
