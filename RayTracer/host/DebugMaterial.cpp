#include "DebugMaterial.h"

namespace rt {

bool DebugMaterial::Scatter(const Ray& ray, const Triangle::HitRecord& hitRecord, ScatterRecord& scatterRecord) const {
    Vector3 scatterDirection = hitRecord.m_normal + RandomUnit();

    if(NearZero(scatterDirection)) {
        scatterDirection = hitRecord.m_normal;
    }

    scatterRecord.m_scattered = Ray(hitRecord.m_point, scatterDirection);
    scatterRecord.m_attenuation = Color(hitRecord.m_coordinates.x, hitRecord.m_coordinates.y, 1 - hitRecord.m_coordinates.x - hitRecord.m_coordinates.y);

    return true;
}

Color DebugMaterial::Emit(const Triangle::HitRecord& record) const {
    return Color(0.0f);
}

}
