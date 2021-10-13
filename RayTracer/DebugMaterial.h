#ifndef DebugMaterial_h
#define DebugMaterial_h

#include "host/IMaterial.h"

namespace rt {

class DebugMaterial : public IMaterial {
public:
    std::unique_ptr<IMaterial> Clone() const override {
        return std::make_unique<DebugMaterial>();
    }

    bool Scatter(const Ray& ray, const Triangle::HitRecord& hitRecord, ScatterRecord& scatterRecord) const override {
        Vector3 scatterDirection = hitRecord.m_normal + RandomUnit();

        if(NearZero(scatterDirection)) {
            scatterDirection = hitRecord.m_normal;
        }

        scatterRecord.m_scattered = Ray(hitRecord.m_point, scatterDirection);
        scatterRecord.m_attenuation = Color(hitRecord.m_coordinates.x, hitRecord.m_coordinates.y, 1 - hitRecord.m_coordinates.x - hitRecord.m_coordinates.y);

        return true;
    }

    Color Emit(const Triangle::HitRecord& record) const override {
        return Color(0.0f);
    }
};

}

#endif
