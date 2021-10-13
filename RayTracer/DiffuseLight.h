#ifndef DiffuseLight_h
#define DiffuseLight_h

#include "host/IMaterial.h"

namespace rt {

class DiffuseLight : public IMaterial {
public:
    explicit DiffuseLight(const Color& emitted)
        : m_emitted(emitted) {}

    std::unique_ptr<IMaterial> Clone() const override {
        return std::make_unique<DiffuseLight>(m_emitted);
    }

    bool Scatter(const Ray& ray, const Triangle::HitRecord& hitRecord, IMaterial::ScatterRecord& scatterRecord) const override {
        return false;
    }

    Color Emit(const Triangle::HitRecord& record) const override {
        return m_emitted;
    }

private:
    Color m_emitted;
};

}

#endif
