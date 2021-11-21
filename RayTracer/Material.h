#ifndef Material_h
#define Material_h

#pragma warning(push, 0)
#include "tiny_obj_loader.h"
#pragma warning(pop)

#include "Color.h"
#include "Ray.h"
#include "Triangle.h"

namespace rt {

namespace {

const static std::string s_lightMaterialName("Light");
const static std::string s_debugMaterialName("Debug");

}

class Material {
public:
    enum class Type {
        Lambertian,
        DiffuseLight,
        Debug
    };

    struct ScatterResult {
        Ray m_scattered;
        Color m_attenuation;
    };

    struct EmitResult {
        Color m_emitted;
    };

    explicit Material(const tinyobj::material_t& material) 
        : m_albedo(material.diffuse[0], material.diffuse[1], material.diffuse[2]) {
        if(material.name == s_lightMaterialName) {
            m_type = Type::DiffuseLight;
        } else if(material.name == s_debugMaterialName) {
            m_type = Type::Debug;
        } else {
            m_type = Type::Lambertian;
        }
    }

    Material() = default;
    Material(const Material&) = default;
    Material& operator=(const Material&) = default;
    Material(Material&&) = default;
    Material& operator=(Material&&) = default;
    ~Material() = default;

    bool Scatter(const Ray& ray, const Triangle::HitRecord& hitRecord, ScatterResult& result) const {
        switch(m_type) {
        case Type::Lambertian:
            return LambertianScatter(ray, hitRecord, result);
            break;

        case Type::DiffuseLight:
            return DiffuseLightScatter(ray, hitRecord, result);
            break;

        case Type::Debug:
            return DebugScatter(ray, hitRecord, result);
            break;
        }

        return false;
    }

    __device__ bool Scatter(const Ray& ray, const Triangle::HitRecord& hitRecord, curandState* randState, ScatterResult& result) const {
        switch(m_type) {
        case Type::Lambertian:
            return LambertianScatter(ray, hitRecord, randState, result);
            break;

        case Type::DiffuseLight:
            return DiffuseLightScatter(ray, hitRecord, randState, result);
            break;

        case Type::Debug:
            return DebugScatter(ray, hitRecord, randState, result);
            break;
        }

        return false;
    }

    bool Emit(const Ray& ray, const Triangle::HitRecord& hitRecord, EmitResult& result) const {
        switch(m_type) {
        case Type::Lambertian:
            return LambertianEmit(ray, hitRecord, result);
            break;

        case Type::DiffuseLight:
            return DiffuseLightEmit(ray, hitRecord, result);
            break;

        case Type::Debug:
            return DebugEmit(ray, hitRecord, result);
            break;
        }

        return false;
    }

    __device__ bool Emit(const Ray& ray, const Triangle::HitRecord& hitRecord, curandState* randState, EmitResult& result) const {
        switch(m_type) {
        case Type::Lambertian:
            return LambertianEmit(ray, hitRecord, randState, result);
            break;

        case Type::DiffuseLight:
            return DiffuseLightEmit(ray, hitRecord, randState, result);
            break;

        case Type::Debug:
            return DebugEmit(ray, hitRecord, randState, result);
            break;
        }

        return false;
    }

private:
    // Lambertian
    bool LambertianScatter(const Ray& ray, const Triangle::HitRecord& hitRecord, ScatterResult& result) const {
        Vector3 scatterDirection = hitRecord.m_normal + RandomUnit();

        if(NearZero(scatterDirection)) {
            scatterDirection = hitRecord.m_normal;
        }

        result.m_scattered = Ray(hitRecord.m_point, scatterDirection);
        result.m_attenuation = m_albedo;
        return true;
    }

    __device__ bool LambertianScatter(const Ray& ray, const Triangle::HitRecord& hitRecord, curandState* randState, ScatterResult& result) const {
        Vector3 scatterDirection = hitRecord.m_normal + RandomInUnitSphere(randState);

        if(NearZero(scatterDirection)) {
            scatterDirection = hitRecord.m_normal;
        }

        result.m_scattered = Ray(hitRecord.m_point, scatterDirection);
        result.m_attenuation = m_albedo;
        return true;
    }

    bool LambertianEmit(const Ray& ray, const Triangle::HitRecord& hitRecord, EmitResult& result) const {
        return false;
    }

    __device__ bool LambertianEmit(const Ray& ray, const Triangle::HitRecord& hitRecord, curandState* randState, EmitResult& result) const {
        return false;
    }

    // Difuse light
    bool DiffuseLightScatter(const Ray& ray, const Triangle::HitRecord& hitRecord, ScatterResult& result) const {
        return false;
    }

    __device__ bool DiffuseLightScatter(const Ray& ray, const Triangle::HitRecord& hitRecord, curandState* randState, ScatterResult& result) const {
        return false;
    }

    bool DiffuseLightEmit(const Ray& ray, const Triangle::HitRecord& hitRecord, EmitResult& result) const {
        result.m_emitted = m_albedo;
        return true;
    }

    __device__ bool DiffuseLightEmit(const Ray& ray, const Triangle::HitRecord& hitRecord, curandState* randState, EmitResult& result) const {
        result.m_emitted = m_albedo;
        return true;
    }

    // Debug
    bool DebugScatter(const Ray& ray, const Triangle::HitRecord& hitRecord, ScatterResult& result) const {
        Vector3 scatterDirection = hitRecord.m_normal + RandomUnit();

        if(NearZero(scatterDirection)) {
            scatterDirection = hitRecord.m_normal;
        }

        result.m_scattered = Ray(hitRecord.m_point, scatterDirection);
        result.m_attenuation = Color(hitRecord.m_coordinates.x, hitRecord.m_coordinates.y, 1 - hitRecord.m_coordinates.x - hitRecord.m_coordinates.y);

        return true;
    }

    __device__ bool DebugScatter(const Ray& ray, const Triangle::HitRecord& hitRecord, curandState* randState, ScatterResult& result) const {
        Vector3 scatterDirection = hitRecord.m_normal + RandomInUnitDisk(randState);

        if(NearZero(scatterDirection)) {
            scatterDirection = hitRecord.m_normal;
        }

        result.m_scattered = Ray(hitRecord.m_point, scatterDirection);
        result.m_attenuation = Color(hitRecord.m_coordinates.x, hitRecord.m_coordinates.y, 1 - hitRecord.m_coordinates.x - hitRecord.m_coordinates.y);

        return true;
    }

    bool DebugEmit(const Ray& ray, const Triangle::HitRecord& hitRecord, EmitResult& result) const {
        return false;
    }

    __device__ bool DebugEmit(const Ray& ray, const Triangle::HitRecord& hitRecord, curandState* randState, EmitResult& result) const {
        return false;
    }

    Type m_type;
    Color m_albedo;
};

}

#endif
