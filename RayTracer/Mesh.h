#ifndef Mesh_h
#define Mesh_h

#pragma warning(push, 0)
#include "tiny_obj_loader.h"
#pragma warning(pop)

#include "host/IMaterial.h"
#include "Color.h"
#include "Ray.h"
#include "Triangle.h"
#include "AABB.h"

#include "host/DiffuseLight.h"
#include "host/Lambertian.h"
#include "host/DebugMaterial.h"

#include <vector>
#include <memory>
#include <optional>
#include <string>

namespace rt {

class Mesh {
public:
    enum class RayTraceResult {
        Missed,
        Scattered,
        Emitted
    };

    enum class SplitAxis {
        X,
        Y,
        Z
    };

    struct RayTraceRecord {
        Ray m_scattered;
        Color m_attenuation;
        Color m_emitted;
        float m_time;
    };

    Mesh(const tinyobj::attrib_t& attrib, const tinyobj::shape_t& shape, const tinyobj::material_t& material)
        : m_name(shape.name) {
        Point3 min(FLT_MAX);
        Point3 max(FLT_MIN);
        for(size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
            const tinyobj::index_t idx1 = shape.mesh.indices[i];
            const tinyobj::real_t x1 = attrib.vertices[3 * size_t(idx1.vertex_index) + 0];
            const tinyobj::real_t y1 = attrib.vertices[3 * size_t(idx1.vertex_index) + 1];
            const tinyobj::real_t z1 = attrib.vertices[3 * size_t(idx1.vertex_index) + 2];
            const Point3 p1(x1, y1, z1);

            const tinyobj::index_t idx2 = shape.mesh.indices[i + 1];
            const tinyobj::real_t x2 = attrib.vertices[3 * size_t(idx2.vertex_index) + 0];
            const tinyobj::real_t y2 = attrib.vertices[3 * size_t(idx2.vertex_index) + 1];
            const tinyobj::real_t z2 = attrib.vertices[3 * size_t(idx2.vertex_index) + 2];
            const Point3 p2(x2, y2, z2);

            const tinyobj::index_t idx3 = shape.mesh.indices[i + 2];
            const tinyobj::real_t x3 = attrib.vertices[3 * size_t(idx3.vertex_index) + 0];
            const tinyobj::real_t y3 = attrib.vertices[3 * size_t(idx3.vertex_index) + 1];
            const tinyobj::real_t z3 = attrib.vertices[3 * size_t(idx3.vertex_index) + 2];
            const Point3 p3(x3, y3, z3);

            min = glm::min(min, glm::min(p1, glm::min(p2, p3)));
            max = glm::max(max, glm::max(p1, glm::max(p2, p3)));

            m_triangles.emplace_back(p1, p2, p3);
        }

        m_volume = AABB(min, max);

        if(material.name == "Light") {
            Color emmit(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
            m_material = std::make_unique<DiffuseLight>(emmit);
        } else {
            Color albedo(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
            m_material = std::make_unique<Lambertian>(albedo);
        }
    }

    Mesh(const std::vector<Triangle>& triangles, std::unique_ptr<IMaterial> material, const std::string& name)
    : m_triangles(triangles)
        , m_material(std::move(material))
        , m_name(name) {
        Point3 min(FLT_MAX);
        Point3 max(FLT_MIN);
        for(const auto& triangle : m_triangles) {
            min = glm::min(min, glm::min(triangle.V0(), glm::min(triangle.V1(), triangle.V2())));
            max = glm::max(max, glm::max(triangle.V0(), glm::max(triangle.V1(), triangle.V2())));
        }

        m_volume = AABB(min, max);
    }

    Mesh() = delete;

    Mesh(const Mesh& other)
        : m_volume(other.m_volume)
        , m_triangles(other.m_triangles)
        , m_material(other.m_material->Clone())
        , m_name(other.m_name) {}

    Mesh& operator=(const Mesh& other) {
        m_volume = other.m_volume;
        m_triangles = other.m_triangles;
        m_material = other.m_material->Clone();
        m_name = other.m_name;

        return *this;
    }

    Mesh(Mesh&& other) = default;
    Mesh& operator=(Mesh&& other) = default;

    RayTraceResult RayTrace(const Ray& ray, float minTime, float maxTime, RayTraceRecord& result) const {
        Triangle::HitRecord closestHit{};
        closestHit.m_time = maxTime;

        bool hitted = false;
        for(const auto& triangle : m_triangles) {
            if(triangle.Hit(ray, minTime, closestHit.m_time, closestHit)) {
                hitted = true;
            }
        }

        if(!hitted) {
            return RayTraceResult::Missed;
        }

        result.m_time = closestHit.m_time;
        result.m_emitted = m_material->Emit(closestHit);

        if(IMaterial::ScatterRecord record{}; m_material->Scatter(ray, closestHit, record)) {
            result.m_attenuation = record.m_attenuation;
            result.m_scattered = record.m_scattered;
            return RayTraceResult::Scattered;
        }

        return RayTraceResult::Emitted;
    }

    std::pair<std::optional<Mesh>, std::optional<Mesh>> Split(const Point3& splitPoint, SplitAxis axis) const {
        std::vector<Triangle> furthers;
        std::vector<Triangle> closers;

        const glm::length_t splitAxisIndex = static_cast<glm::length_t>(axis);
        for(const auto& triangle : m_triangles) {
            const Point3 midPoint = triangle.MidPoint();

            if(midPoint[splitAxisIndex] < splitPoint[splitAxisIndex]) {
                furthers.push_back(triangle);
            } else {
                closers.push_back(triangle);
            }
        }

        std::optional<Mesh> further = furthers.empty() ? std::nullopt : std::make_optional<Mesh>(furthers, std::move(m_material->Clone()), m_name);
        std::optional<Mesh> closer = closers.empty() ? std::nullopt : std::make_optional<Mesh>(closers, std::move(m_material->Clone()), m_name);

        return std::make_pair(closer, further);
    }

    AABB Volume() const {
        return m_volume;
    }

private:
    AABB m_volume;
    std::vector<Triangle> m_triangles;
    std::unique_ptr<IMaterial> m_material;
    std::string m_name;
};

}

#endif
