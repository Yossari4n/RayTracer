#ifndef Mesh_h
#define Mesh_h

#pragma warning(push, 0)
#ifdef RT_CUDA_ENABLED
    #include "cuda_runtime.h"
    #include "device_launch_parameters.h"
    #include "curand_kernel.h"
    #include <curand_kernel.h>
#endif
#pragma warning(pop)

#include "Build.h"
#include "Material.h"
#include "Color.h"
#include "Ray.h"
#include "Triangle.h"
#include "AABB.h"
#include "Metrics.h"

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
        : m_triangleCount(shape.mesh.num_face_vertices.size())
        , m_triangles(new Triangle[m_triangleCount])
        , m_material(material) {

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

            m_triangles[i / 3] = Triangle(p1, p2, p3);
        }

        m_volume = AABB(min, max);
    }

    Mesh(const std::vector<Triangle>& triangles, const Material& material)
        : m_triangles(new Triangle[m_triangleCount])
        , m_triangleCount(triangles.size())
        , m_material(material) {
        std::copy(triangles.begin(), triangles.end(), m_triangles);

        Point3 min(FLT_MAX);
        Point3 max(FLT_MIN);
        for(int i = 0; i < m_triangleCount; i++) {
            const auto& triangle = triangles[i];
            min = glm::min(min, glm::min(triangle.V0(), glm::min(triangle.V1(), triangle.V2())));
            max = glm::max(max, glm::max(triangle.V0(), glm::max(triangle.V1(), triangle.V2())));
        }

        m_volume = AABB(min, max);
    }

    Mesh() = default;

    Mesh(const Mesh& rhs)
        : m_volume(rhs.m_volume)
        , m_triangleCount(rhs.m_triangleCount)
        , m_triangles(new Triangle[m_triangleCount])
        , m_material(rhs.m_material) {
        std::copy(rhs.m_triangles, rhs.m_triangles + m_triangleCount, m_triangles);
    }

    Mesh& operator=(const Mesh& rhs) {
        delete[] m_triangles;

        m_volume = rhs.m_volume;
        m_triangleCount = rhs.m_triangleCount;
        m_triangles = new Triangle[m_triangleCount];
        std::copy(rhs.m_triangles, rhs.m_triangles + m_triangleCount, m_triangles);
        m_material = Material(rhs.m_material);

        return *this;
    }

    Mesh(Mesh&& rhs) noexcept
        : m_volume(rhs.m_volume)
        , m_triangleCount(std::exchange(rhs.m_triangleCount, 0))
        , m_triangles(std::exchange(rhs.m_triangles, nullptr))
        , m_material(rhs.m_material) {}

    Mesh& operator=(Mesh&& rhs) noexcept {
        delete[] m_triangles;

        m_volume = rhs.m_volume;
        m_triangleCount = std::exchange(rhs.m_triangleCount, 0);
        m_triangles = std::exchange(rhs.m_triangles, nullptr);
        m_material = rhs.m_material;

        return *this;
    }

    ~Mesh() {
        delete[] m_triangles;
    }

    RayTraceResult RayTrace(const Ray& ray, float minTime, float maxTime, RayTraceRecord& result) const {
        Triangle::HitRecord closestHit{};
        closestHit.m_time = maxTime;

        bool hitted = false;
        for(int i = 0; i < m_triangleCount; i++) {
            const auto& triangle = m_triangles[i];

            Metrics::Instance().TriangleTested();
            if(triangle.Hit(ray, minTime, closestHit.m_time, closestHit)) {
                hitted = true;
                Metrics::Instance().TriangleIntesected();
            }
        }

        if(!hitted) {
            return RayTraceResult::Missed;
        }

        result.m_time = closestHit.m_time;
        if(Material::EmitResult emitResult{}; m_material.Emit(ray, closestHit, emitResult)) {
            result.m_emitted = emitResult.m_emitted;
        }

        if(Material::ScatterResult scatterResult{}; m_material.Scatter(ray, closestHit, scatterResult)) {
            result.m_attenuation = scatterResult.m_attenuation;
            result.m_scattered = scatterResult.m_scattered;
            return RayTraceResult::Scattered;
        }

        return RayTraceResult::Emitted;
    }

#ifdef RT_CUDA_ENABLED
    RT_DEVICE RayTraceResult RayTrace(const Ray& ray, float minTime, float maxTime, curandState* randState, RayTraceRecord& result) const {
        Triangle::HitRecord closestHit{};
        closestHit.m_time = maxTime;

        bool hitted = false;
        for(int i = 0; i < m_triangleCount; i++) {
            const auto& triangle = m_triangles[i];
            if(triangle.Hit(ray, minTime, closestHit.m_time, closestHit)) {
                hitted = true;
            }
        }

        if(!hitted) {
            return RayTraceResult::Missed;
        }

        result.m_time = closestHit.m_time;
        if(Material::EmitResult emitResult{}; m_material.Emit(ray, closestHit, randState, emitResult)) {
            result.m_emitted = emitResult.m_emitted;
        }

        if(Material::ScatterResult scatterResult{}; m_material.Scatter(ray, closestHit, randState, scatterResult)) {
            result.m_attenuation = scatterResult.m_attenuation;
            result.m_scattered = scatterResult.m_scattered;
            return RayTraceResult::Scattered;
        }

        return RayTraceResult::Emitted;
    }
#endif

    std::pair<std::optional<Mesh>, std::optional<Mesh>> Split(const Point3& splitPoint, SplitAxis axis) const {
        std::vector<Triangle> furthers;
        std::vector<Triangle> closers;

        const glm::length_t splitAxisIndex = static_cast<glm::length_t>(axis);
        for(int i = 0; i < m_triangleCount; i++) {
            const auto& triangle = m_triangles[i];
            const Point3 midPoint = triangle.MidPoint();

            if(midPoint[splitAxisIndex] < splitPoint[splitAxisIndex]) {
                furthers.push_back(triangle);
            } else {
                closers.push_back(triangle);
            }
        }

        std::optional<Mesh> further = furthers.empty() ? std::nullopt : std::make_optional<Mesh>(furthers, m_material);
        std::optional<Mesh> closer = closers.empty() ? std::nullopt : std::make_optional<Mesh>(closers, m_material);

        return std::make_pair(closer, further);
    }

    AABB Volume() const {
        return m_volume;
    }

//private:
    AABB m_volume;

    size_t m_triangleCount{0};
    Triangle* m_triangles{nullptr};

    Material m_material;
};

}

#endif
