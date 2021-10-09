#include "Mesh.h"

#include "../DiffuseLight.h"
#include "../Lambertian.h"
#include "../DebugMaterial.h"

#include <iostream>

namespace rt {

Mesh::Mesh(const tinyobj::attrib_t& attrib, const tinyobj::shape_t& shape, const tinyobj::material_t& material) {
    for(size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
        const auto idx1 = shape.mesh.indices[i];
        tinyobj::real_t x1 = attrib.vertices[3 * size_t(idx1.vertex_index) + 0];
        tinyobj::real_t y1 = attrib.vertices[3 * size_t(idx1.vertex_index) + 1];
        tinyobj::real_t z1 = attrib.vertices[3 * size_t(idx1.vertex_index) + 2];

        const auto idx2 = shape.mesh.indices[i + 1];
        tinyobj::real_t x2 = attrib.vertices[3 * size_t(idx2.vertex_index) + 0];
        tinyobj::real_t y2 = attrib.vertices[3 * size_t(idx2.vertex_index) + 1];
        tinyobj::real_t z2 = attrib.vertices[3 * size_t(idx2.vertex_index) + 2];

        const auto idx3 = shape.mesh.indices[i + 2];
        tinyobj::real_t x3 = attrib.vertices[3 * size_t(idx3.vertex_index) + 0];
        tinyobj::real_t y3 = attrib.vertices[3 * size_t(idx3.vertex_index) + 1];
        tinyobj::real_t z3 = attrib.vertices[3 * size_t(idx3.vertex_index) + 2];

        m_triangles.emplace_back(
            Point3(x1, y1, z1),
            Point3(x2, y2, z2),
            Point3(x3, y3, z3)
        );
    }

    if(material.name == "Light") {
        Color emmit(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
        m_material = std::make_unique<DiffuseLight>(emmit);
    } else {
        Color albedo(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
        m_material = std::make_unique<Lambertian>(albedo);
    }
}

Mesh::Mesh(const Mesh& other) 
    : m_triangles(other.m_triangles)
    , m_material(other.m_material->Clone()) {}

Mesh& Mesh::operator=(const Mesh& other)
{
    m_triangles = other.m_triangles;
    m_material = other.m_material->Clone();

    return *this;
}

Mesh::RayTraceResult Mesh::RayTrace(const Ray& ray, float minTime, float maxTime, RayTraceRecord& result) const {
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

}
