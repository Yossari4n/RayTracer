#include "Mesh.h"
#include <iostream>

namespace rt {

Mesh::Mesh(const tinyobj::attrib_t& attrib, const tinyobj::shape_t& shape) {
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

        m_Triangles.emplace_back(
            Point3(x1, y1, z1),
            Point3(x2, y2, z2),
            Point3(x3, y3, z3)
        );
    }

    void;
}

bool Mesh::RayTrace(const Ray& ray, float minTime, float maxTime, RayTraceResult& result) const {
    Triangle::HitResult closestHit{};
    closestHit.m_Time = maxTime;

    bool hitted = false;
    for(const auto& triangle : m_Triangles) {
        if(triangle.Hit(ray, minTime, closestHit.m_Time, closestHit)) {
            hitted = true;
        }
    }

    if(!hitted) {
        return false;
    }

    result.m_Time = closestHit.m_Time;
    result.m_Attenuation = Color(1.0f, 0.0f, 0.0f);

    return true;
}

}
