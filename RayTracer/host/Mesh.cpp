#include "Mesh.h"
#include <iostream>

namespace rt {

Mesh::Mesh(const tinyobj::attrib_t& attrib, const tinyobj::shape_t& shape) {
    const size_t size = shape.mesh.num_face_vertices.size();

    for(size_t j = 0; j < size; j++) {
        tinyobj::index_t idx = shape.mesh.indices[j];
        tinyobj::real_t x1 = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
        tinyobj::real_t y1 = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
        tinyobj::real_t z1 = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

        tinyobj::real_t x2 = attrib.vertices[3 * size_t(idx.vertex_index) + 3];
        tinyobj::real_t y2 = attrib.vertices[3 * size_t(idx.vertex_index) + 4];
        tinyobj::real_t z2 = attrib.vertices[3 * size_t(idx.vertex_index) + 5];

        tinyobj::real_t x3 = attrib.vertices[3 * size_t(idx.vertex_index) + 6];
        tinyobj::real_t y3 = attrib.vertices[3 * size_t(idx.vertex_index) + 7];
        tinyobj::real_t z3 = attrib.vertices[3 * size_t(idx.vertex_index) + 8];

        m_Triangles.emplace_back(
            Point3(x1, y1, z1),
            Point3(x2, y2, z2),
            Point3(x3, y3, z3)
        );
    }
}

}
