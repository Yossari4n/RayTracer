#ifndef Mesh_h
#define Mesh_h

#pragma warning(push, 0)
#include "../tiny_obj_loader.h"
#pragma warning(pop)

#include "IMaterial.h"
#include "../Color.h"
#include "../Ray.h"
#include "../Triangle.h"
#include "../AABB.h"

#include <vector>
#include <memory>
#include <optional>

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

    Mesh(const tinyobj::attrib_t& attrib, const tinyobj::shape_t& shape, const tinyobj::material_t& materials);
    Mesh(const std::vector<Triangle>& triangles, std::unique_ptr<IMaterial> material);

    Mesh() = delete;
    Mesh(const Mesh& other);
    Mesh& operator=(const Mesh& other);
    Mesh(Mesh&& other) = default;
    Mesh& operator=(Mesh&& other) = default;

    RayTraceResult RayTrace(const Ray& ray, float minTime, float maxTime, RayTraceRecord& result) const;
    std::pair<std::optional<Mesh>, std::optional<Mesh>> Split(const Point3& splitPoint, SplitAxis axis) const;

    AABB Volume() const;

private:
    AABB m_volume;
    std::vector<Triangle> m_triangles;
    std::unique_ptr<IMaterial> m_material;
};

}

#endif
