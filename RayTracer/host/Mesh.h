#ifndef Mesh_h
#define Mesh_h

#pragma warning(push, 0)
#include "../tiny_obj_loader.h"
#pragma warning(pop)

#include "IMaterial.h"
#include "../Color.h"
#include "../Ray.h"
#include "../Triangle.h"

#include <vector>
#include <memory>

namespace rt {

class Mesh {
public:
    enum class RayTraceResult {
        Missed,
        Scattered,
        Emitted
    };

    struct RayTraceRecord {
        Ray m_scattered;
        Color m_attenuation;
        Color m_emitted;
        float m_time;
    };

    Mesh(const tinyobj::attrib_t& attrib, const tinyobj::shape_t& shape, const tinyobj::material_t& materials);

    Mesh() = delete;
    Mesh(const Mesh& other);
    Mesh& operator=(const Mesh& other);
    Mesh(Mesh&& other) = default;
    Mesh& operator=(Mesh&& other) = default;

    RayTraceResult RayTrace(const Ray& ray, float minTime, float maxTime, RayTraceRecord& result) const;

private:
    std::vector<Triangle> m_triangles;
    std::unique_ptr<IMaterial> m_material;
};

}

#endif
