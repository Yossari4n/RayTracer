#ifndef Mesh_h
#define Mesh_h

#pragma warning(push, 0)
#include "../tiny_obj_loader.h"
#pragma warning(pop)

#include "../Color.h"
#include "../Ray.h"
#include "../Triangle.h"

#include <vector>

namespace rt {

class Mesh {
public:
    struct RayTraceResult {
        Ray m_Scattered;
        Color m_Attenuation;
        Color m_Emitted;
        float m_Time;
    };

    Mesh(const tinyobj::attrib_t& attrib, const tinyobj::shape_t& shape);

    bool RayTrace(const Ray& ray, float minTime, float maxTime, RayTraceResult& result) const;

private:
    std::vector<Triangle> m_triangles;
};

}

#endif
