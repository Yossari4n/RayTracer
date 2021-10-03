#ifndef Mesh_h
#define Mesh_h

#include "../tiny_obj_loader.h"

#include "../Triangle.h"

#include <vector>

namespace rt {

class Mesh {
public:
    Mesh(const tinyobj::attrib_t& attrib, const tinyobj::shape_t& shape);

private:
    std::vector<Triangle> m_Triangles;
};

}

#endif
