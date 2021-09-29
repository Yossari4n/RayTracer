#ifndef Mesh_h
#define Mesh_h

#pragma warning(push, 0)
#include <assimp/scene.h>
#pragma warning(pop)

#include "../Triangle.h"

#include <vector>

namespace rt {

class Mesh {
public:
    explicit Mesh(const aiMesh* mesh);

private:
    std::vector<Triangle> m_Triangles;
};

}

#endif
