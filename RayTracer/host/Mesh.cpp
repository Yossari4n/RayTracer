#include "Mesh.h"
#include <iostream>

namespace rt {

Mesh::Mesh(const aiMesh* mesh) {
    for(int i = 0; i < mesh->mNumFaces; i++) {
        const aiFace* face = &mesh->mFaces[i];
        for(int j = 0; j < face->mNumIndices - 2; j += 3) {
            const Point3 v1(mesh->mVertices[face->mIndices[j]].x, mesh->mVertices[face->mIndices[j]].y, mesh->mVertices[face->mIndices[j]].z);
            const Point3 v2(mesh->mVertices[face->mIndices[j + 1]].x, mesh->mVertices[face->mIndices[j + 1]].y, mesh->mVertices[face->mIndices[j + 1]].z);
            const Point3 v3(mesh->mVertices[face->mIndices[j + 2]].x, mesh->mVertices[face->mIndices[j + 2]].y, mesh->mVertices[face->mIndices[j + 2]].z);
            m_Triangles.emplace_back(v1, v2, v3);
        }
    }
}

}
