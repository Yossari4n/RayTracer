#ifndef MaterialFactory_h
#define MaterialFactory_h

#pragma warning(push, 0)
#include "../tiny_obj_loader.h"
#pragma warning(pop)

#include "IMaterial.h"

#include <string>
#include <map>

namespace rt {

class MaterialFactory {
public:
    IMaterial* CreateMaterial(const tinyobj::material_t& material);

private:
    static std::string s_lightMaterialName;
    static std::string s_debugMaterialName;

    std::map<std::string, std::unique_ptr<IMaterial>> m_materials;
};

}

#endif
