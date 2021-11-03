#include "MaterialFactory.h"

#include "DiffuseLight.h"
#include "Lambertian.h"
#include "DebugMaterial.h"

namespace rt {

std::string MaterialFactory::s_lightMaterialName = "Light";
std::string MaterialFactory::s_debugMaterialName = "Debug";

IMaterial* MaterialFactory::CreateMaterial(const tinyobj::material_t& material) {
    if(auto it = m_materials.find(material.name); it == m_materials.end()) {
        if(material.name == s_lightMaterialName) {
            m_materials.emplace(s_lightMaterialName, std::make_unique<DiffuseLight>(Color(material.diffuse[0], material.diffuse[1], material.diffuse[2])));
        } else if(material.name == s_debugMaterialName) {
            m_materials.emplace(s_debugMaterialName, std::make_unique<DebugMaterial>());
        } else {
            m_materials.emplace(material.name, std::make_unique<Lambertian>(Color(material.diffuse[0], material.diffuse[1], material.diffuse[2])));
        }
    }

    return m_materials[material.name].get();
}

}