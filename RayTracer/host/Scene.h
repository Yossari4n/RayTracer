#ifndef Scene_h
#define Scene_h

#include "IRayGenerator.h"
#include "IAccelerationStructure.h"
#include "IRenderTarget.h"

#include <string>

namespace rt {

class Scene {
public:
    Scene(IRayGenerator* rayGenerator, IAccelerationStructure* accelerationStructure, IRenderTarget* renderTarget);

    void LoadScene(const std::string& path);
    void GenerateFrame(unsigned int samplesPerPixel, unsigned int maxDepth) const;

protected:
    IRayGenerator* m_rayGenerator;
    IAccelerationStructure* m_accelerationStructure;
    IRenderTarget* m_renderTarget;
};

}

#endif