#ifndef Scene_h
#define Scene_h

#include "IRayGenerator.h"
#include "IAccelerationStructure.h"
#include "IRenderTarget.h"
#include "../Metrics.h"
#include "../Color.h"

#include <string>

namespace rt {

class Scene {
public:
    Scene(IRayGenerator* rayGenerator, IAccelerationStructure* accelerationStructure, IRenderTarget* renderTarget);

    void LoadScene(const std::string& path);
    Metrics::Result GenerateFrame(unsigned int samplesPerPixel, unsigned int maxDepth, const Color& missColor) const;

protected:
    IRayGenerator* m_rayGenerator;
    IAccelerationStructure* m_accelerationStructure;
    IRenderTarget* m_renderTarget;
};

}

#endif
