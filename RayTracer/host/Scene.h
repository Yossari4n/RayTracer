#ifndef Scene_h
#define Scene_h

#include <string>

namespace rt {

class IRayGenerator;
class ISpacePartitioner;
class IRenderTarget;

class Scene {
public:
    Scene(IRayGenerator* rayGenerator, ISpacePartitioner* spacePartitioner, IRenderTarget* renderTarget);

    void LoadScene(const std::string& path);
    void GenerateFrame(unsigned int samplesPerPixel, unsigned int maxDepth) const;

protected:
    IRayGenerator* m_rayGenerator;
    ISpacePartitioner* m_spacePartitioner;
    IRenderTarget* m_renderTarget;
};

}

#endif