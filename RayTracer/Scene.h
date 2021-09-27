#ifndef Scene_h
#define Scene_h

namespace rt {

class IRayGenerator;
class ISpacePartitioner;
class IRenderTarget;

class Scene {
public:
    Scene(IRayGenerator* ray_generator, ISpacePartitioner* space_partitioner, IRenderTarget* render_target);
    virtual ~Scene() = default;

    virtual void GenerateFrame(unsigned int samples_per_pixel, unsigned int max_depth) const = 0;

protected:
    IRayGenerator* m_RayGenerator;
    ISpacePartitioner* m_SpacePartitioner;
    IRenderTarget* m_RenderTarget;
};

}

#endif