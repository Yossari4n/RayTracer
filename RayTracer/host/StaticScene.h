#ifndef StaticScene_h
#define StaticScene_h

#include "../Scene.h"

namespace rt {

class StaticScene : public Scene {
public:
    StaticScene(IRayGenerator* ray_generator, ISpacePartitioner* space_partitioner, IRenderTarget* render_target);

    void GenerateFrame(unsigned int samples_per_pixel, unsigned int max_depth) const override;

private:
};

}

#endif
