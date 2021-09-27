#include "StaticScene.h"

namespace rt {



StaticScene::StaticScene(IRayGenerator* ray_generator, ISpacePartitioner* space_partitioner, IRenderTarget* render_target)
    : Scene(ray_generator, space_partitioner, render_target) {}

void StaticScene::GenerateFrame(unsigned int samples_per_pixel, unsigned int max_depth) const {

}

}
