#ifndef Scene_cuh
#define Scene_cuh

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>

#include <string>

namespace rt::device {

class IRayGenerator;
class ISpacePartitioner;
class IRenderTarget;

class Scene {
public:
    __host__ Scene(IRayGenerator* rayGenerator, ISpacePartitioner* spacePartitioner, IRenderTarget* renderTarget);

    __host__ void LoadScene(const std::string& path);
    __host__ void GenerateFrame(unsigned int samplesPerPixel, unsigned int maxDepth) const;

private:
    IRayGenerator* m_rayGenerator;
    ISpacePartitioner* m_spacePartitioner;
    IRenderTarget* m_renderTarget;
};

}

#endif
