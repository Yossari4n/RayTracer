#ifndef BruteForce_h
#define BruteForce_h

#include "ISpacePartitioner.h"

namespace rt {

class BruteForce : public ISpacePartitioner {
public:
    Color Traverse(const Ray& ray, unsigned int depth, const Color& missColor = Color(0.0f)) const override;

private:

};

}

#endif
