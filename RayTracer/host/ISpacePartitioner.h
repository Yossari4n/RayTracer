#ifndef ISpacePartitioner_h
#define ISpacePartitioner_h

#include "../Color.h"
#include "../Ray.h"

namespace rt {

class ISpacePartitioner {
public:
    virtual ~ISpacePartitioner() = default;

    virtual Color Traverse(const Ray& ray, unsigned int depth, const Color& missColor = Color(0.0f)) const = 0;
};

}

#endif
