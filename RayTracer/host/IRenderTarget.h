#ifndef IRenderTarget_h
#define IRenderTarget_h

#include "../Color.h"

namespace rt {

class IRenderTarget {
public:
    virtual ~IRenderTarget() = default;

    virtual void WriteColor(size_t x, size_t y, const Color& color, unsigned int samplesPerPixel) = 0;
    virtual void SaveBuffer() = 0;

    virtual size_t Width() const = 0;
    virtual size_t Height() const = 0;
};

}

#endif