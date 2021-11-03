#ifndef PPMTarget_h
#define PPMTarget_h

#include "IRenderTarget.h"
#include "../Debug.h"

#include <iostream>
#include <vector>

namespace rt {

class PPMTarget : public IRenderTarget {
public:
    PPMTarget(size_t width, size_t height);

    void WriteColor(size_t x, size_t y, const Color& color, unsigned int samplesPerPixel) override;
    void SaveBuffer() override;

    size_t Width() const override;
    size_t Height() const override;

private:
    size_t m_width;
    size_t m_height;

    std::vector<Color> m_frameBuffer;
};

}

#endif
