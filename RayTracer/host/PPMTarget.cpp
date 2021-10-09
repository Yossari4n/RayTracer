#include "PPMTarget.h"

#include "../Debug.h"

#include <iostream>

namespace rt {

PPMTarget::PPMTarget(size_t width, size_t height)
    : m_Width(width)
    , m_Height(height) {
    m_FrameBuffer.resize(m_Width * m_Height);
}

void PPMTarget::WriteColor(size_t x, size_t y, const Color& color, unsigned int samples_per_pixel ) {
    const float scale = 1.0f / samples_per_pixel;
    const float r = sqrtf(scale * color.x);
    const float g = sqrtf(scale * color.y);
    const float b = sqrtf(scale * color.z);

    const size_t index = y * m_Width + x;
    m_FrameBuffer.at(index) = Color(r, g, b);
}

void PPMTarget::SaveBuffer() {
    LOG_INFO("Saving buffer\n");

    std::cout << "P3\n" << m_Width << ' ' << m_Height << "\n255\n";

    const size_t size = m_Width * m_Height;
    for (size_t i = 0; i < size; i++) {
        const Color color = m_FrameBuffer.at(i);

        std::cout   << static_cast<int>(256 * glm::clamp(color.r, 0.0f, 0.999f)) << ' '
                    << static_cast<int>(256 * glm::clamp(color.g, 0.0f, 0.999f)) << ' '
                    << static_cast<int>(256 * glm::clamp(color.b, 0.0f, 0.999f)) << '\n';
    }
}

size_t PPMTarget::Width() const {
    return m_Width;
}

size_t PPMTarget::Height() const {
    return m_Height;
}

}