#include "PPMTarget.h"

namespace rt {

PPMTarget::PPMTarget(size_t width, size_t height)
    : m_width(width)
    , m_height(height) {
    m_FrameBuffer.reserve(m_width * m_height);
}

void PPMTarget::WriteColor(size_t x, size_t y, const Color& color, unsigned int samplesPerPixel) {
    const float scale = 1.0f / samplesPerPixel;
    const float r = sqrtf(scale * color.x);
    const float g = sqrtf(scale * color.y);
    const float b = sqrtf(scale * color.z);

    const size_t index = y * m_width + x;
    m_FrameBuffer[index] = Color(r, g, b);
}

void PPMTarget::SaveBuffer() {
    LOG_INFO("Saving buffer\n");

    std::cout << "P3\n" << m_width << ' ' << m_height << "\n255\n";

    const size_t size = m_width * m_height;
    for(size_t i = 0; i < size; i++) {
        const Color color = m_FrameBuffer[i];

        std::cout << static_cast<int>(256 * glm::clamp(color.r, 0.0f, 0.999f)) << ' '
            << static_cast<int>(256 * glm::clamp(color.g, 0.0f, 0.999f)) << ' '
            << static_cast<int>(256 * glm::clamp(color.b, 0.0f, 0.999f)) << '\n';
    }
}


size_t PPMTarget::Width() const {
    return m_width;
}

size_t PPMTarget::Height() const {
    return m_height;
}

}
