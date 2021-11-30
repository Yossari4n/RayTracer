#include "Metrics.h"

namespace rt {

void Metrics::Begin() {
    m_current = std::make_optional<Result>();
}

void Metrics::RayCreated() {
    m_current->m_rayCreations++;
}

void Metrics::VolumeTested() {
    m_current->m_volumeTests++;
}

void Metrics::TriangleTested() {
    m_current->m_triangleTests++;
}

void Metrics::TriangleIntesected() {
    m_current->m_triangleIntersections++;
}

Metrics::Result Metrics::End() {
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    m_current->m_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_begin).count() / 1000.0f;
    return std::move(m_current).value();
}

}