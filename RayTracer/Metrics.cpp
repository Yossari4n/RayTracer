#include "Metrics.h"

namespace rt {

void Metrics::BeginSpacePartitioning() {
    m_spacePartitioningBegin = std::chrono::steady_clock::now();
}

void Metrics::BeginFrame() {
    m_frameBegin = std::chrono::steady_clock::now();
}

void Metrics::BeginSaveBuffer()
{
    m_saveBufferBegin = std::chrono::steady_clock::now();
}

float Metrics::EndSpaceParitioning()
{
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    m_current.m_spacePartitioningTime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end - m_spacePartitioningBegin).count());
    return m_current.m_spacePartitioningTime;
}

float Metrics::EndFrame()
{
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    m_current.m_frameTime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end - m_frameBegin).count());
    return m_current.m_frameTime;
}

float Metrics::EndSaveBuffer()
{
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    m_current.m_saveBufferTime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end - m_saveBufferBegin).count());
    return m_current.m_saveBufferTime;
}

void Metrics::RayCreated() {
    m_current.m_rayCreations++;
}

void Metrics::VolumeTested() {
    m_current.m_volumeTests++;
}

void Metrics::TriangleTested() {
    m_current.m_triangleTests++;
}

void Metrics::TriangleIntesected() {
    m_current.m_triangleIntersections++;
}

Metrics::Result Metrics::Value() {
    return m_current;
}

}