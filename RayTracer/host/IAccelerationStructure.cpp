#include "IAccelerationStructure.h"
#include "..\Metrics.h"

namespace rt {

Color IAccelerationStructure::Traverse(const Ray& ray, unsigned int depth, const Color& missColor) const {
    if(depth == 0) {
        return Color(1.0f);
    }

    Metrics::Instance().RayCreated();

    Mesh::RayTraceRecord record{};
    const Mesh::RayTraceResult result = FindClosestHit(ray, 0.001f, FLT_MAX, record);
    switch(result) {
    case Mesh::RayTraceResult::Scattered:
        return record.m_emitted + record.m_attenuation * Traverse(record.m_scattered, depth - 1, missColor);
        break;

    case Mesh::RayTraceResult::Emitted:
        return record.m_emitted;
        break;

    case Mesh::RayTraceResult::Missed:
    default:
        return missColor;
        break;
    }
}

}