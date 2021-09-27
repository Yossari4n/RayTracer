#ifndef Camera_h
#define Camera_h

#include "IRayGenerator.h"

#include "../Math.h"

namespace rt {

class Camera : public IRayGenerator {
public:
    Camera(const Point3& look_from, const Point3& look_at, const Vector3& up, float fov, float aspect_ratio, float aperture, float focus_distance);

    Ray GenerateRay(float s, float t) const override;

private:
    Point3 m_Origin;
    Point3 m_LowerLeft;
    Vector3 m_Horizontal;
    Vector3 m_Vertical;

    // Orthonomal basis
    Vector3 m_U, m_V, m_W;
    float m_LensRadius;
};

}

#endif
