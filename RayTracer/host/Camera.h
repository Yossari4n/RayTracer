#ifndef Camera_h
#define Camera_h

#include "IRayGenerator.h"

#include "Math.h"

namespace rt {

class Camera : public IRayGenerator {
public:
    Camera(const Point3& lookFrom, const Point3& lookAt, const Vector3& up, float fov, float aspectRatio, float aperture, float focusDistance);

    Ray GenerateRay(float s, float t) const override;

private:
    Point3 m_origin;
    Point3 m_lowerLeft;
    Vector3 m_horizontal;
    Vector3 m_vertical;

    // Orthonomal basis
    Vector3 m_u, m_v, m_w;
    float m_lensRadius;
};

}

#endif
