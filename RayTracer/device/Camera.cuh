#ifndef Camera_cuh
#define Camera_cuh

#pragma warning(push, 0)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>
#pragma warning(pop)

#include "IRayGenerator.cuh"
#include "../Debug.h"

namespace rt::device {

class Camera : public IRayGenerator {
public:
    class DeviceCamera : public IRayGenerator::IDevice {
    public:
        __device__ DeviceCamera(const Point3& lookFrom, const Point3& lookAt, const Vector3& up, float fov, float aspectRatio, float aperture, float focusDistance)
            : m_lensRadius(aperture / 2) {
            const float theta = glm::radians(fov);
            const float h = tan(theta / 2);
            const float viewport_height = 2.0f * h;
            const float viewport_width = aspectRatio * viewport_height;

            Vector3 m_w = glm::normalize(lookFrom - lookAt);
            Vector3 m_u = glm::normalize(glm::cross(up, m_w));
            Vector3 m_v = glm::cross(m_w, m_u);

            Vector3 m_origin = lookFrom;
            Vector3 m_horizontal = focusDistance * viewport_width * m_u;
            Vector3 m_vertical = focusDistance * viewport_height * m_v;
            Vector3 m_lowerLeft = m_origin - m_horizontal * 0.5f - m_vertical * 0.5f - focusDistance * m_w;
        }

        __device__ Ray GenerateRay(float s, float t, curandState* localRandState) const override {
            const Vector3 rd = m_lensRadius * RandomInUnitDisk(localRandState);
            const Vector3 offset = m_u * rd.x + m_v * rd.y;

            return Ray(
                m_origin + offset,
                m_lowerLeft + s * m_horizontal + t * m_vertical - m_origin - offset
            );
        }

    private:
        Point3 m_origin;
        Point3 m_lowerLeft;
        Vector3 m_horizontal;
        Vector3 m_vertical;

        // Orthonomal basis
        Vector3 m_u, m_v, m_w;
        float m_lensRadius;
    };

    Camera(const Point3& lookFrom, const Point3& lookAt, const Vector3& up, float fov, float aspectRatio, float aperture, float focusDistance);
    ~Camera();

    DevicePtr ToDevice() override {
        return d_camera;
    }

private:
    DevicePtr d_camera;
};

namespace {

__global__ void CreateCameraDeviceObject(IRayGenerator::DevicePtr cameraPtr, Point3 lookFrom, Point3 lookAt, Vector3 up, float fov, float aspectRatio, float aperture, float focusDistance) {
    (*cameraPtr) = new Camera::DeviceCamera(lookFrom, lookAt, up, fov, aspectRatio, aperture, focusDistance);
}

__global__ void DeleteCameraDeviceObject(IRayGenerator::DevicePtr cameraPtr) {
    delete cameraPtr;
}

}

Camera::Camera(const Point3& lookFrom, const Point3& lookAt, const Vector3& up, float fov, float aspectRatio, float aperture, float focusDistance) {
    CHECK_CUDA(cudaMalloc((void**)&d_camera, sizeof(IDevice*)));
    CreateCameraDeviceObject << <1, 1 >> > (d_camera, lookFrom, lookAt, up, fov, aspectRatio, aperture, focusDistance);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

Camera::~Camera() {
    DeleteCameraDeviceObject<<<1, 1>>>(d_camera);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

}

#endif
