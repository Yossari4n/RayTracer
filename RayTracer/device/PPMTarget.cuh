#ifndef PPMTarget_cuh
#define PPMTarget_cuh

#include "IRenderTarget.cuh"
#include "../Debug.h"

#include <vector>

namespace rt::device {

class PPMTarget : public IRenderTarget {
public:
    class DevicePPMTarget : public IRenderTarget::IDevice {
    public:
        __device__ DevicePPMTarget(size_t width, size_t height, Color* frameBuffer)
            : m_width(width)
            , m_height(height)
            , m_frameBuffer(frameBuffer) {
            //m_frameBuffer = new Color[m_width * m_height];
        }

        __device__ ~DevicePPMTarget() {
            //delete[] m_frameBuffer;
        }

        __device__ void WriteColor(size_t x, size_t y, const Color& color, unsigned int samplesPerPixel) override {
            const float scale = 1.0f / samplesPerPixel;
            const float r = sqrtf(scale * color.x);
            const float g = sqrtf(scale * color.y);
            const float b = sqrtf(scale * color.z);

            const size_t index = y * m_width + x;
            m_frameBuffer[index] = Color(r, g, b);
        }

        __device__ size_t Width() const override {
            return m_width;
        }

        __device__ size_t Height() const override {
            return m_height;
        }

        __device__ Color* FrameBuffer() override {
            return m_frameBuffer;
        }

    private:
        size_t m_width;
        size_t m_height;
        Color* m_frameBuffer;
    };

    PPMTarget(size_t width, size_t height);
    ~PPMTarget();

    void SaveBuffer() override;

    size_t Width() const override {
        return m_width;
    }

    size_t Height() const override {
        return m_height;
    }

    DevicePtr ToDevice() const override {
        return d_target;
    }

private:
    DevicePtr d_target;

    size_t m_width;
    size_t m_height;
};

namespace {

__global__ void CreatePPMTargetDeviceObject(IRenderTarget::DevicePtr d_target, size_t width, size_t height, Color* frameBuffer) {
    (*d_target) = new PPMTarget::DevicePPMTarget(width, height, frameBuffer);
}

__global__ void DeletePPMTargetDeviceObject(IRenderTarget::DevicePtr d_target) {
    delete (*d_target);
}

__global__ void CopyFrameBuffer(IRenderTarget::DevicePtr d_target, Color* frameBuffer) {
    size_t width = (*d_target)->Width();
    size_t height = (*d_target)->Height();
    memcpy(frameBuffer, (*d_target)->FrameBuffer(), sizeof(Color) * width * height);
}

}

PPMTarget::PPMTarget(size_t width, size_t height)
    : m_width(width)
    , m_height(height) {
    Color* d_frameBuffer;
    CHECK_CUDA( cudaMalloc(&d_frameBuffer, sizeof(Color) * m_width * m_height) );

    CHECK_CUDA( cudaMalloc((void**)&d_target, sizeof(IRenderTarget)) );
    CreatePPMTargetDeviceObject<<<1, 1>>>(d_target, m_width, m_height, d_frameBuffer);
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );
}

PPMTarget::~PPMTarget() {
    DeletePPMTargetDeviceObject<<<1, 1>>>(d_target);
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );
}

void PPMTarget::SaveBuffer() {
    Color* frameBuffer;
    cudaMallocManaged((void**)&frameBuffer, sizeof(Color) * m_width * m_height);

    CopyFrameBuffer<<<1, 1>>>(d_target, frameBuffer);
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );

    std::cout << "P3\n" << m_width << ' ' << m_height << "\n255\n";
    for(int i = static_cast<int>(m_width * m_height - 1); i >= 0; i--) {
        const Color color = frameBuffer[i];

        std::cout << static_cast<int>(256 * glm::clamp(color.r, 0.0f, 0.999f)) << ' '
            << static_cast<int>(256 * glm::clamp(color.g, 0.0f, 0.999f)) << ' '
            << static_cast<int>(256 * glm::clamp(color.b, 0.0f, 0.999f)) << '\n';
    }

    LOG_INFO("Saving buffer to standard output\n");
}

}

#endif
