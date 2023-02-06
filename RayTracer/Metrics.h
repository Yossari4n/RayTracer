#ifndef Metrics_h
#define Metrics_h

#include <chrono>

namespace rt {

class Metrics {
public:
    struct Result {
        float m_spacePartitioningTime{ 0.0f };
        float m_frameTime{ 0.0f };
        float m_saveBufferTime{ 0.0f };
        unsigned long long int m_rayCreations{ 0U };
        unsigned long long int m_volumeTests{ 0U };
        unsigned long long int m_triangleTests{ 0U };
        unsigned long long int m_triangleIntersections{ 0U };
    };

    Metrics(const Metrics&) = delete;
    Metrics& operator=(const Metrics&) = delete;
    Metrics(Metrics&&) = delete;
    Metrics& operator=(Metrics&&) = delete;

    static Metrics& Instance() {
        static Metrics instance;
        return instance;
    }

    void BeginSpacePartitioning();
    void BeginFrame();
    void BeginSaveBuffer();

    float EndSpaceParitioning();
    float EndFrame();
    float EndSaveBuffer();

    void RayCreated();
    void VolumeTested();
    void TriangleTested();
    void TriangleIntesected();

    Result Value();

private:
    Metrics() = default;

    Result m_current;
    std::chrono::steady_clock::time_point m_spacePartitioningBegin;
    std::chrono::steady_clock::time_point m_frameBegin;
    std::chrono::steady_clock::time_point m_saveBufferBegin;
};

}

#endif
