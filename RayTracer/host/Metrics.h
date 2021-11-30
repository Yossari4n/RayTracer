#ifndef Metrics_h
#define Metrics_h

#include <optional>
#include <chrono>

namespace rt {

class Metrics {
public:
    struct Result {
        float m_time{ 0U };
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

    void Begin();
    void RayCreated();
    void VolumeTested();
    void TriangleTested();
    void TriangleIntesected();
    Result End();

private:
    Metrics() = default;

    std::optional<Result> m_current;
    std::chrono::steady_clock::time_point m_begin;
};

}

#endif
