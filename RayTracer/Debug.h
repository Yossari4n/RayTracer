#ifndef Debug_h
#define Debug_h

#pragma warning(push, 0)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>
#pragma warning(pop)

#include <iostream>

#define LOG_ERROR(...) do { fprintf(stderr, "\r[%s][Error] ", __func__); fprintf(stderr, __VA_ARGS__); } while(0)
#define LOG_WARNING( ...) do { fprintf(stderr, "\r[%s][Warning] ", __func__); fprintf(stderr, __VA_ARGS__); } while(0)
#define LOG_INFO(...) do { fprintf(stderr, "\r[%s][Info] ", __func__); fprintf(stderr, __VA_ARGS__); } while(0)
#define LOG_FLUSH() do { fflush(stderr); } while(0)

inline void CheckCUDA(cudaError_t result, char const* const func, const char* const file, int const line) {
    if(result) {
        std::cerr << "\nCUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";

        cudaDeviceReset();
        exit(99);
    }
}

#define CHECK_CUDA(val) CheckCUDA( (val), #val, __FILE__, __LINE__ )

#endif
