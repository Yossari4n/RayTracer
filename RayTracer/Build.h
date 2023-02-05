#ifndef Build_h
#define Build_h

#ifdef RT_CUDA_ENABLED
    #define RT_DEVICE __device__
    #define RT_HOST __host__
#else
    #define RT_DEVICE
    #define RT_HOST
#endif

#endif