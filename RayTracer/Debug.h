#ifndef Debug_h
#define Debug_h

#include <iostream>

#define LOG_ERROR(x) do { std::cerr << "\r[" << __func__ << "][Error] " << x; } while(0)
#define LOG_WARNING(x) do { std::cerr << "\r[" << __func__ << "][Warning] " << x; } while(0)
#define LOG_INFO(x) do { std::cerr << "\r[" << __func__ << "][Info] " << x; } while(0)

#endif
