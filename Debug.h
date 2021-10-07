#ifndef Debug_h
#define Debug_h

#include <iostream>

#define LOG_ERROR(x) do { std::cout << "[" << __func__ << "][Error] " << x << '\n'; } while(0)
#define LOG_WARNING(x) do { std::cout << "[" << __func__ << "][Warning] " << x << '\n'; } while(0)
#define LOG_INFO(x) do { std::cout << "[" << __func__ << "][Info] " << x << '\n'; } while(0)

#endif
