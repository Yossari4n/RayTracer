cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(RayTracer LANGUAGES CXX CUDA)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(RayTracer)
add_subdirectory(App)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE CONFIGURATION="$(ConfigurationName)")