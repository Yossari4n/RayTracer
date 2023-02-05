cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(RayTracer LANGUAGES CUDA CXX)

option(ENABLE_CUDA "" OFF)

if(ENABLE_CUDA)
	include(CheckLanguage)
	check_language(CUDA)
	if(CMAKE_CUDA_COMPILER)
		enable_language(CUDA)
	add_compile_definitions(RT_CUDA_ENABLED)
	else(CMAKE_CUDA_COMPILER)
		message(STATUS "No CUDA compiler found")
	endif(CMAKE_CUDA_COMPILER)
endif(ENABLE_CUDA)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(RayTracer)
add_subdirectory(App)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE CONFIGURATION="$(ConfigurationName)")