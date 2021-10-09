set(LIBRARY_NAME "${PROJECT_NAME}")

set(LIBRARY_SRC_PATH     		"${LIBRARY_MODULE_PATH}")
set(LIBRARY_PUBLIC_INCLUDE_PATH 	"${PROJECT_SOURCE_DIR}")
set(LIBRARY_PRIVATE_INCLUDE_PATH	"${LIBRARY_MODULE_PATH}")

set(
	SOURCE_LIST
	"Color.h"
	"Math.h"
	"Ray.h"
	"Ray.cpp"
	"Triangle.h"
	"Triangle.cpp"
	"AABB.h"
	"AABB.cpp"
	"Camera.h"
	"Camera.cpp"
	"Lambertian.h"
	"Lambertian.cpp"
	"DiffuseLight.h"
	"DiffuseLight.cpp"
	"DebugMaterial.h"
	"DebugMaterial.cpp"
	"Debug.h"
	"tiny_obj_loader.h"

	"host/Scene.h"
	"host/Scene.cpp"
	"host/IRayGenerator.h"
	"host/ISpacePartitioner.h"
	"host/BruteForce.h"
	"host/BruteForce.cpp"
	"host/BVH.h"
	"host/BVH.cpp"
	"host/IRenderTarget.h"
	"host/PPMTarget.h"
	"host/PPMTarget.cpp"
	"host/Mesh.h"
	"host/Mesh.cpp"
	"host/IMaterial.h"
)

add_library(
	${LIBRARY_NAME}
	${SOURCE_LIST}
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_LIST})

target_include_directories(${LIBRARY_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${LIBRARY_PUBLIC_INCLUDE_PATH}>
    PRIVATE
        ${LIBRARY_MODULE_PATH}
)

target_compile_features(${LIBRARY_NAME} PRIVATE cxx_std_17)
target_link_libraries(${LIBRARY_NAME} ${CONAN_LIBS})

set(LIBRARY_NAME ${LIBRARY_NAME} PARENT_SCOPE)