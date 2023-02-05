set(APP_NAME "App")

set(APP_MODULE_DIRECTORY "${PROJECT_SOURCE_DIR}/App")
set(APP_SOURCE_DIRECTORY "${APP_MODULE_DIRECTORY}")

add_executable(
	${APP_NAME}
	main.cu
)

target_link_libraries(${APP_NAME} ${LIBRARY_NAME})