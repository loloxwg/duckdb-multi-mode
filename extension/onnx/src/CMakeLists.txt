cmake_minimum_required(VERSION 3.5)

# Set extension name here
set(TARGET_NAME onnx)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(../include)
include_directories(../onnxruntime/include)

set(EXTENSION_SOURCES onnx_extension.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

find_library(ONNXRUNTIME_LIB onnxruntime PATHS ../onnxruntime/lib)
if (NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "onnxruntime library not found!")
endif()
target_link_libraries(${EXTENSION_NAME} ${ONNXRUNTIME_LIB})

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
