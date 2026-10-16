cmake_minimum_required(VERSION 3.16)
project(plr_purest LANGUAGES CXX)

# PuReST engine used by plr/purest.py (both eyes in parallel, live + offline).
#   bash scripts/build_native.sh
# Requires: sudo apt install cmake g++ libopencv-dev

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc video videoio objdetect)
if(OpenCV_VERSION VERSION_GREATER_EQUAL "5.0")
  find_package(OpenCV REQUIRED COMPONENTS core imgproc video videoio objdetect geometry features xobjdetect)
endif()
find_package(Threads REQUIRED)

set(ALGO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pupil-detection-methods)
add_library(pupil_methods STATIC
  ${ALGO_DIR}/PupilDetectionMethod.cpp
  ${ALGO_DIR}/PuRe.cpp
  ${ALGO_DIR}/PuReST.cpp)
target_include_directories(pupil_methods SYSTEM PUBLIC ${ALGO_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pupil_methods PUBLIC ${OpenCV_LIBS})
target_compile_options(pupil_methods PRIVATE -w)   # third-party code

add_library(soft_flow STATIC src/soft_flow.cpp src/eye_locator.cpp)
target_include_directories(soft_flow PUBLIC src)
target_link_libraries(soft_flow PUBLIC pupil_methods)
target_compile_options(soft_flow PRIVATE -Wall -Wextra)

add_executable(plr_purest src/plr_purest.cpp)
target_link_libraries(plr_purest PRIVATE soft_flow Threads::Threads)
target_compile_options(plr_purest PRIVATE -Wall -Wextra)
target_compile_definitions(plr_purest PRIVATE PLR_CASCADE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models")
include(CTest)
if(BUILD_TESTING)
  add_executable(test_eye_locator tests/test_eye_locator.cpp)
  target_link_libraries(test_eye_locator PRIVATE soft_flow)
  add_test(NAME eye_pair_geometry COMMAND test_eye_locator)
endif()

# Pi 5 (Cortex-A76): use the native instruction set.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)
if(HAS_MCPU_NATIVE AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  target_compile_options(pupil_methods PRIVATE -mcpu=native)
  target_compile_options(plr_purest PRIVATE -mcpu=native)
  target_compile_options(soft_flow PRIVATE -mcpu=native)
endif()
