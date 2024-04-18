cmake_minimum_required(VERSION 3.13.4 FATAL_ERROR)

project(launch-hsa)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Debug or Release" FORCE)
endif()

find_package(hsa-runtime64 PATHS /opt/rocm REQUIRED)

add_library(hsa_platform hsa_platform.cpp hsa_platform.h log.h)
target_link_libraries(hsa_platform PUBLIC hsa-runtime64::hsa-runtime64)
