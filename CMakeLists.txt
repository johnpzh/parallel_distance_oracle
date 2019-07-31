cmake_minimum_required(VERSION 3.10)
project(pado)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# FLAGS
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_VERBOSE_MAKEFILE ON)
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_compile_options(-O0 -Wall -Wextra -pedantic -Werror)
    add_definitions(-DDEBUG_MESSAGES_ON)
    message("DEBUG ON")
    if ((${CMAKE_CXX_COMPILER} MATCHES "icpc" OR ${CMAKE_C_COMPILER} MATCHES "icc") AND (NOT (${CMAKE_C_COMPILER} MATCHES "mpicc")))
        add_compile_options(-trace -traceback -debug all)
    endif()
else()
#    add_compile_options(-O3)
    message("DEBUG OFF")
endif()

# PADO
add_executable(pado pado.cpp)
# query_distance
add_executable(query_distance query_distance.cpp)

# Headers
#include_directories(${PROJECT_SOURCE_DIR}/includes)
target_include_directories(pado PRIVATE ${PROJECT_SOURCE_DIR}/includes)
target_include_directories(query_distance PRIVATE ${PROJECT_SOURCE_DIR}/includes)

# FLAGS
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(PAPI_LIBRARIES "/usr/local/lib/libpapi.a")
    target_link_libraries(pado PRIVATE ${PAPI_LIBRARIES})
endif ()

# OpenMP
find_package(OpenMP REQUIRED)
target_compile_options(pado PRIVATE "${OpenMP_CXX_FLAGS}")
#target_link_libraries(pado PRIVATE ${OpenMP_CXX_LIBRARIES}) # Does not work for icpc
target_link_libraries(pado PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(query_distance PRIVATE "${OpenMP_CXX_FLAGS}")
target_link_libraries(query_distance PRIVATE OpenMP::OpenMP_CXX)

# Distributed Version
add_subdirectory(distributed_version)
