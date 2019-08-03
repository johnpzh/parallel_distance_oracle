# dpado
add_executable(dpado dpado.cpp)

# test
message("COMPILE_OPTIONS for dpado: ${COMPILE_OPTIONS}")

# includes
target_include_directories(dpado
        PRIVATE
        "${PROJECT_SOURCE_DIR}/distributed_version/includes"
        "${PROJECT_SOURCE_DIR}/includes")

# Tools
add_subdirectory(tools)

# MPI
find_package(MPI REQUIRED)
target_include_directories(dpado PRIVATE "${MPI_CXX_INCLUDE_DIRS}")
#target_compile_options(dpado PRIVATE "${MPI_CXX_LINK_FLAGS}")
target_link_libraries(dpado PRIVATE "${MPI_CXX_LIBRARIES}")

## MPI
#find_package(MPI REQUIRED)
#include_directories("${MPI_CXX_INCLUDE_DIRS}")
##message("MPI includes: ${MPI_CXX_INCLUDE_DIRS}")
#add_compile_options("${MPI_CXX_LINK_FLAGS}")
##message("MPI Flags: ${MPI_CXX_LINK_FLAGS}")
#target_link_libraries(bc PRIVATE ${MPI_CXX_LIBRARIES})
##message("MPI Libraries: ${MPI_CXX_LIBRARIES}")
#target_link_libraries(bfs PRIVATE ${MPI_CXX_LIBRARIES})
#target_link_libraries(cc PRIVATE ${MPI_CXX_LIBRARIES})
#target_link_libraries(pagerank PRIVATE ${MPI_CXX_LIBRARIES})
#target_link_libraries(sssp PRIVATE ${MPI_CXX_LIBRARIES})

# OpenMP
