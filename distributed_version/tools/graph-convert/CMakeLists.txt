# edgelist2binary
add_executable(edgelist2binary edgelist2binary.cpp)
# binary2edgelist
add_executable(binary2edgelist binary2edgelist.cpp)
# remove_equivalent_vertices
add_executable(remove_equivalent_vertices remove_equivalent_vertices.cpp)

# includes
target_include_directories(
        edgelist2binary
        PRIVATE
        "${PROJECT_SOURCE_DIR}/distributed_version/includes"
        "${PROJECT_SOURCE_DIR}/includes")
target_include_directories(
        binary2edgelist
        PRIVATE
        "${PROJECT_SOURCE_DIR}/distributed_version/includes"
        "${PROJECT_SOURCE_DIR}/includes")
target_include_directories(
        remove_equivalent_vertices
        PRIVATE
        "${PROJECT_SOURCE_DIR}/distributed_version/includes"
        "${PROJECT_SOURCE_DIR}/includes")

# OpenMP
target_compile_options(edgelist2binary PRIVATE "${OpenMP_CXX_FLAGS}")
target_link_libraries(edgelist2binary PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(binary2edgelist PRIVATE "${OpenMP_CXX_FLAGS}")
target_link_libraries(binary2edgelist PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(remove_equivalent_vertices PRIVATE "${OpenMP_CXX_FLAGS}")
target_link_libraries(remove_equivalent_vertices PRIVATE OpenMP::OpenMP_CXX)