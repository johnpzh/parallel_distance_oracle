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

target_compile_options(
        remove_equivalent_vertices
        PRIVATE
#        -xCORE-AVX512
        -march=native
#        -ftree-vectorizer-verbose=5
#        -fopt-info-vec-missed
#        -fopt-info-vec-all
#        -fopt-info-vec-optimized
#        -march=skylake-avx512
#        -march=broadwell
#        -mavx512f
#        -mavx512cd
#        -mavx512bw
#        -mavx512dq
#        -mavx512vl
#        -mavx2
#        -mavx
)


# OpenMP
target_compile_options(edgelist2binary PRIVATE "${OpenMP_CXX_FLAGS}")
target_link_libraries(edgelist2binary PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(binary2edgelist PRIVATE "${OpenMP_CXX_FLAGS}")
target_link_libraries(binary2edgelist PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(remove_equivalent_vertices PRIVATE "${OpenMP_CXX_FLAGS}")
target_link_libraries(remove_equivalent_vertices PRIVATE OpenMP::OpenMP_CXX)