# edgelist2binary
add_executable(edgelist2binary edgelist2binary.cpp)
# binary2edgelist
add_executable(binary2edgelist binary2edgelist.cpp)

# includes
target_include_directories(edgelist2binary
        PRIVATE
        "${PROJECT_SOURCE_DIR}/distributed_version/includes"
        "${PROJECT_SOURCE_DIR}/includes")
target_include_directories(binary2edgelist
        PRIVATE
        "${PROJECT_SOURCE_DIR}/distributed_version/includes"
        "${PROJECT_SOURCE_DIR}/includes")