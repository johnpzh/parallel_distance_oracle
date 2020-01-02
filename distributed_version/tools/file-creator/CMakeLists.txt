# createfile
add_executable(createfile createfile.cpp)

# includes
target_include_directories(createfile
        PRIVATE
        "${PROJECT_SOURCE_DIR}/distributed_version/includes"
        "${PROJECT_SOURCE_DIR}/includes")

# OpenMP
target_compile_options(createfile PRIVATE "${OpenMP_CXX_FLAGS}")
target_link_libraries(createfile PRIVATE OpenMP::OpenMP_CXX)
