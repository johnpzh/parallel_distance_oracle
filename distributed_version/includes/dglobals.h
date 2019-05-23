//
// Created by Zhen Peng on 5/14/19.
//

#ifndef PADO_DGLOBALS_H
#define PADO_DGLOBALS_H

#include <stdlib.h>
#include <sys/stat.h>
#include <assert.h>
#include <mpi.h>
#include <type_traits>
#include <typeinfo>
#include <iostream>

namespace PADO {

typedef uint32_t VertexID;
typedef uint64_t EdgeID;

// Get the file size
unsighed long get_file_size(char *filename)
{
    struct stat file_stat;
    assert(stat(filename, &file_stat) == 0);
    return file_stat.st_size;
}

// MPI_Instance class: for MPI initialization
// Heavily refer to Gemini (https://github.com/thu-pacman/GeminiGraph/blob/master/core/mpi.hpp)
class MPI_Instance final {
private:
    int host_id = 0; // host ID
    int num_hosts = 0; // number of hosts

public:
    MPI_Instance(int argc, char *argv[]) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &host_id);
        MPI_Comm_size(MPI_COMM_WORLD, &num_hosts);
#ifdef DEBUG_MESSAGES_ON
        if (0 == host_id) {
            printf("MPI Initialization:\n");
            printf("num_hosts: %d\n", num_hosts);
            printf("Thread support level provided by MPI: ");
            switch (provided) {
                case MPI_THREAD_SINGLE:
                    printf("MPI_THREAD_SINGLE\n");
                    break;
                case MPI_THREAD_FUNNELED:
                    printf("MPI_THREAD_FUNNELED\n");
                    break;
                case MPI_THREAD_SERIALIZED:
                    printf("MPI_THREAD_SERIALIZED\n");
                    break;
                case MPI_THREAD_MULTIPLE:
                    printf("MPI_THREAD_MULTIPLE\n");
                    break;
                default:
                    assert(false);
            }
        }
#endif
    }

    ~MPI_Instance() {
        MPI_Finalize();
    }

    template<typename T>
    static MPI_Datatype get_mpi_datatype() {
        if (std::is_same<T, char>::value) {
            return MPI_CHAR;
        } else if (std::is_same<T, unsigned char>::value) {
            return MPI_UNSIGNED_CHAR;
        } else if (std::is_same<T, short>::value) {
            return MPI_SHORT;
        } else if (std::is_same<T, unsigned short>::value) {
            return MPI_UNSIGNED_SHORT;
        } else if (std::is_same<T, int>::value) {
            return MPI_INT;
        } else if (std::is_same<T, unsigned>::value) {
            return MPI_UNSIGNED;
        } else if (std::is_same<T, long>::value) {
            return MPI_LONG;
        } else if (std::is_same<T, unsigned long>::value) {
            return MPI_UNSIGNED_LONG;
        } else if (std::is_same<T, long long>::value) {
            return MPI_LONG_LONG;
        } else if (std::is_same<T, unsigned long long>::value) {
            return MPI_UNSIGNED_LONG_LONG;
        } else if (std::is_same<T, int8_t>::value) {
            return MPI_INT8_T;
        } else if (std::is_same<T, uint8_t>::value) {
            return MPI_UINT8_T;
        } else if (std::is_same<T, int16_t>::value) {
            return MPI_INT16_T;
        } else if (std::is_same<T, uint16_t>::value) {
            return MPI_UINT16_T;
        } else if (std::is_same<T, int32_t>::value) {
            return MPI_INT32_T;
        } else if (std::is_same<T, uint32_t>::value) {
            return MPI_UINT32_T;
        } else if (std::is_same<T, int64_t>::value) {
            return MPI_INT64_T;
        } else if (std::is_same<T, uint64_t>::value) {
            return MPI_UINT64_T;
        } else if (std::is_same<T, float>::value) {
            return MPI_FLOAT;
        } else if (std::is_same<T, double>::value) {
            return MPI_DOUBLE;
        } else {
//            printf("MPI datatype not supported\n");
            printf("Error %s (%d): MPI datatype not support ", __FILE__, __LINE__);
            std::cout << typeid(T).name() << std::endl;
            exit(EXIT_FAILURE);
        }
    }

}; // End class MPI_Instance

} // End namespace PADO

#endif //PADO_DGLOBALS_H
