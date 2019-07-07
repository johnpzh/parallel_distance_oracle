//
// Created by Zhen Peng on 5/23/19.
//

#ifndef PADO_MPI_DPADO_H
#define PADO_MPI_DPADO_H

#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <type_traits>
#include <typeinfo>

namespace PADO {

enum MessageTags {
    GRAPH_SHUFFLE,
    SENDING_NUM_ROOT_MASTERS,
    SENDING_ROOT_ID,
    SENDING_INDEXTYPE_BATCHES,
    SENDING_INDEXTYPE_DISTANCES,
    SENDING_INDEXTYPE_VERTICES,
//    SENDING_PUSHED_LABELS,
    SENDING_MASTERS_TO_MIRRORS,
	SENDING_DIST_TABLE,
    SYNC_DIST_TABLE,
    SENDING_QUERY_LABELS,
    SENDING_BP_ACTIVES,
    SENDING_SETS_UPDATES_BP,
    SENDING_ROOT_NEIGHBORS,
    SENDING_SELECTED_NEIGHBORS,
    SENDING_ROOT_BP_LABELS
};

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

    // Function: get the corresponding data type in MPI (MPI_Datatype)
    template <typename T>
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

    // Function: Receive MPI message with the dynamic buffer size from any source with a certain tag.
    // 1. Use MPI_Probe to get the message source and size.
    // 2. Allocate the buffer_recv according to the Status.
    // 3. Use MPI_Recv to receive the message from the source.
    template <typename E_T>
    static int receive_dynamic_buffer_from_any(std::vector<E_T> &buffer_recv, int num_hosts, int message_tag)
    {
        size_t ETypeSize = sizeof(E_T);
        MPI_Status status_prob;
        MPI_Probe(MPI_ANY_SOURCE,
                  message_tag,
                  MPI_COMM_WORLD,
                  &status_prob);
        int source_host_id = status_prob.MPI_SOURCE;
        assert(source_host_id >=0 && source_host_id < num_hosts);
        int bytes_recv;
        MPI_Get_count(&status_prob, MPI_CHAR, &bytes_recv);
        assert(bytes_recv % ETypeSize == 0);
        int num_e_recv = bytes_recv / ETypeSize;
        buffer_recv.resize(num_e_recv);
		MPI_Status status_recv;
        MPI_Recv(buffer_recv.data(),
                 bytes_recv,
                 MPI_CHAR,
                 source_host_id,
                 message_tag,
                 MPI_COMM_WORLD,
                 &status_recv);
		assert(status_prob.MPI_SOURCE == status_recv.MPI_SOURCE);
//		{// test
//			if (!buffer_recv.empty()) {
//				printf("@%u host_id: %u receive_dynamic_buffer_from_source: source_host_id: %u buffer_recv[0]: ", __LINE__, host_id, source_host_id);
//				std::cout << buffer_recv[0] << std::endl;
//			}
//		}
        return source_host_id;
    }

    // Function: Receive MPI message with dynamic buffer size from a certain source with a certain tag.
    // 1. User MPI_Probe to get the message size.
    // 2. Allocate the buffer_recv according to the Status.
    // 3. Use MPI_Recv to receive the message.
    template <typename E_T>
    static int receive_dynamic_buffer_from_source(std::vector<E_T> &buffer_recv,
            int num_hosts,
            int source,
            int message_tag)
    {
        assert(source >= 0 && source < num_hosts);
        size_t ETypeSize = sizeof(E_T);
        MPI_Status status_recv;
        MPI_Probe(source,
                  message_tag,
                  MPI_COMM_WORLD,
                  &status_recv);
        int bytes_recv;
        MPI_Get_count(&status_recv, MPI_CHAR, &bytes_recv);
        assert(bytes_recv % ETypeSize == 0);
        int num_e_recv = bytes_recv / ETypeSize;
        buffer_recv.resize(num_e_recv);
        MPI_Recv(buffer_recv.data(),
                 bytes_recv,
                 MPI_CHAR,
                 source,
                 message_tag,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        return num_e_recv;
    }

    // Function: return the size (bytes) of the sending buffer.
    // It's equal to the length of the buffer times the size of one element.
    template <typename EdgeT>
    static size_t get_sending_size(const std::vector<EdgeT> &buffer_send)
    {
        return buffer_send.size() * sizeof(EdgeT);
    }

}; // End class MPI_Instance
} // End namespace PADO
#endif //PADO_MPI_DPADO_H
