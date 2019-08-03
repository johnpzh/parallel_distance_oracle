//
// Created by Zhen Peng on 5/23/19.
//

#ifndef PADO_MPI_DPADO_H
#define PADO_MPI_DPADO_H

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <iostream>
#include <type_traits>
#include <typeinfo>

namespace PADO {

enum MessageTags {
//    GRAPH_SHUFFLE,
//    SIZE_GRAPH_SHUFFLE,
//    SENDING_NUM_ROOT_MASTERS,
//    SENDING_ROOT_ID,
//    SENDING_INDEXTYPE_BATCHES,
//    SENDING_INDEXTYPE_DISTANCES,
//    SENDING_INDEXTYPE_VERTICES,
//    SENDING_PUSHED_LABELS,
//    SENDING_MASTERS_TO_MIRRORS,
//    SENDING_SIZE_MASTERS_TO_MIRRORS,
//	SENDING_DIST_TABLE,
//    SENDING_SIZE_DIST_TABLE,
//    SYNC_DIST_TABLE,
//    SYNC_SIZE_DIST_TABLE,
    SENDING_QUERY_LABELS,
    SENDING_SIZE_QUERY_LABELS,
//    SENDING_BP_ACTIVES,
//    SENDING_SIZE_BP_ACTIVES,
//    SENDING_SETS_UPDATES_BP,
//    SENDING_SIZE_SETS_UPDATES_BP,
    SENDING_ROOT_NEIGHBORS,
    SENDING_SIZE_ROOT_NEIGHBORS,
//    SENDING_SIZE_ROOT_BP_LABELS,
    SENDING_SELECTED_NEIGHBORS,
    SENDING_SIZE_SELETED_NEIGHBORS,
//    SENDING_ROOT_BP_LABELS,
    SENDING_QUERY_BP_LABELS,
//    SENDING_NUM_UNIT_BUFFERS,
    SENDING_BUFFER_SEND,
    SENDING_SIZE_BUFFER_SEND,
    SENDING_EDGELIST,
    SENDING_INDICES,
    SENDING_SIZE_INDICES,
    SENDING_LABELS,
    SENDING_SIZE_LABELS
};

// MPI_Instance class: for MPI initialization
// Heavily refer to Gemini (https://github.com/thu-pacman/GeminiGraph/blob/master/core/mpi.hpp)
class MPI_Instance final {
private:
//    int host_id = 0; // host ID
//    int num_hosts = 0; // number of hosts
//    static const uint32_t UNIT_BUFFER_SIZE = 16U << 20U;
//    static char unit_buffer_send[UNIT_BUFFER_SIZE];

public:
    MPI_Instance(int argc, char *argv[]) {
        int host_id = 0; // host ID
        int num_hosts = 0; // number of hosts
        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
		//printf("MPI_Init_thread\n");
        MPI_Comm_rank(MPI_COMM_WORLD, &host_id);
		//printf("MPI_Comm_rank\n");
        MPI_Comm_size(MPI_COMM_WORLD, &num_hosts);
		//printf("MPI_Comm_size\n");
//#ifdef DEBUG_MESSAGES_ON
		char hostname[256];
		gethostname(hostname, sizeof(hostname));
		printf("host_id: %d num_hosts: %d HostName: %s PID: %d\n", host_id, num_hosts, hostname, getpid());
        if (0 == host_id) {
            printf("MPI Initialization:\n");
//            printf("num_hosts: %d\n", num_hosts);
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
//#endif
		if (0 == host_id) {
			printf("MPI initialized.\n");
		}
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

//    // DEPRECATED Function: Receive MPI message with the dynamic buffer size from any source with a certain tag.
//    // 1. Use MPI_Probe to get the message source and size.
//    // 2. Allocate the buffer_recv according to the Status.
//    // 3. Use MPI_Recv to receive the message from the source.
//    template <typename E_T>
//    static int receive_dynamic_buffer_from_any(
//            std::vector<E_T> &buffer_recv,
//            int num_hosts,
//            int message_tag)
//    {
//        size_t ETypeSize = sizeof(E_T);
//        MPI_Status status_prob;
//        MPI_Probe(MPI_ANY_SOURCE,
//                  message_tag,
//                  MPI_COMM_WORLD,
//                  &status_prob);
//        int source_host_id = status_prob.MPI_SOURCE;
//        assert(source_host_id >=0 && source_host_id < num_hosts);
//        int bytes_recv;
//        MPI_Get_count(&status_prob, MPI_CHAR, &bytes_recv);
//        assert(bytes_recv % ETypeSize == 0);
//        int num_e_recv = bytes_recv / ETypeSize;
//        buffer_recv.resize(num_e_recv);
//		MPI_Status status_recv;
//        MPI_Recv(buffer_recv.data(),
//                 bytes_recv,
//                 MPI_CHAR,
//                 source_host_id,
//                 message_tag,
//                 MPI_COMM_WORLD,
//                 &status_recv);
//		assert(status_prob.MPI_SOURCE == status_recv.MPI_SOURCE);
////		{// test
////			if (!buffer_recv.empty()) {
////				printf("@%u host_id: %u receive_dynamic_buffer_from_source: source_host_id: %u buffer_recv[0]: ", __LINE__, host_id, source_host_id);
////				std::cout << buffer_recv[0] << std::endl;
////			}
////		}
//        return source_host_id;
//    }

//    // DEPRECATED Function: Receive MPI message with dynamic buffer size from a certain source with a certain tag.
//    // 1. User MPI_Probe to get the message size.
//    // 2. Allocate the buffer_recv according to the Status.
//    // 3. Use MPI_Recv to receive the message.
//    template <typename E_T>
//    static int receive_dynamic_buffer_from_source(
//            std::vector<E_T> &buffer_recv,
//            int num_hosts,
//            int source,
//            int message_tag)
//    {
//        assert(source >= 0 && source < num_hosts);
//        size_t ETypeSize = sizeof(E_T);
//        MPI_Status status_recv;
//        MPI_Probe(source,
//                  message_tag,
//                  MPI_COMM_WORLD,
//                  &status_recv);
//        int bytes_recv;
//        MPI_Get_count(&status_recv, MPI_CHAR, &bytes_recv);
//        assert(bytes_recv % ETypeSize == 0);
//        int num_e_recv = bytes_recv / ETypeSize;
//        buffer_recv.resize(num_e_recv);
//        MPI_Recv(buffer_recv.data(),
//                 bytes_recv,
//                 MPI_CHAR,
//                 source,
//                 message_tag,
//                 MPI_COMM_WORLD,
//                 MPI_STATUS_IGNORE);
//        return num_e_recv;
//    }

    // Function: return the size (bytes) of the sending buffer.
    // It's equal to the length of the buffer times the size of one element.
    template <typename EdgeT>
    static size_t get_sending_size(
            const std::vector<EdgeT> &buffer_send)
    {
        return buffer_send.size() * sizeof(EdgeT);
    }

    // Function: send the large-sized buffer_send by multiple unit sending.
    template <typename E_T>
    static void send_buffer_2_dst(
            const std::vector<E_T> &buffer_send,
            int dst,
            int message_tag,
            int size_message_tag)
    {
        const size_t ETypeSize = sizeof(E_T);
        size_t size_buffer_send = buffer_send.size();
        // Send the total size
        MPI_Send(&size_buffer_send,
                 1,
                 MPI_UINT64_T,
                 dst,
                 size_message_tag,
                 MPI_COMM_WORLD);
        if (!size_buffer_send) {
            return;
        }
        uint64_t bytes_buffer_send = size_buffer_send * ETypeSize;
        if (bytes_buffer_send < static_cast<size_t>(INT_MAX)) {
            // Only need 1 send
            MPI_Send(buffer_send.data(),
                     size_buffer_send * ETypeSize,
                     MPI_CHAR,
                     dst,
                     message_tag,
                     MPI_COMM_WORLD);
        } else {
            const uint32_t num_unit_buffers = ((bytes_buffer_send - 1) / static_cast<size_t>(INT_MAX)) + 1;
            const uint64_t unit_buffer_size = ((size_buffer_send - 1) / num_unit_buffers) + 1;
            for (uint64_t b_i = 0; b_i < num_unit_buffers; ++b_i) {
                size_t offset = b_i * unit_buffer_size;
                size_t size_unit_buffer = b_i == num_unit_buffers - 1
                                          ? size_buffer_send - offset
                                          : unit_buffer_size;
                MPI_Send(buffer_send.data() + offset,
                         size_unit_buffer * ETypeSize,
                         MPI_CHAR,
                         dst,
                         message_tag,
                         MPI_COMM_WORLD);
            }
        }
//        /////////////////////////////////////////////////
//        //
//        // Send by multiple unit buffers
//        uint32_t num_unit_buffers = (size_buffer_send + UNIT_BUFFER_SIZE - 1) / UNIT_BUFFER_SIZE;
//        if (1 == num_unit_buffers) {
//            MPI_Send(buffer_send.data(),
//                    size_buffer_send * ETypeSize,
//                    MPI_CHAR,
//                    dst,
//                    message_tag,
//                    MPI_COMM_WORLD);
//            return;
//        }
//        for (uint32_t b_i = 0; b_i < num_unit_buffers; ++b_i) {
//            size_t offset = b_i * UNIT_BUFFER_SIZE;
//            size_t size_unit_buffer = b_i == num_unit_buffers - 1
//                                      ? size_buffer_send - offset
//                                      : UNIT_BUFFER_SIZE;
//            assert(size_unit_buffer * ETypeSize <= static_cast<size_t>(INT_MAX));
//            MPI_Send(buffer_send.data() + offset,
//                     size_unit_buffer * ETypeSize,
//                     MPI_CHAR,
//                     dst,
//                     message_tag,
//                     MPI_COMM_WORLD);
//        }
//        //
//        /////////////////////////////////////////////////
    }

    // Function: receive data (from source) into large-sized buffer_recv by receiving multiple unit sending.
    template <typename E_T>
    static void recv_buffer_from_src(
            std::vector<E_T> &buffer_recv,
            int src,
            int message_tag,
            int size_message_tag)
    {
        const size_t ETypeSize = sizeof(E_T);
        size_t size_buffer_send;
        MPI_Recv(&size_buffer_send,
                 1,
                 MPI_UINT64_T,
                 src,
                 size_message_tag,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        buffer_recv.resize(size_buffer_send);
        if (!size_buffer_send) {
            return;
        }
        uint64_t bytes_buffer_send = size_buffer_send * ETypeSize;
        if (bytes_buffer_send < static_cast<size_t>(INT_MAX)) {
            // Only need 1 receive
            MPI_Recv(buffer_recv.data(),
                     bytes_buffer_send,
                     MPI_CHAR,
                     src,
                     message_tag,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        } else {
            const uint32_t num_unit_buffers = ((bytes_buffer_send - 1) / static_cast<size_t>(INT_MAX)) + 1;
            const uint64_t unit_buffer_size = ((size_buffer_send - 1) / num_unit_buffers) + 1;
            for (uint64_t b_i = 0; b_i < num_unit_buffers; ++b_i) {
                size_t offset = b_i * unit_buffer_size;
                size_t size_unit_buffer = b_i == num_unit_buffers - 1
                                          ? size_buffer_send - offset
                                          : unit_buffer_size;
                MPI_Recv(buffer_recv.data() + offset,
                         size_unit_buffer * ETypeSize,
                         MPI_CHAR,
                         src,
                         message_tag,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        }
//        /////////////////////////////////////////////////
//        //
//        // Receive multiple unit buffers
//        uint32_t num_unit_buffers = (size_buffer_send + UNIT_BUFFER_SIZE - 1) / UNIT_BUFFER_SIZE;
//        if (1 == num_unit_buffers) {
//            MPI_Recv(buffer_recv.data(),
//                    size_buffer_send * ETypeSize,
//                    MPI_CHAR,
//                    src,
//                    message_tag,
//                    MPI_COMM_WORLD,
//                    MPI_STATUS_IGNORE);
//            return;
//        }
//        for (uint32_t b_i = 0; b_i < num_unit_buffers; ++b_i) {
//            size_t offset = b_i * UNIT_BUFFER_SIZE;
//            size_t size_unit_buffer = b_i == num_unit_buffers - 1
//                                      ? size_buffer_send - offset
//                                      : UNIT_BUFFER_SIZE;
//            MPI_Recv(buffer_recv.data() + offset,
//                     size_unit_buffer * ETypeSize,
//                     MPI_CHAR,
//                     src,
//                     message_tag,
//                     MPI_COMM_WORLD,
//                     MPI_STATUS_IGNORE);
//        }
//        //
//        /////////////////////////////////////////////////
    }

    // Function: receive data (from any source) into large-sized buffer_recv by receiving multiple unit sending.
    template <typename E_T>
    static int recv_buffer_from_any(
            std::vector<E_T> &buffer_recv,
            int message_tag,
            int size_message_tag)
    {
        const size_t ETypeSize = sizeof(E_T);
        size_t size_buffer_send;
        MPI_Status status_recv;
        MPI_Recv(&size_buffer_send,
                 1,
                 MPI_UINT64_T,
                 MPI_ANY_SOURCE,
                 size_message_tag,
                 MPI_COMM_WORLD,
                 &status_recv);
        buffer_recv.resize(size_buffer_send);
        int src = status_recv.MPI_SOURCE;
        if (!size_buffer_send) {
            return src;
        }
        uint64_t bytes_buffer_send = size_buffer_send * ETypeSize;
        if (bytes_buffer_send < static_cast<size_t>(INT_MAX)) {
            // Only need 1 receive
            MPI_Recv(buffer_recv.data(),
                     bytes_buffer_send,
                     MPI_CHAR,
                     src,
                     message_tag,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            return src;
        } else {
            const uint32_t num_unit_buffers = ((bytes_buffer_send - 1) / static_cast<size_t>(INT_MAX)) + 1;
            const uint64_t unit_buffer_size = ((size_buffer_send - 1) / num_unit_buffers) + 1;
            for (uint64_t b_i = 0; b_i < num_unit_buffers; ++b_i) {
                size_t offset = b_i * unit_buffer_size;
                size_t size_unit_buffer = b_i == num_unit_buffers - 1
                                          ? size_buffer_send - offset
                                          : unit_buffer_size;
                MPI_Recv(buffer_recv.data() + offset,
                         size_unit_buffer * ETypeSize,
                         MPI_CHAR,
                         src,
                         message_tag,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        }
//        /////////////////////////////////////////////////
//        //
//        // Receive multiple unit buffers
//        uint32_t num_unit_buffers = (size_buffer_send + UNIT_BUFFER_SIZE - 1) / UNIT_BUFFER_SIZE;
//        if (1 == num_unit_buffers) {
//            MPI_Recv(buffer_recv.data(),
//                     size_buffer_send * ETypeSize,
//                     MPI_CHAR,
//                     src,
//                     message_tag,
//                     MPI_COMM_WORLD,
//                     MPI_STATUS_IGNORE);
//            return src;
//        }
//        for (uint32_t b_i = 0; b_i < num_unit_buffers; ++b_i) {
//            size_t offset = b_i * UNIT_BUFFER_SIZE;
//            size_t size_unit_buffer = b_i == num_unit_buffers - 1
//                                      ? size_buffer_send - offset
//                                      : UNIT_BUFFER_SIZE;
//            MPI_Recv(buffer_recv.data() + offset,
//                     size_unit_buffer * ETypeSize,
//                     MPI_CHAR,
//                     src,
//                     message_tag,
//                     MPI_COMM_WORLD,
//                     MPI_STATUS_IGNORE);
//        }
//        //
//        /////////////////////////////////////////////////
        return src;
    }

//    // DEPRECATED Function
//    template<typename E_T>
//    static void recv_buffer_from_source(std::vector<E_T> &buffer_recv,
//                                     int source,
//                                     int message_tag,
//                                     int size_message_tag)
//    {
//        size_t ETypeSize = sizeof(E_T);
//        size_t bytes_buffer;
//        uint32_t num_unit_buffers;
//        uint32_t size_unit_buffer;
//        // Receive the message about size.
//        // .first: bytes_buffer_send
//        // .second: num_unit_buffers
//        {
//            std::pair<size_t, uint32_t> size_msg;
//            MPI_Status status_recv;
//            MPI_Recv(&size_msg,
//                     sizeof(size_msg),
//                     MPI_CHAR,
//                     source,
//                     size_message_tag,
//                     MPI_COMM_WORLD,
//                     &status_recv);
//            bytes_buffer = size_msg.first;
//            size_unit_buffer = size_msg.second;
//            num_unit_buffers = (bytes_buffer + size_unit_buffer - 1) / size_unit_buffer;
//        }
//        assert(0 == bytes_buffer % ETypeSize);
//        buffer_recv.resize(bytes_buffer / ETypeSize);
//        // Receive the whole data
//        if (num_unit_buffers) {
//            size_t offset = 0;
//            // Except for the last one, all unit buffer's size is fixed.
//            for (uint32_t b_i = 0; b_i < num_unit_buffers - 1; ++b_i) {
//                MPI_Recv(reinterpret_cast<char *>(buffer_recv.data()) + offset,
//                         size_unit_buffer,
//                         MPI_CHAR,
//                         source,
//                         message_tag,
//                         MPI_COMM_WORLD,
//                         MPI_STATUS_IGNORE);
//                offset += size_unit_buffer;
//            }
//            // The last unit buffer
//            MPI_Recv(reinterpret_cast<char *>(buffer_recv.data()) + offset,
//                     bytes_buffer - offset,
//                     MPI_CHAR,
//                     source,
//                     message_tag,
//                     MPI_COMM_WORLD,
//                     MPI_STATUS_IGNORE);
//        }
//    }

//    template <typename E_T, typename F>
//    static void every_host_bcasts_buffer(std::vector<E_T> &buffer_send,
//            int num_hosts,
//            F &fun)
//    {
//        for (int h_i = 0; h_i < num_hosts; ++h_i) {
//            uint64_t size_buffer_send = buffer_send.size();
//            // Sync the size_buffer_send.
//            MPI_Bcast(&size_buffer_send,
//                    1,
//                    MPI_UINT64_T,
//                    h_i,
//                    MPI_COMM_WORLD);
//            uint32_t num_unit_buffers = (size_buffer_send + UNIT_BUFFER_SIZE - 1) / UNIT_BUFFER_SIZE;
//
//            for (uint32_t b_i = 0; b_i < num_unit_buffers; ++b_i) {
//                size_t offset = b_i * UNIT_BUFFER_SIZE;
//                uint32_t size_unit_buffer = b_i == num_unit_buffers - 1
//                                            ? size_buffer_send - offset
//                                            : UNIT_BUFFER_SIZE;
//                std::vector<E_T> unit_buffer(size_unit_buffer);
//                if (host_id == h_i) {
//
//                }
//            }
////            size_t size_unit_buffer;
////            // Broadcast the size of buffer_send
////            // Broadcast buffer_send to buffer_recv
////            std::vector<E_T> unit_buffer(size_unit_buffer);
////            // Process every element of buffer_recv by fun
////            for (const E_T &e : unit_buffer) {
////                fun(e);
////            }
//        }
//    }
}; // End class MPI_Instance
//char MPI_Instance::unit_buffer_send[UNIT_BUFFER_SIZE];
} // End namespace PADO
#endif //PADO_MPI_DPADO_H
