//
// Created by Zhen Peng on 5/14/19.
//


#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include "dglobals.h"
#include "dgraph.h"
//#include "dpado.h"

//#include "dpado.202001011226.openmp.h"
#include "dpado.202001060737.multiple_rounds_for_memory.h"
//#include "dpado.202001071517.push_back_ShortIndex.h"

using namespace PADO;


void dpado(char *argv[])
{
    DistGraph G(argv[1]);
	printf("host_id: %u num_masters: %u /%u %.2f%% num_edges_local %lu /%lu %.2f%%\n",
	        G.host_id, G.num_masters, G.num_v, 100.0 * G.num_masters / G.num_v, G.num_edges_local, 2 * G.num_e, 100.0 * G.num_edges_local / (2 * G.num_e));//test

    int num_runs = 1;
	for (int i = 0; i < num_runs; ++i) {
//        DistBVCPLL<1024, 50> *dist_bvcpll = new DistBVCPLL<1024, 50>(G); // batch size 1024, bit-parallel size 50.
//        delete dist_bvcpll;
//		double mem_for_graph;
//		double memtotal;
//		{// Get the memory foot print.
//			double memfree;
//			Utils::system_memory(memtotal, memfree);
//			mem_for_graph = memtotal - memfree;
//		}
//        {
//            NUM_THREADS = 20;
//            omp_set_num_threads(NUM_THREADS);
//            DistBVCPLL<128> *dist_bvcpll = new DistBVCPLL<128>(G); // batch size 128, bit-parallel size 50.
//            delete dist_bvcpll;
//        }
//        {
//            NUM_THREADS = 20;
//            omp_set_num_threads(NUM_THREADS);
//            DistBVCPLL<256> *dist_bvcpll = new DistBVCPLL<256>(G); // batch size 128, bit-parallel size 50.
//            delete dist_bvcpll;
//        }
//        {
//            NUM_THREADS = 20;
//            omp_set_num_threads(NUM_THREADS);
//            DistBVCPLL<512> *dist_bvcpll = new DistBVCPLL<512>(G); // batch size 128, bit-parallel size 50.
//            delete dist_bvcpll;
//        }
        {// OpenMP Version
            NUM_THREADS = 24;
            omp_set_num_threads(NUM_THREADS);
			DistBVCPLL<1024> *dist_bvcpll = new DistBVCPLL<1024>(G); // batch size 1024, bit-parallel size 50.
//			DistBVCPLL<1024> dist_bvcpll(G); // batch size 1024, bit-parallel size 50.
            delete dist_bvcpll;
		}
//		{// Sequential Version
//			DistBVCPLL<1024> *dist_bvcpll = new DistBVCPLL<1024>(G); // batch size 1024, bit-parallel size 50.
////			DistBVCPLL<1024> dist_bvcpll(G); // batch size 1024, bit-parallel size 50.
//            delete dist_bvcpll;
//		}
//		{// Clear cache
//		    if (num_runs - 1 == i) {
//		        continue;
//		    }
//            // Adaptively clean up memory
////            double memtotal;
//            double memfree;
//            Utils::system_memory(memtotal, memfree);
//            uint64_t bytes_chunk = ((static_cast<uint64_t>(memtotal - mem_for_graph) >> 10ULL) - 5ULL) << 30ULL;
////            uint64_t bytes_chunk = static_cast<uint64_t>(memtotal * 0.8 / (1 << 10)) * (1ULL << 30ULL);
//            std::vector<uint64_t> chunk(bytes_chunk / 8ULL, 0);
//
//            double virtmen;
//            double resmem;
//            Utils::memory_usage(virtmen, resmem);
//            Utils::system_memory(memtotal, memfree);
//            printf("host_id: %u chunk_bytes: %luGB virtmem: %.2fGB resmem: %.2fGB memtotal: %.2fGB memfree: %.2fGB\n",
//                   G.host_id, chunk.size() * 8ULL / (1ULL << 30ULL), virtmen / (1 << 10), resmem / (1 << 10), memtotal / (1 << 10), memfree / (1 << 10));
//            printf("========================================\n");
////            uint64_t bytes_chunk = 100ULL * (1ULL << 30ULL);
////            std::vector<uint64_t> chunk(bytes_chunk / 8ULL, 0);
////		    printf("host_id: %u chunk_bytes: %luGB\n", G.host_id, chunk.size() * 8ULL / (1ULL << 30ULL));
////            double virtmen;
////            double resmem;
////		    double memtotal;
////		    double memfree;
////            Utils::memory_usage(virtmen, resmem);
////		    Utils::system_memory(memtotal, memfree);
////		    printf("host_id: %u virtmem: %.2fGB resmem: %.2fGB memtotal: %.2fGB memfree: %.2fGB\n",
////		            G.host_id, virtmen / (1 << 10), resmem / (1 << 10), memtotal / (1 << 10), memfree / (1 << 10));
//
////            std::ifstream fin(argv[2]);
////            if (!fin.is_open()) {
////                fprintf(stderr, "Error: cannot open file %s\n", argv[2]);
////                exit(EXIT_FAILURE);
////            }
////            std::vector< std::pair<VertexID, VertexID> > buffer;
////            VertexID head;
////            VertexID tail;
////            while (fin.read(reinterpret_cast<char *>(&head), sizeof(head))) {
////                fin.read(reinterpret_cast<char *>(&tail), sizeof(tail));
////                buffer.emplace_back(head, tail);
////            }
////            printf("host_id: %u input_buffer.size(): %lu\n", G.host_id, buffer.size());
////            printf("========================================\n");
//		}
	}

    /*
     * Global_num_labels: 67727254 average: 213.596065
     */

}

//void test_dynamic_receive()
//{
//    struct MsgBPLabel {
//        VertexID r_root_id;
//        UnweightedDist bp_dist[2];
//        uint64_t bp_sets[2][2];
//
//        MsgBPLabel() = default;
//        MsgBPLabel(VertexID r, UnweightedDist dist[], uint64_t sets[][2])
//                : r_root_id(r)
//        {
//            memcpy(bp_dist, dist, 2 * sizeof(UnweightedDist));
//            memcpy(bp_sets, sets, 2 * 2 * sizeof(uint64_t));
//        }
//    };
//    int host_id;
//    int num_hosts;
//    MPI_Comm_rank(MPI_COMM_WORLD, &host_id);
//    MPI_Comm_size(MPI_COMM_WORLD, &num_hosts);
//    // Send
//    std::vector<MsgBPLabel> send_buffer;
//    UnweightedDist dist0[2] = {1, 2};
//    uint64_t sets0[2][2] = {{1024, 1025},
//                            {1026, 1027}};
//    UnweightedDist dist1[2] = {3, 4};
//    uint64_t sets1[2][2] = {{512, 513},
//                            {514, 515}};
//    send_buffer.emplace_back(0, dist0, sets0);
//    send_buffer.emplace_back(1, dist1, sets1);
//
//    if (0 == host_id) {
//        // Send
////		MPI_Request request;
//        MPI_Send(send_buffer.data(),
//                 MPI_Instance::get_sending_size(send_buffer),
//                 MPI_CHAR,
//                 1,
//                 GRAPH_SHUFFLE,
//                 MPI_COMM_WORLD);
////		MPI_Wait(&request,
////				MPI_STATUS_IGNORE);
//    }
//
//    if (1 == host_id) {
//        // Receive
//        std::vector<MsgBPLabel> recv_buffer;
////		std::vector< std::pair<int, int> > recv_buffer;
//        int source = MPI_Instance::receive_dynamic_buffer_from_any(recv_buffer, num_hosts, GRAPH_SHUFFLE);
////        printf("source: %u recv_buffer.size(): %lu\n", source, recv_buffer.size());
//        for (const auto &p : recv_buffer) {
//            for (int i = 0; i < 2; ++i) {
//                printf("source: %u host_id: %u r: %u d[%u]: %u s_n1[%u]: %lu s_0[%u]: %lu\n",
//                       source, host_id, p.r_root_id, i, p.bp_dist[i], i, p.bp_sets[i][0], i, p.bp_sets[i][1]);
//            }
//        }
////		MPI_Wait(&request,
////				MPI_STATUS_IGNORE);
//    }
//}

//void test_dynamic_receive()
//{
//    struct MsgUnitBP {
//        VertexID v_global;
//        UnweightedDist dist;
//        uint64_t S_n1;
//        uint64_t S_0;
//
//        MsgUnitBP() = default;
//        MsgUnitBP(VertexID v, UnweightedDist d, uint64_t sn1, uint64_t s0) :
//            v_global(v), dist(d), S_n1(sn1), S_0(s0) { }
//    };
//    int host_id;
//    int num_hosts;
//    MPI_Comm_rank(MPI_COMM_WORLD, &host_id);
//    MPI_Comm_size(MPI_COMM_WORLD, &num_hosts);
//    // Send
//    std::vector<MsgUnitBP> send_buffer;
////    std::vector< std::pair<int, int> > send_buffer;
//
//	send_buffer.emplace_back(host_id, host_id * 1, host_id + 10, host_id + 100);
//	send_buffer.emplace_back(host_id, host_id * 2, host_id + 11, host_id + 101);
//	send_buffer.emplace_back(host_id, host_id * 3, host_id + 12, host_id + 102);
////	send_buffer.emplace_back(host_id, host_id * 1);
////	send_buffer.emplace_back(host_id, host_id * 2);
////	send_buffer.emplace_back(host_id, host_id * 3);
//
//	MPI_Request request;
//	if (0 == host_id) {
//		// Send
////		MPI_Request request;
//		MPI_Isend(send_buffer.data(),
//				MPI_Instance::get_sending_size(send_buffer),
//				MPI_CHAR,
//				1,
//				GRAPH_SHUFFLE,
//				MPI_COMM_WORLD,
//				&request);
////		MPI_Wait(&request,
////				MPI_STATUS_IGNORE);
//	} else if (1 == host_id) {
//		MPI_Isend(send_buffer.data(),
//				MPI_Instance::get_sending_size(send_buffer),
//				MPI_CHAR,
//				0,
//				GRAPH_SHUFFLE,
//				MPI_COMM_WORLD,
//				&request);
//	}
//
//	if (1 == host_id) {
//		// Receive
//		std::vector<MsgUnitBP> recv_buffer;
////		std::vector< std::pair<int, int> > recv_buffer;
//		int source = MPI_Instance::receive_dynamic_buffer_from_any(recv_buffer, num_hosts, GRAPH_SHUFFLE);
//		printf("source: %u recv_buffer.size(): %lu\n", source, recv_buffer.size());
//		for (const auto &p : recv_buffer) {
//			printf("host_id: %u (%u %u %lu %lu)\n", host_id, p.v_global, p.dist, p.S_n1, p.S_0);
//		}
////		MPI_Wait(&request,
////				MPI_STATUS_IGNORE);
//	} else if (0 == host_id) {
//		std::vector<MsgUnitBP> recv_buffer;
//		int source = MPI_Instance::receive_dynamic_buffer_from_any(recv_buffer, num_hosts, GRAPH_SHUFFLE);
//		printf("source: %u recv_buffer.size(): %lu\n", source, recv_buffer.size());
//		for (const auto &p : recv_buffer) {
//            printf("host_id: %u (%u %u %lu %lu)\n", host_id, p.v_global, p.dist, p.S_n1, p.S_0);
//		}
//	}
//	MPI_Wait(&request,
//			MPI_STATUS_IGNORE);
//}

void test_recv()
{
    int host_id;
    int num_hosts;
    MPI_Comm_rank(MPI_COMM_WORLD, &host_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_hosts);

    int a = 0;
    MPI_Recv(&a,
            1,
            MPI_INT,
            1,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

}


void usage_print()
{
    fprintf(stderr,
            "Usage: ./dpado <input_file>\n");
}

int main(int argc, char *argv[])
{
    std::string input_file;

    if (argc < 2) {
        usage_print();
        exit(EXIT_FAILURE);
    } else {
        input_file = std::string(argv[1]);
    }

	setbuf(stdout, nullptr); // stdout no buffer
//    printf("input_file: %s\n", input_file.c_str());
    MPI_Instance mpi_instance(argc, argv);

    dpado(argv);
//    test_system();
//    test_recv();
//    test_dynamic_receive();
    return EXIT_SUCCESS;
}

