//
// Created by Zhen Peng on 5/14/19.
//

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "dglobals.h"
#include "dgraph.h"
#include "dpado.h"

using namespace PADO;


void dpado(char *argv[])
{
    DistGraph G(argv[1]);
	printf("host_id: %u num_masters: %u /%u %.2f%% num_edges_local %lu /%lu %.2f%%\n",
	        G.host_id, G.num_masters, G.num_v, 100.0 * G.num_masters / G.num_v, G.num_edges_local, 2 * G.num_e, 100.0 * G.num_edges_local / (2 * G.num_e));//test
    {//test
        system("free -h");
    }

	DistBVCPLL<1024, 50> dist_bvcpll(G); // batch size 1024, bit-parallel size 0.
//	DistBVCPLL<8, 50> dist_bvcpll(G); // batch size 1024, bit-parallel size 0.

//    {// test the index by distance queries
//        std::ifstream fin(argv[2]);
//        if (!fin.is_open()) {
//            fprintf(stderr, "Error: cannot open file %s", argv[2]);
//            exit(EXIT_FAILURE);
//        }
//        VertexID a;
//        VertexID b;
//        while (fin >> a >> b) {
//            UnweightedDist dist = dist_bvcpll.dist_distance_query_pair(a, b, G);
//            MPI_Barrier(MPI_COMM_WORLD);
//            if (0 == G.host_id) {
//                if (dist == 255) {
//                    printf("2147483647\n");
//                } else {
//                    printf("%u\n", dist);
//                }
//            }
//        }
//    }

    /*
     * Global_num_labels: 67727254 average: 213.596065
     */

//    // test the index
//    {
//        std::vector<VertexID> rank2id(G.num_v);
//        for (VertexID v = 0; v < G.num_v; ++v) {
//            rank2id[G.rank[v]] = v;
//        }
//        dist_bvcpll.switch_labels_to_old_id(rank2id);
//    }

//    // print the local graph
//	{
//        std::vector<VertexID> rank2id(G.num_v);
//        for (VertexID v = 0; v < G.num_v; ++v) {
//            rank2id[G.rank[v]] = v;
//        }
//        std::vector< std::pair<VertexID, VertexID> > tmp_edges_by_dst;
//		std::string filename = "output_" + std::to_string(G.host_id) + ".txt";
//		FILE *fout = fopen(filename.c_str(), "w");
//		fprintf(fout, "num_master: %d (/%u) num_edges_local: %lu (/%lu) host_id: %d (/%d)\n",
//						G.num_masters, G.num_v, G.num_edges_local, G.num_e * 2, G.host_id, G.num_hosts);
//        // Traverse the local G
//        for (VertexID v_i = 0; v_i < G.num_v; ++v_i) {
//            VertexID head = v_i;
////            VertexID head = rank2id[v_i + G.offset_vertex_id];
//            EdgeID start_e_i = G.vertices_idx[v_i];
//            EdgeID bound_e_i = start_e_i + G.local_out_degrees[v_i];
//            for (EdgeID e_i = start_e_i; e_i < bound_e_i; ++e_i) {
//                VertexID tail = G.out_edges[e_i];
////                VertexID tail = rank2id[G.out_edges[e_i]];
////                tmp_edges_by_dst.emplace_back(head, tail);
//                tmp_edges_by_dst.emplace_back(tail, head);
//            }
//        }
//        std::sort(tmp_edges_by_dst.begin(), tmp_edges_by_dst.end());
//        for (const auto &e : tmp_edges_by_dst) {
////            fprintf(fout, "%u %u\n", e.first, e.second);
//            fprintf(fout, "%u %u\n", e.second, e.first);
//        }
////        // Traverse the local G
////        for (VertexID v_i = 0; v_i < G.num_masters; ++v_i) {
////            VertexID head = v_i + G.offset_vertex_id;
//////            VertexID head = rank2id[v_i + G.offset_vertex_id];
////            EdgeID start_e_i = G.vertices_idx[v_i];
////            EdgeID bound_e_i = G.num_edges_local;
////            if (v_i != G.num_masters - 1) {
////                bound_e_i = G.vertices_idx[v_i + 1];
////            }
////            for (EdgeID e_i = start_e_i; e_i < bound_e_i; ++e_i) {
////                VertexID tail = G.out_edges[e_i];
//////                VertexID tail = rank2id[G.out_edges[e_i]];
////                //std::cout << head << " " << tail << std::endl;
////                fprintf(fout, "%u %u\n", head, tail);
////            }
////        }
//		fclose(fout);
//	}

//    if (0 == G.host_id) {
//		std::cout << "num_master: " << G.num_masters << " (/" << G.num_v << ")"
//				<< " num_edges_local: " << G.num_edges_local << "(/" << G.num_e * 2 << ")"
//				<< " host_id: " << G.host_id << "(/" << G.num_hosts << ")" << std::endl;
//        // Traverse the local G
//        for (VertexID v_i = 0; v_i < G.num_masters; ++v_i) {
//            VertexID head = v_i + G.offset_vertex_id;
////            VertexID head = rank2id[v_i + G.offset_vertex_id];
//            EdgeID start_e_i = G.vertices_idx[v_i];
//            EdgeID bound_e_i = G.num_edges_local;
//            if (v_i != G.num_masters - 1) {
//                bound_e_i = G.vertices_idx[v_i + 1];
//            }
//            for (EdgeID e_i = start_e_i; e_i < bound_e_i; ++e_i) {
//                VertexID tail = G.out_edges[e_i];
////                VertexID tail = rank2id[G.out_edges[e_i]];
//                std::cout << head << " " << tail << std::endl;
//            }
//        }
//		printf("@%d host %d send to host %d\n", __LINE__, G.host_id, (G.host_id + 1) % G.num_hosts); //test
//        MPI_Send(nullptr, 0, MPI_CHAR, (G.host_id + 1) % G.num_hosts, SENDING_MESSAGE, MPI_COMM_WORLD);
//        MPI_Recv(nullptr, 0, MPI_CHAR, G.num_hosts - 1, SENDING_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//		printf("@%d host %d received from host %d\n", __LINE__, G.host_id, G.num_hosts - 1); //test
//    } else {
//		MPI_Recv(nullptr, 0, MPI_CHAR, G.host_id - 1, SENDING_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//		printf("@%d host %d received from host %d\n", __LINE__, G.host_id, G.host_id - 1); //test
//		std::cout << "num_master: " << G.num_masters << " (/" << G.num_v << ")"
//				<< " num_edges_local: " << G.num_edges_local << "(/" << G.num_e * 2 << ")"
//				<< " host_id: " << G.host_id << "(/" << G.num_hosts << ")" << std::endl;
//        // Traverse the local G
//        for (VertexID v_i = 0; v_i < G.num_masters; ++v_i) {
////            VertexID head = rank2id[v_i + G.offset_vertex_id];
//            VertexID head = v_i + G.offset_vertex_id;
//            EdgeID start_e_i = G.vertices_idx[v_i];
//            EdgeID bound_e_i = G.num_edges_local;
//            if (v_i != G.num_masters - 1) {
//                bound_e_i = G.vertices_idx[v_i + 1];
//            }
//            for (EdgeID e_i = start_e_i; e_i < bound_e_i; ++e_i) {
////                VertexID tail = rank2id[G.out_edges[e_i]];
//                VertexID tail = G.out_edges[e_i];
//                std::cout << head << " " << tail << std::endl;
//            }
//        }
//		printf("@%d host %d send to host %d\n", __LINE__, G.host_id, (G.host_id + 1) % G.num_hosts); //test
//		MPI_Send(nullptr, 0, MPI_CHAR, (G.host_id + 1) % G.num_hosts, SENDING_MESSAGE, MPI_COMM_WORLD);
//    }

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

void test_system()
{
    system("free -h");
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

