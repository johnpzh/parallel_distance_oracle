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

void test_dynamic_receive()
{
    int host_id;
    int num_hosts;
    MPI_Comm_rank(MPI_COMM_WORLD, &host_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_hosts);
    // Send
    std::vector< std::pair<int, int> > send_buffer;

	send_buffer.emplace_back(host_id, host_id * 1);
	send_buffer.emplace_back(host_id, host_id * 2);
	send_buffer.emplace_back(host_id, host_id * 3);

	MPI_Request request;
	if (0 == host_id) {
		// Send
//		MPI_Request request;
		MPI_Isend(send_buffer.data(),
				MPI_Instance::get_sending_size(send_buffer),
				MPI_CHAR,
				1,
				GRAPH_SHUFFLE,
				MPI_COMM_WORLD,
				&request);
//		MPI_Wait(&request,
//				MPI_STATUS_IGNORE);
	} else if (1 == host_id) {
		MPI_Isend(send_buffer.data(),
				MPI_Instance::get_sending_size(send_buffer),
				MPI_CHAR,
				0,
				GRAPH_SHUFFLE,
				MPI_COMM_WORLD,
				&request);
	}

	if (1 == host_id) {
		// Receive
		std::vector< std::pair<int, int> > recv_buffer;
		int source = MPI_Instance::receive_dynamic_buffer_from_any(recv_buffer, num_hosts, GRAPH_SHUFFLE);
		printf("source: %u recv_buffer.size(): %lu\n", source, recv_buffer.size());
		for (const auto &p : recv_buffer) {
			printf("host_id: %u %d %d\n", host_id, p.first, p.second);
		}
//		MPI_Wait(&request,
//				MPI_STATUS_IGNORE);
	} else if (0 == host_id) {
		std::vector< std::pair<int, int> > recv_buffer;
		int source = MPI_Instance::receive_dynamic_buffer_from_any(recv_buffer, num_hosts, GRAPH_SHUFFLE);
		printf("source: %u recv_buffer.size(): %lu\n", source, recv_buffer.size());
		for (const auto &p : recv_buffer) {
			printf("host_id: %u %d %d\n", host_id, p.first, p.second);
		}
	}
	MPI_Wait(&request,
			MPI_STATUS_IGNORE);
}

void dpado(char *argv[])
{
    DistGraph G(argv[1]);
	printf("host_id: %u num_masters: %u(/%u %.2f%%) num_edges_local %lu(/%lu %.2f%%)\n",
	        G.host_id, G.num_masters, G.num_v, 100.0 * G.num_masters / G.num_v, G.num_edges_local, 2 * G.num_e, 100.0 * G.num_edges_local / (2 * G.num_e));//test

	DistBVCPLL<1024, 0> dist_bvcpll(G); // batch size 1024, bit-parallel size 0.
//	DistBVCPLL<8, 0> dist_bvcpll(G); // batch size 1024, bit-parallel size 0.

    {// test the index by distance queries
        std::ifstream fin(argv[2]);
        if (!fin.is_open()) {
            fprintf(stderr, "Error: cannot open file %s", argv[2]);
            exit(EXIT_FAILURE);
        }
        VertexID a;
        VertexID b;
        while (fin >> a >> b) {
            UnweightedDist dist = dist_bvcpll.dist_distance_query_pair(a, b, G);
            MPI_Barrier(MPI_COMM_WORLD);
            if (0 == G.host_id) {
                if (dist == 255) {
                    printf("2147483647\n");
                } else {
                    printf("%u\n", dist);
                }
            }
        }
    }

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
//                tmp_edges_by_dst.emplace_back(head, tail);
////                tmp_edges_by_dst.emplace_back(tail, head);
//            }
//        }
//        std::sort(tmp_edges_by_dst.begin(), tmp_edges_by_dst.end());
//        for (const auto &e : tmp_edges_by_dst) {
//            fprintf(fout, "%u %u\n", e.first, e.second);
////            fprintf(fout, "%u %u\n", e.second, e.first);
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
    printf("input_file: %s\n", input_file.c_str());
    MPI_Instance mpi_instance(argc, argv);

    dpado(argv);
//    test_dynamic_receive();
    return EXIT_SUCCESS;
}

