//
// Created by Zhen Peng on 5/14/19.
//

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "dglobals.h"
#include "dgraph.h"

using namespace PADO;

void test_dynamic_receive()
{
    int host_id;
    int num_hosts;
    MPI_Comm_rank(MPI_COMM_WORLD, &host_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_hosts);
    // Send
    std::vector< std::pair<int, int> > send_buffer;
    send_buffer.emplace_back(1, 2);
    send_buffer.emplace_back(2, 4);
    send_buffer.emplace_back(3, 5);

    MPI_Send(send_buffer.data(),
             sizeof(send_buffer),
             MPI_CHAR,
             0,
             GRAPH_SHUFFLE,
             MPI_COMM_WORLD);
    // Receive
    std::vector< std::pair<int, int> > recv_buffer;
    int count = MPI_Instance::receive_dynamic_buffer(recv_buffer, num_hosts);
    printf("received_count: %d\n", count);
    for (const auto &p : recv_buffer) {
        printf("%d %d\n", p.first, p.second);
    }
}

void dpado(char *argv[])
{
    DistGraph G(argv[1]);
    printf("File: %s\n", argv[1]);
    std::cout << "num_master: " << G.num_masters << " (/" << G.num_v << ")"
                << " num_edges_local: " << G.num_edges_local << "(/" << G.num_e * 2 << ")"
                << " host_id: " << G.host_id << "(/" << G.num_hosts << ")" << std::endl;

    // print the local graph
    std::vector<VertexID> rank2id(G.num_v);
    for (VertexID v = 0; v < G.num_v; ++v) {
        rank2id[G.rank[v]] = v;
    }
    if (0 == G.host_id) {
        // Traverse the local G
        for (VertexID v_i = 0; v_i < G.num_masters; ++v_i) {
            VertexID head = v_i + G.offset_vertex_id;
//            VertexID head = rank2id[v_i + G.offset_vertex_id];
            EdgeID start_e_i = G.vertices_idx[v_i];
            EdgeID bound_e_i = G.num_edges_local;
            if (v_i != G.num_masters - 1) {
                bound_e_i = G.vertices_idx[v_i + 1];
            }
            for (EdgeID e_i = start_e_i; e_i < bound_e_i; ++e_i) {
                VertexID tail = G.out_edges[e_i];
//                VertexID tail = rank2id[G.out_edges[e_i]];
                std::cout << head << " " << tail << std::endl;
            }
        }
        MPI_Send(nullptr, 0, MPI_CHAR, (G.host_id + 1) % G.num_hosts, SENDING_MESSAGE, MPI_COMM_WORLD);
        MPI_Recv(nullptr, 0, MPI_CHAR, G.num_hosts - 1, SENDING_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
//        MPI_Recv(nullptr, 0, MPI_CHAR, G.host_id - 1, SENDING_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Traverse the local G
        for (VertexID v_i = 0; v_i < G.num_masters; ++v_i) {
//            VertexID head = rank2id[v_i + G.offset_vertex_id];
            VertexID head = v_i + G.offset_vertex_id;
            EdgeID start_e_i = G.vertices_idx[v_i];
            EdgeID bound_e_i = G.num_edges_local;
            if (v_i != G.num_masters - 1) {
                bound_e_i = G.vertices_idx[v_i + 1];
            }
            for (EdgeID e_i = start_e_i; e_i < bound_e_i; ++e_i) {
//                VertexID tail = rank2id[G.out_edges[e_i]];
                VertexID tail = G.out_edges[e_i];
                std::cout << head << " " << tail << std::endl;
            }
        }
//        MPI_Send(nullptr, 0, MPI_CHAR, (G.host_id + 1) % G.num_hosts, SENDING_MESSAGE, MPI_COMM_WORLD);
    }

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

    printf("input_file: %s\n", input_file.c_str());
    MPI_Instance mpi_instance(argc, argv);
//    dpado(argv);
    test_dynamic_receive();
    return EXIT_SUCCESS;
}

