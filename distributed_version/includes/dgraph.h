//
// Created by Zhen Peng on 5/14/19.
//

#ifndef PADO_DGRAPH_H
#define PADO_DGRAPH_H

#include <fstream>
#include <vector>
#include <utility>
#include <algorithm>
#include "mpi_dpado.h"
#include "dglobals.h"
#include "utils.h"

namespace PADO {

class DistGraph final {
private:
//    VertexID vertex_divide = 0; // the (maximum) number of vertices assigned to a host, supposed to be ceiling(num_v / num_hosts).
//    VertexID offset_vertex_id = 0; // The offset for global vertex id to local id.
    std::vector<VertexID> out_degrees; // out degrees of vertices

    // Init function: do some initialization work for the system.
    // List:    MPI,
    //          class members.
    void initialization() {
        MPI_Comm_rank(MPI_COMM_WORLD, &host_id);
        MPI_Comm_size(MPI_COMM_WORLD, &num_hosts);

//        host_id = 1; // test
//        num_hosts = 3; // test

//        vertex_divide = get_vertex_divide(); // the (maximum) number of vertices a host can be assigned to.
//        offset_vertex_id = host_id * vertex_divide;
//        num_masters = vertex_divide;
//        if (host_id == num_hosts - 1) {
//            num_masters = num_v - offset_vertex_id;
//        }

        // Get number of masters
        /* Example: assume 4 hosts and 15 vertices.
         *       +---------------+
         *   Host| 0 | 1 | 2 | 3 |
         *       +---------------+
         *       | 0 | 1 | 2 | 3 |
         *       | 7 | 6 | 5 | 4 |
         *       | 8 | 9 |10 |11 |
         *       |   |14 |13 |12 |
         */
        {
            VertexID quotient = num_v / num_hosts;
            VertexID remainder = num_v % num_hosts;
            num_masters = quotient;
            if (remainder) {
                if (quotient & 1U) {
                    if (static_cast<VertexID>(host_id) > num_hosts - 1 - remainder) {
                        num_masters += 1; // be assigned one more vertex
                    }
                } else {
                    if (static_cast<VertexID>(host_id) < remainder) {
                        num_masters += 1; // be assigned one more vertex
                    }
                }
            }
        }
        rank.resize(num_v);
        vertices_idx.resize(num_v);
//        vertices_idx.resize(num_masters);
        out_degrees.resize(num_v, 0);
        local_out_degrees.resize(num_v, 0);
    }

//    // Function: compute the vertex divide value, which is ceiling(num_v / num_hosts).
//    VertexID get_vertex_divide() const
//    {
//        assert(num_hosts && num_v);
//        VertexID tmp_divide = num_v / num_hosts;
//        if (tmp_divide * num_hosts < num_v) {
//            ++tmp_divide;
//        }
//
//        return tmp_divide;
//    }

public:
    int num_hosts = 0; // number of hosts
    int host_id = 0; // host ID
    VertexID num_v = 0; // number of vertices
    EdgeID num_e = 0; // number of edges
    VertexID num_masters = 0; // Number of masters on this host.
    EdgeID num_edges_local = 0; // Number of local edges on this host.
    MPI_Datatype V_ID_Type = MPI_Instance::get_mpi_datatype<VertexID>(); // MPI type of the type VertexID
    std::vector<VertexID> rank;
    std::vector<VertexID> vertices_idx; // vertices indices
    std::vector<VertexID> out_edges; // out edges
    std::vector<VertexID> local_out_degrees; // out degrees based on local edges.

    DistGraph() = default;
    ~DistGraph() = default;

    explicit DistGraph(const char *input_filename);

    // Function: return the global out degree of a vertex
    VertexID get_global_out_degree(VertexID v_global) const
    {
        assert(v_global < num_v);
        return out_degrees[v_global];
    }
    // Function: convert a vertex ID to its master host ID.
    // For example, vertex v should belong to host v / ceiling(num_v / num_hosts).
    int get_master_host_id(VertexID v_global) const
    {
        assert(num_hosts);
        return v_global % num_hosts == v_global % (2 * num_hosts)
                ? v_global % num_hosts
                : num_hosts - 1 - v_global % num_hosts;
    }
//    int get_master_host_id(VertexID v_global) const
//    {
//        assert(vertex_divide);
//        return v_global / vertex_divide;
//    }

    // Function: get the local vertex ID from the global ID
    VertexID get_local_vertex_id(VertexID global_id) const
    {
        assert(num_hosts);
        return global_id / num_hosts;
    }
//    VertexID get_local_vertex_id(VertexID global_id) const
//    {
//        return global_id - offset_vertex_id;
//    }

    // Function: get the global vertex ID from the local ID
    VertexID get_global_vertex_id(VertexID local_id) const
    {
        return local_id & 1U
                ? local_id * num_hosts + (num_hosts - 1 - host_id)
                : local_id * num_hosts + host_id;
    }
//    VertexID get_global_vertex_id(VertexID local_id) const
//    {
//        return local_id + offset_vertex_id;
//    }

    // Function: convert the master host id to the location of local sending buffer list.
    // For example, a message belonging to host x should be put into
    // buffer_send_list[(x + num_hosts - host_id - 1) % num_hosts]
    int master_host_id_2_buffer_send_list_loc(int master_host_id) const
    {
        assert(master_host_id >= 0 && master_host_id < num_hosts);
        return (master_host_id + num_hosts - host_id - 1) % num_hosts;
    }

    // Function: convert the location of local sending buffer list to the master host id.
    // For example, buffer_send_list[i] should be sent to host (host_id + i + 1) % num_hosts.
    int buffer_send_list_loc_2_master_host_id(int loc) const
    {
        assert(loc >= 0 && loc < num_hosts - 1);
        return (host_id + loc + 1) % num_hosts;
    }

    // Function: get the destination host id which is i hop from the root.
    // For example, 1 hop from host 2 is host 0 (assume total 3 hosts);
    // -1 hop from host 0 is host 2.
    int hop_2_dest_host_id(int hop, int root) const
    {
        assert(hop >= -(num_hosts - 1) && hop < num_hosts && hop != 0);
        assert(root >= 0 && root < num_hosts);
        return (root + hop + num_hosts) % num_hosts;
    }
};

// Constructor from a input file.
inline DistGraph::DistGraph(const char *input_filename)
{
    std::ifstream fin(input_filename);
    if (!fin.is_open()) {
        fprintf(stderr, "Error %s(%d): cannot open file %s\n", __FILE__, __LINE__, input_filename);
        exit(EXIT_FAILURE);
    }

    // Read num_v and num_e
    fin.read(reinterpret_cast<char *>(&num_v), sizeof(VertexID));
    fin.read(reinterpret_cast<char *>(&num_e), sizeof(EdgeID));
    // Test the file size
    uint64_t file_size = get_file_size(input_filename);
    uint64_t bytes_size = sizeof(VertexID) + sizeof(EdgeID) + num_e * 2 * sizeof(VertexID);
    assert(bytes_size == file_size);
    // Initialize class members.
    initialization();
    if (0 == host_id) {
        printf("Input: %s\n", input_filename);
    }

    // Get the offset (in bytes) for reading.
    uint64_t edge_byte_size = 2 * sizeof(VertexID); // the size (in bytes) of one edge
    uint64_t edge_divide = num_e / num_hosts; // divide the number of edges to the number of hosts
    uint64_t read_offset = host_id * edge_divide * edge_byte_size + sizeof(VertexID) + sizeof(EdgeID); //
        // reading offset for this host.
        // Need to consider the first two numbers.
    uint64_t edges_to_read = edge_divide; // number of edges that a host needs to read
    if (host_id == num_hosts - 1) {
        edges_to_read = num_e - edge_divide * host_id;
    }
    // Read from the offset.
//	printf("@%u host_id: %u read_offset: %lu\n", __LINE__, host_id, read_offset);//test
    fin.seekg(read_offset); // set reading offset
    std::vector< std::pair<VertexID, VertexID> > edgelist_buffer(edges_to_read);
    for (uint64_t e_i = 0; e_i < edges_to_read; ++e_i) {
        // TODO: optimization: read in chunk using a buffer, may speed up the reading process.
        VertexID head;
        VertexID tail;
        fin.read(reinterpret_cast<char *>(&head), sizeof(VertexID));
        fin.read(reinterpret_cast<char *>(&tail), sizeof(VertexID));
//        edgelist_buffer.emplace_back(head, tail);
        edgelist_buffer[e_i].first = head;
        edgelist_buffer[e_i].second = tail;
        ++out_degrees[head];
        ++out_degrees[tail]; // undirected graph
    }
    // All-reduce the degrees.
    MPI_Allreduce(MPI_IN_PLACE,
                  out_degrees.data(),
                  num_v,
                  V_ID_Type,
                  MPI_SUM,
                  MPI_COMM_WORLD);
//	printf("@%u host_id: %u out_degree\n", __LINE__, host_id);//test
    // Reorder the graph by host0.
    if (0 == host_id) {
        std::vector< std::pair<float, VertexID> > degree2id;
        for (VertexID v_i = 0; v_i < num_v; ++v_i) {
            degree2id.emplace_back(out_degrees[v_i] + float(rand()) / RAND_MAX, v_i);
        }
        std::sort(degree2id.rbegin(), degree2id.rend()); // sort according to degrees
        for (VertexID r = 0; r < num_v; ++r) {
            rank[degree2id[r].second] = r;
        }
    }
    // Send the rank to other hosts
    MPI_Bcast(rank.data(),
            num_v,
            V_ID_Type,
            0,
            MPI_COMM_WORLD);
//	printf("@%u host_id: %u bcast_rank\n", __LINE__, host_id);//test
    // Update the out_degree array according to the rank.
    {
        std::vector<VertexID> tmp_degrees(num_v);
        for (VertexID v = 0; v < num_v; ++v) {
            tmp_degrees[rank[v]] = out_degrees[v];
        }
        out_degrees.swap(tmp_degrees);
    }
    // Put reordered edges into corresponding buffer_sending
    std::vector< std::vector<VertexID> > edgelist_recv(num_v); // local received edges
    EdgeID num_edges_recv = 0;
    using EdgeType = std::pair<VertexID, VertexID>;
    std::vector< std::vector< EdgeType > > buffer_send_list(num_hosts - 1); //
        // buffer_send_list[i] should be sending to (host_id + i + 1) % num_hosts.
        // A host x's message should be put to buffer_send_list[(x + num_hosts - host_id - 1) % num_hosts].
    for (const auto &edge : edgelist_buffer) {
        VertexID head_new = rank[edge.first]; // rank[head]
        VertexID tail_new = rank[edge.second]; // rank[tail]
        int master_host_id_head = get_master_host_id(head_new); // master host id
        int master_host_id_tail = get_master_host_id(tail_new);
        if (master_host_id_tail != host_id) { // Add edge (head_new, tail_new) to the host of tail
            int loc_tail = master_host_id_2_buffer_send_list_loc(master_host_id_tail);
            buffer_send_list[loc_tail].emplace_back(head_new, tail_new);
        } else {
            edgelist_recv[head_new].push_back(tail_new);
            ++num_edges_recv;
        }
        if (master_host_id_head != host_id) { // Add edge (tail_new, head_new) to the host of head
            int loc_head = master_host_id_2_buffer_send_list_loc(master_host_id_head);
            buffer_send_list[loc_head].emplace_back(tail_new, head_new);
        } else {
            edgelist_recv[tail_new].push_back(head_new);
            ++num_edges_recv;
        }
    }
//	printf("@%u host_id: %u buffer_sending\n", __LINE__, host_id);//test
    {// Exchange edge lists
//        for (int hop = 1; hop < num_hosts; ++hop) {
//            int src = hop_2_dest_host_id(-hop, host_id);
//            int dst = hop_2_dest_host_id(hop, host_id);
//
//            // If host_id is higher than dst, first send, then receive
//            if (host_id < dst) {
//                // Send
//                MPI_Instance::send_buffer_2_dst(buffer_send_list[hop - 1],
//                                                dst,
//                                                SENDING_EDGELIST,
//                                                SENDING_SIZE_BUFFER_SEND);
//                // Receive
//                std::vector<EdgeType> buffer_recv;
//                MPI_Instance::recv_buffer_from_src(buffer_recv,
//                                                   src,
//                                                   SENDING_EDGELIST,
//                                                   SENDING_SIZE_BUFFER_SEND);
//                num_edges_recv += buffer_recv.size();
//                // Process
//                if (buffer_recv.empty()) {
//                    continue;
//                }
//                for (const auto &e : buffer_recv) {
//                    VertexID head = e.first;
//                    VertexID tail = e.second;
//                    edgelist_recv[head].push_back(tail);
//                }
//            } else { // Otherwise, if host_id is lower than dst, first receive, then send
//                // Receive
//                std::vector<EdgeType> buffer_recv;
//                MPI_Instance::recv_buffer_from_src(buffer_recv,
//                                                   src,
//                                                   SENDING_EDGELIST,
//                                                   SENDING_SIZE_BUFFER_SEND);
//                num_edges_recv += buffer_recv.size();
//                // Send
//                MPI_Instance::send_buffer_2_dst(buffer_send_list[hop - 1],
//                                                dst,
//                                                SENDING_EDGELIST,
//                                                SENDING_SIZE_BUFFER_SEND);
//                // Process
//                if (buffer_recv.empty()) {
//                    continue;
//                }
//                for (const auto &e : buffer_recv) {
//                    VertexID head = e.first;
//                    VertexID tail = e.second;
//                    edgelist_recv[head].push_back(tail);
//                }
//            }
//        }
        /////////////////////////////////////////////////
        //
        for (int h_i = 0; h_i < num_hosts; ++h_i) {
            // Send from h_i
            if (host_id == h_i) {
                for (int loc = 0; loc < num_hosts - 1; ++loc) {
                    int dst = buffer_send_list_loc_2_master_host_id(loc);
                    MPI_Instance::send_buffer_2_dst(buffer_send_list[loc],
                            dst,
                            SENDING_EDGELIST,
                            SENDING_SIZE_BUFFER_SEND);
                }
            } else { // Receive from h_i
                for (int hop = 1; hop < num_hosts; ++hop) {
                    int dest = hop_2_dest_host_id(hop, h_i);
                    if (host_id == dest) {
                        std::vector<EdgeType> buffer_recv;
                        MPI_Instance::recv_buffer_from_src(buffer_recv,
                                h_i,
                                SENDING_EDGELIST,
                                SENDING_SIZE_BUFFER_SEND);
                        num_edges_recv += buffer_recv.size();
                        // Process buffer_recv
                        for (const auto &e : buffer_recv) {
                            VertexID head = e.first;
                            VertexID tail = e.second;
                            edgelist_recv[head].push_back(tail);
                        }
                    }
                }
            }
        }
        //
        /////////////////////////////////////////////////
    }

//	printf("@%u host_id: %u received\n", __LINE__, host_id);//test
    // Build up local graph structure
    num_edges_local = num_edges_recv;
    out_edges.resize(num_edges_recv);
    std::vector<VertexID> test_in_degrees(num_masters, 0);
    EdgeID loc = 0;
    for (VertexID v_i = 0; v_i < num_v; ++v_i) {
        vertices_idx[v_i] = loc;
        size_t bound_e_i = edgelist_recv[v_i].size();
        local_out_degrees[v_i] = bound_e_i;
        std::sort(edgelist_recv[v_i].rbegin(), edgelist_recv[v_i].rend()); // sort neighbors by ranks from low to high
        for (EdgeID e_i = 0; e_i < bound_e_i; ++e_i) {
            out_edges[loc + e_i] = edgelist_recv[v_i][e_i];
            ++test_in_degrees[get_local_vertex_id(edgelist_recv[v_i][e_i])];
        }
        loc += bound_e_i;
    }
    assert(loc == num_edges_recv);
    for (VertexID v_local = 0; v_local < num_masters; ++v_local) {
        VertexID v_global = get_global_vertex_id(v_local);
        assert(out_degrees[v_global] == test_in_degrees[v_local]); // undirected graph.
    }
//	printf("@%u host_id: %u built_up\n", __LINE__, host_id);//test
//    {//test
//        double virtual_memory;
//        double resident_memory;
//        Utils::memory_usage(virtual_memory, resident_memory);
//        printf("host_id: %u virtual_memory: %.2fMB resident_memory: %.2fMB\n", host_id, virtual_memory, resident_memory);
//    }
}

} // End namespace PADO

#endif //PADO_DGRAPH_H
