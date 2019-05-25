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

namespace PADO {

class DistGraph final {
private:
    VertexID vertex_divide = 0; // the (maximum) number of vertices assigned to a host, supposed to be ceiling(num_v / num_hosts).


    // Init function: do some initialization work for the system.
    // List:    MPI,
    //          class members.
    void init() {
        MPI_Comm_rank(MPI_COMM_WORLD, &host_id);
        MPI_Comm_size(MPI_COMM_WORLD, &num_hosts);
        vertex_divide = get_vertex_divide(); // the (maximum) number of vertices a host can be assigned to.
        offset_vertex_id = host_id * vertex_divide;
        num_masters = vertex_divide;
        if (host_id == num_hosts - 1) {
            num_masters = num_v - offset_vertex_id;
        }
        rank.resize(num_v);
        vertices_idx.resize(num_masters);
        out_degrees.resize(num_v, 0);
    }

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

    // Function: convert a vertex ID to its master host ID.
    // For example, vertex v should belong to host v / ceiling(num_v / num_hosts).
    int get_master_host_id(VertexID v_id)
    {
        assert(vertex_divide);
        return v_id / vertex_divide;
    }

    // Function: compute the vertex divide value, which is ceiling(num_v / num_hosts).
    VertexID get_vertex_divide()
    {
        assert(num_hosts && num_v);
        VertexID tmp_divide = num_v / num_hosts;
        if (tmp_divide * num_hosts < num_v) {
            ++tmp_divide;
        }

        return tmp_divide;
    }

    // Function: get the local vertex ID from the global ID
    VertexID get_local_vertex_id(VertexID global_id)
    {
        return global_id - offset_vertex_id;
    }

public:
    int num_hosts = 1; // number of hosts
    int host_id = 0; // host ID
    VertexID num_v = 0; // number of vertices
    EdgeID num_e = 0; // number of edges
    VertexID offset_vertex_id = 0; // The offset for global vertex id to local id.
    VertexID num_masters = 0; // Number of masters on this host.
    EdgeID num_edges_local = 0; // Number of local edges on this host.
    MPI_Datatype vid_type = MPI_Instance::get_mpi_datatype<VertexID>(); // MPI type of the type VertexID

    std::vector<VertexID> rank;
    std::vector<VertexID> vertices_idx; // vertices indices
    std::vector<VertexID> out_edges; // out edges
    std::vector<VertexID> out_degrees; // out degrees of vertices

    DistGraph() = default;
    ~DistGraph() = default;

    explicit DistGraph(char *input_filename);
};

// Constructor from a input file.
DistGraph::DistGraph(char *input_filename)
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
    init();

    // Get the offset (in bytes) for reading.
    uint64_t edge_byte_size = 2 * sizeof(VertexID); // the size (in bytes) of one edge
    uint64_t edge_divide = num_e / num_hosts; // divide the number of edges to the number of hosts
    uint64_t read_offset = host_id * edge_divide * edge_byte_size + sizeof(VertexID) + sizeof(EdgeID); //
        // reading offset for this host.
        // Need to consider the first two numbers.
    uint64_t edges_to_read; // number of edges that a host needs to read
    if (host_id != num_hosts - 1) {
        edges_to_read = edge_divide;
    } else {
        edges_to_read = num_e - edge_divide * host_id;
    }
    // Read from the offset.
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
                  vid_type,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    // Reorder the graph by host0.
    if (0 == host_id) {
        std::vector< std::pair<VertexID, VertexID> > degree2id;
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
            vid_type,
            0,
            MPI_COMM_WORLD);
    // Put reordered edges into corresponding buffer_sending
    std::vector< std::vector<VertexID> > edgelist_recv(num_masters); // local received edges
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
        if (master_host_id_head != host_id) {
            int loc_head = master_host_id_2_buffer_send_list_loc(master_host_id_head); // location in the sending buffer list
            buffer_send_list[loc_head].emplace_back(head_new, tail_new); // add the edge into the sending buffer.
        } else {
            // put edge into local edgelist
            edgelist_recv[get_local_vertex_id(head_new)].push_back(tail_new);
            ++num_edges_recv;
        }
        if (master_host_id_tail != host_id) {
            int loc_tail = master_host_id_2_buffer_send_list_loc(master_host_id_tail);
            buffer_send_list[loc_tail].emplace_back(tail_new, head_new);
        } else {
            // put edge into local edgelist
            edgelist_recv[get_local_vertex_id(tail_new)].push_back(head_new);
            ++num_edges_recv;
        }
    }
    // Send the edges in buffer_sending to corresponding hosts.
    for (int loc = 0; loc < num_hosts - 1; ++loc) {
        int master_host_id = buffer_send_list_loc_2_master_host_id(loc);
        MPI_Send(buffer_send_list[loc].data(),
                sizeof(buffer_send_list[loc]),
                MPI_CHAR,
                master_host_id,
                GRAPH_SHUFFLE,
                MPI_COMM_WORLD);
    }
    // Receive the edges
    std::vector<EdgeType> buffer_recv;
    for (int h_i = 0; h_i < num_hosts - 1; ++h_i) {
        // Receive into the buffer_recv.
        num_edges_recv += MPI_Instance::receive_dynamic_buffer(buffer_recv, num_hosts);
        // Put into edgelist_recv
        for (const auto &e : buffer_recv) {
            VertexID head = e.first;
            VertexID tail = e.second;
            edgelist_recv[get_local_vertex_id(head)].push_back(tail);
        }
    }
    // Build up local graph structure
    num_edges_local = num_edges_recv;
    out_edges.resize(num_edges_recv);
    EdgeID loc = 0;
    for (VertexID v_i = 0; v_i < num_masters; ++v_i) {
        vertices_idx[v_i] = loc;
        size_t bound_e_i = edgelist_recv[v_i].size();
        for (EdgeID e_i = 0; e_i < bound_e_i; ++e_i) {
            out_edges[loc + e_i] = edgelist_recv[v_i][e_i];
        }
        loc += bound_e_i;
    }
    assert(loc == num_edges_recv);
}

} // End namespace PADO

#endif //PADO_DGRAPH_H