//
// Created by Zhen Peng on 5/14/19.
//

#ifndef PADO_DGRAPH_H
#define PADO_DGRAPH_H

#include <fstream>
#include <vector>
#include <utility>
#include <algorithm>
//#include <math.h>
#include "dglobals.h"

namespace PADO {

class DistGraph final {
private:
    int num_hosts = 1; // number of hosts
    int host_id = 0; // host ID
    VertexID vertex_divide = 0; // the (maximum) number of vertices assigned to a host, supposed to be ceiling(num_v / num_hosts).

    // Init function: do some initialization work for the system.
    // List:    MPI.
    void init() {
        MPI_Comm_rank(MPI_COMM_WORLD, &host_id);
        MPI_Comm_size(MPI_COMM_WORLD, &num_hosts);
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

public:
    VertexID num_v = 0; // number of vertices
    EdgeID num_e = 0; // number of edges

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
    init(); // Initialization
    MPI_Datatype vid_type = MPI_Instance::get_mpi_datatype<VertexID >();
    std::ifstream fin(input_filename);
    if (!fin.is_open()) {
        fprintf(stderr, "Error %s(%d): cannot open file %s\n", __FILE__, __LINE__, input_filename);
        exit(EXIT_FAILURE);
    }

    // Read num_v and num_e
    fin.read(reinterpret_cast<char *>(&num_v), sizeof(VertexID));
    fin.read(reinterpret_cast<char *>(&num_e), sizeof(EdgeID));
    vertex_divide = get_vertex_divide(); // the (maximum) number of vertices a host can be assigned to.
    out_degrees.resize(num_v, 0);
    // Test the file size
    uint64_t file_size = get_file_size(input_filename);
    uint64_t bytes_size = sizeof(VertexID) + sizeof(EdgeID) + num_e * 2 * sizeof(VertexID);
    assert(bytes_size == file_size);

    // Get the offset (in bytes) for reading.
    uint64_t edge_byte_size = 2 * sizeof(VertexID); // the size (in bytes) of one edge
    uint64_t edge_divide = num_e / num_hosts; // divide the number of edges to the number of hosts
    uint64_t read_offset = edge_divide * host_id * edge_byte_size; // reading offset for this host
    uint64_t edges_to_read; // number of edges that a host needs to read
    if (host_id != num_hosts - 1) {
        edges_to_read = edge_divide;
    } else {
        edges_to_read = num_e - edge_divide * host_id;
    }
    // Read from the offset.
    fin.seekg(read_offset); // set reading offset
    std::vector< std::pair<VertexID, VertexID> > edgelist_buffer(edges_to_read);
    VertexID head;
    VertexID tail;
    for (uint64_t e_i = 0; e_i < edges_to_read; ++e_i) {
        // TODO: optimization: read in chunk using a buffer, may speed up the reading process.
        fin.read(reinterpret_cast<char *>(&head), sizeof(VertexID));
        fin.read(reinterpret_cast<char *>(&tail), sizeof(VertexID));
        edgelist_buffer.emplace_back(head, tail);
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
    std::vector<VertexID> rank(num_v);
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
    MPI_Bcast(rank.data(), num_v, vid_type, 0, MPI_COMM_WORLD);
    // Put reordered edges into corresponding buffer_sending
    std::vector< std::vector< std::pair<VertexID, VertexID> > > buffer_send_list(num_hosts - 1); //
        // buffer_send_list[i] should be sending to (host_id + i + 1) % num_hosts.
        // A host x's message should be put to buffer_send_list[(x + num_hosts - host_id - 1) % num_hosts].
    for (const auto &edge : edgelist_buffer) {
        VertexID head_new = rank[edge.first]; // rank[head]
        VertexID tail_new = rank[edge.second]; // rank[tail]
        int master_host_id_head = get_master_host_id(head_new); // master host id
        int master_host_id_tail = get_master_host_id(tail_new);
        int loc_head = master_host_id_2_buffer_send_list_loc(master_host_id_head); // location in the sending buffer list
        int loc_tail = master_host_id_2_buffer_send_list_loc(master_host_id_tail);
        buffer_send_list[loc_head].emplace_back(head_new, tail_new); // add the edge into the sending buffer.
        buffer_send_list[loc_tail].emplace_back(tail_new, head_new);
    }
    // Send the edges in buffer_sending to corresponding hosts.
    for (int loc = 0; loc < num_hosts - 1; ++loc) {
        int master_host_id = buffer_send_list_loc_2_master_host_id(loc);

    }
    // Receive the edges
}

} // End namespace PADO

#endif //PADO_DGRAPH_H
