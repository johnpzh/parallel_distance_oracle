//
// Created by Zhen Peng on 5/15/19.
//

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "dglobals.h"
#include "globals.h"

using namespace PADO;

// Convert a binary file to a edgelist text file (unweighted).
// The binary file is generated by the tool edgelist2binary.
// The first two elements of the binary file are the number of vertex (type of VertexID) and the number of
// edges (type of EdgeID).

void convert(char *input_filename, char *output_filename)
{
    double running_time = -WallTimer::get_time_mark();
    std::ifstream fin(input_filename);
    if (!fin.is_open()) {
        fprintf(stderr, "Error: cannot open file %s\n", input_filename);
        exit(EXIT_FAILURE);
    }

    std::ofstream fout(output_filename);
    if (!fout.is_open()) {
        fprintf(stderr, "Error: cannot create file %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    VertexID num_v; // number of vertices;
    EdgeID num_e; // number of edges;
    fin.read(reinterpret_cast<char *>(&num_v), sizeof(num_v));
    fin.read(reinterpret_cast<char *>(&num_e), sizeof(num_e));
    // Set up a read buffer
    uint32_t max_buffer_size = 65536;
    std::vector<VertexID> buffer_read(max_buffer_size);
//    uint32_t buffer_size = 0;
    EdgeID edge_count = 0;
    while (edge_count < num_e) {
        if (edge_count + max_buffer_size / 2 < num_e) {
            fin.read(reinterpret_cast<char *>(buffer_read.data()), max_buffer_size * sizeof(VertexID));
            // Write to the output file.
            for (uint32_t i = 0; i < max_buffer_size; i += 2) {
                fout << buffer_read[i] << " " << buffer_read[i + 1] << std::endl;
            }
            edge_count += max_buffer_size / 2;
        } else {
            uint32_t buffer_size = (num_e - edge_count) * 2;
            fin.read(reinterpret_cast<char *>(buffer_read.data()), buffer_size * sizeof(VertexID));
            // Write to the output file.
            for (uint32_t i = 0; i < buffer_size; i += 2) {
                fout << buffer_read[i] << " " << buffer_read[i + 1] << std::endl;
            }
            edge_count += buffer_size / 2;
        }
    }
//    assert(fin.eof()); // Check if the fin reached the end of file. Mistake!
    running_time += WallTimer::get_time_mark();
    std::cout << "num_v: " << num_v << " num_e: " << num_e << std::endl;
    printf("running_time: %f\n", running_time);
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr,
                "Usage: ./binary2edgelist <input_binary_file> <output_edgelist_file>\n"
                "\t<input_binary_file> should be unweighted.\n");
        exit(EXIT_FAILURE);
    }

    convert(argv[1], argv[2]);

    return EXIT_SUCCESS;
}