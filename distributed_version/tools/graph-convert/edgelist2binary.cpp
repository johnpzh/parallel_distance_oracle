//
// Created by Zhen Peng on 5/14/19.
//

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include "dglobals.h"
#include "globals.h"

using namespace PADO;

// Convert a edgelist text (unweighted) file to a binary file.
// Edgelist file contains E lines. Every line is an edge which is two vertex IDs are separated with a whitespace.
// The first two elements of the output binary file are the number of vertex (type of VertexID) and the number
// of edges (type of EdgeID). Those following elements are edges (with two VertexID number per edge).
/*
 * V E
 * v v
 * v v
 * v v
 * ...
 */
void convert(char *input_filename, char *output_filename)
{
    double time_running = -WallTimer::get_time_mark();
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

    std::string line;
    VertexID head;
    VertexID tail;
    VertexID num_v = 0; // number of vertices.
    EdgeID num_e = 0; // number of edges.

    // Read from the file, get only the num_v and num_e.
    while (std::getline(fin, line)) {
        if (line[0] == '#' || line[0] == '%') {
            continue;
        }
        ++num_e;
        std::istringstream iss(line);
        iss >> head >> tail;
        num_v = std::max(num_v, std::max(head, tail) + static_cast<VertexID>(1));
    }

    // Output the binary
    // Read the input file again at first.
    fin.clear();
    fin.seekg(0); // set to the beginning
    fout.write(reinterpret_cast<char *>(&num_v), sizeof(num_v)); // write the number of vertices.
    fout.write(reinterpret_cast<char *>(&num_e), sizeof(num_e)); // write the number of edges.
    // Set up the write buffer.
    uint32_t max_buffer_size = 65536;
    std::vector<VertexID> buffer_write(max_buffer_size);
    uint32_t buffer_size = 0;
    while (std::getline(fin, line)) {
        if (line[0] == '#' || line[0] == '%') {
            continue;
        }
        std::istringstream iss(line);
        iss >> head >> tail;
        buffer_write[buffer_size++] = head;
        buffer_write[buffer_size++] = tail;
        if (buffer_size == max_buffer_size) {
            // write to the output file
            fout.write(reinterpret_cast<char *>(buffer_write.data()), buffer_size * sizeof(VertexID));
            buffer_size = 0;
        }
    }
    if (buffer_size) {
        // write remaining to the output file
        fout.write(reinterpret_cast<char *>(buffer_write.data()), buffer_size * sizeof(VertexID));
    }

    time_running += WallTimer::get_time_mark();
    std::cout << "num_v: " << num_v << " num_edges: " << num_e << std::endl;
    printf("running_time: %f\n", time_running);
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr,
                "Usage: ./edge2list2binary <input_edgelist_file> <output_binary_file>\n"
                "\t<input_edgelist_file> should be unweighted.\n");
        exit(EXIT_FAILURE);
    }

    convert(argv[1], argv[2]);

    return EXIT_SUCCESS;
}


