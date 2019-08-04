//
// Created by Zhen Peng on 8/4/19.
//

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <time.h>
#include "dglobals.h"
#include "globals.h"

using namespace PADO;

/*
 * Create a binary file. The format is just sequence of pairs of Vertex IDs.
 */


void create(const char *filename, VertexID num_v, EdgeID num_e)
{
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        fprintf(stderr, "Error: cannot create file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    srand(time(0));

    double time_running = -WallTimer::get_time_mark();
    for (EdgeID e_i = 0; e_i < num_e; ++e_i) {
        VertexID head = rand() % num_v;
        VertexID tail = rand() % num_v;
        fout.write(reinterpret_cast<char *>(&head), sizeof(head));
        fout.write(reinterpret_cast<char *>(&tail), sizeof(tail));
    }
    time_running += WallTimer::get_time_mark();
    printf("running_time: %f\n", time_running);
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        fprintf(stderr,
                "Usage: ./createfile <output_binary_file> <num_v> <num_e>\n");
        exit(EXIT_FAILURE);
    }

    create(argv[1], strtoull(argv[2], nullptr, 0), strtoull(argv[3], nullptr, 0));
    return EXIT_SUCCESS;
}