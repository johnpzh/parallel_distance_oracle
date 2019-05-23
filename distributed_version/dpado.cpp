//
// Created by Zhen Peng on 5/14/19.
//

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "dglobals.h"
#include "dgraph.h"

using namespace PADO;

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

    return EXIT_SUCCESS;
}

