#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "globals.h"
#include "graph.h"

#include "pado_unw_unv_unp.h"
#include "pado_unw_vec_unp.h"
#include "pado_unw_para_unv.h"
#include "pado_unw_para_vec.h"
#include "pado_weighted_unv_unp.h"
#include "pado_weighted_vec_unp.h"
#include "pado_weighted_para_unv.h"
#include "pado_weighted_para_vec.h"

using namespace PADO;

void usage_print()
{
	fprintf(stderr,
			"Usage: ./query_distance <input_index> [-w 0|1] [-v 0|1]\n"
			"\t-w: 0 for unweighted-graph-generated index (default), 1 for weighted graph\n"
			"\t-v: 0 for sequential-version-generated index (default), 1 for AVX512 version (needs CPU support)\n");
}

int main(int argc, char *argv[])
{
	string input_index;
	if (argc < 2) {
		usage_print();
		exit(EXIT_FAILURE);
	} else {
		input_index = string(argv[1]);
	}
	bool is_weighted = false;
	int opt;
	while ((opt = getopt(argc, argv, "w:")) != -1) {
		switch(opt) {
		case 'w':
			if (strtoul(optarg, NULL, 0)) {
				is_weighted = true;
			}
			break;
		default:
			fprintf(stderr, "Error: unknown opt %c.\n", opt);
			exit(EXIT_FAILURE);
		}
	}
	setvbuf(stdout, NULL, _IONBF, 0); //  Set stdout no buffer
	setlocale(LC_NUMERIC, ""); // Print large integer in comma format.
//	pado(argv[1]);
	if (!is_weighted) {
		// Unweighted
		printf("unw unv\n");//test
		// Single Thread
		VertexCentricPLL<1024> *VCPLL = new VertexCentricPLL<1024>(); // 1024 is the batch size
		VCPLL->load_index_from_file(input_index.c_str());
		idi a;
		idi b;
		while (std::cin >> a >> b) {
			printf("%u\n", VCPLL->query_distance(a, b));
		}
		delete VCPLL;
 	} else {
		// Weighted
 		printf("W VEC\n");//test
 		// Single Thread
 		WeightedVertexCentricPLLVec<512> *VCPLL = new WeightedVertexCentricPLLVec<512>();
		VCPLL->load_index_from_file(input_index.c_str());
		idi a;
		idi b;
		while (std::cin >> a >> b) {
			printf("%u\n", VCPLL->query_distance(a, b));
		}
		delete VCPLL;
	}
	//printf("Done!\n");
	return EXIT_SUCCESS;
}
