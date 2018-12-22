//============================================================================
// Name        : pado.cpp
// Author      : Zhen Peng
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "globals.h"
#include "graph.h"
//#include "pado.h"
#include "pado_weighted.20181218.beginning.h"
//#include "pado_para.h"
//#include "pado_para.20181106.tmp.scalability.h"
//#include "pado_para.20181115.tmp.parallel_bp.h"

using namespace PADO;

void pado(const char filename[])
{
	printf("Reading...\n"); fflush(stdout);//test
	WeightedGraph G(filename);
	printf("Ranking...\n"); fflush(stdout);//test
	vector<idi> rank = G.make_rank();
//	for (idi v = 0; v < rank.size(); ++v) {
//		printf("vertices %u: rank %u\n", v, rank[v]);//test
//	}
	vector<idi> rank2id = G.id_transfer(rank);
	//G.print();
	printf("Labeling...\n"); fflush(stdout);//test
	WeightedVertexCentricPLL VCPLL(G);
	VCPLL.switch_labels_to_old_id(rank2id, rank);

//	NUM_THREADS = 1;
//	omp_set_num_threads(NUM_THREADS);
//	ParaVertexCentricPLL VCPLL(G);
//	VertexCentricPLL VCPLL(G);
//	VCPLL.switch_labels_to_old_id(rank2id, rank);


//	for (inti t_num = 1; t_num <= 32; t_num *= 2) {
//		NUM_THREADS = t_num;
//		omp_set_num_threads(NUM_THREADS);
//		ParaVertexCentricPLL VCPLL(G);
////		VCPLL.switch_labels_to_old_id(rank2id, rank);
//	}
//	{
//		NUM_THREADS = 40;
//		omp_set_num_threads(NUM_THREADS);
//		ParaVertexCentricPLL VCPLL(G);
////		VCPLL.switch_labels_to_old_id(rank2id, rank);
//	}
//	VCPLL.print();//test

}

int main(int argc, char *argv[])
{
//	// By Johnpzh
//	char separator = ' ';
//	//uint64_t kNum = 50;
//	bool is_directed = false;
//	uint64_t start_id = 1;
//	int opt;
//	if (argc < 3) {
//		fprintf(stderr,
//				"Usage: ./construct_index <input_data> <output_index> [-s | -t] [-d] [-i StartID]\n"
//				"	-s: separator is space (default)\n"
//				"	-t: separator is tab\n"
//				"	-d: is directed graph\n"
//				"	-i n: the start ID is n (default 0)\n");
//
//		exit(EXIT_FAILURE);
//	}
//	while ((opt = getopt(argc, argv, "stk:di:")) != -1) {
//		switch (opt) {
//		case 't':
//			separator = '\t';
//			break;
//		case 'd':
//			is_directed = true;
//			break;
//		case 'i':
//			start_id = strtoul(optarg, NULL, 0);
//			break;
//		default:
//			fprintf(stderr, "Error: unknown opt %c.\n", opt);
//			exit(EXIT_FAILURE);
//		}
//	}
//	// End by Johnpzh
//	test_bit();
	if (argc < 2) {
		fprintf(stderr,
				"Usage: ./pado <input_data>\n");
		exit(EXIT_FAILURE);
	}
	pado(argv[1]);
	printf("Done!\n");
	return EXIT_SUCCESS;
}
