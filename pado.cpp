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
#include "pado.h"

using namespace PADO;

void test()
{
//	freopen("output.txt", "w", stdout); // test
//	const char filename[] = "input.dat";
	const char filename[] = "/Users/johnz/pppp/datasets/dblp/dblp";
	printf("Reading...\n");//test
	Graph G(filename);
//	G.print();
//	vector<idi> rank = {
//			14,
//			10,
//			7,
//			8,
//			15,
//			16,
//			4,
//			5,
//			6,
//			11,
//			17,
//			12,
//			1,
//			2,
//			9,
//			3,
//			18,
//			13,
//			19,
//			20
//	};

	printf("Ranking...\n");//test
	vector<idi> rank = VertexCentricPLL::make_rank(G);
//	for (idi v = 0; v < rank.size(); ++v) {
//		printf("vertices %llu: rank %llu\n", v, rank[v]);//test
//	}
	WallTimer timer("Labeling");
	printf("Labeling...\n");//test
	VertexCentricPLL(G, rank);
	timer.print_runtime();
}

int main(int argc, char *argv[]) {
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
	test();
	puts("Done!");
	return EXIT_SUCCESS;
}
