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

void pado()
{
//	freopen("output.txt", "w", stdout); // test
//	const char filename[] = "input.dat";
//	const char filename[] = "/Users/johnz/pppp/datasets/dblp/dblp";
//	const char filename[] = "/Users/johnz/pppp/datasets/chicago/chicago";
//	const char filename[] = "/Users/johnz/pppp/datasets/wikitalk/wikitalk";
	const char filename[] = "tools/edgelist.txt";
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
	vector<idi> rank = G.make_rank();
//	for (idi v = 0; v < rank.size(); ++v) {
//		printf("vertices %u: rank %u\n", v, rank[v]);//test
//	}
	vector<idi> rank2id = G.id_transfer(rank);
//	G.print();
	WallTimer timer("Labeling");
	printf("Labeling...\n");//test
//	VertexCentricPLL(G, rank);
	VertexCentricPLL VCPLL(G);
	timer.print_runtime();
	VCPLL.switch_labels_to_old_id(rank2id);
	VCPLL.print();//test

	// Test for query
//	idi u, v;
//	while (std::cin >> u >> v) {
//		idi d = VCPLL.query(u,v);
//		if (d == 255) {
//			d = 2147483647;
//		}
//		printf("%llu\n", d);
//	}
}

void test_bit()
{
	BitArray B(128, 1099511627776 + 4294967296 + 65536 + 2 + 1);
//	void (* fun) (inti loc) = nullptr;
	auto fun = [] (inti loc) {
		printf("loc: %u\n", loc);
	};
	puts("=======");
	B.process_every_bit(fun);
//	B.unset_bit(32);
//	puts("=======");
//	B.process_every_bit(fun);
	B.set_bit(64);
	B.set_bit(127);
	B.set_bit(8);
	puts("=======");
	B.process_every_bit(fun);
	B.unset_bit(64);
	B.unset_bit(0);
	puts("=======");
	B.process_every_bit(fun);

	printf("loc %d set: %u\n", 64, B.is_bit_set(64));
	printf("loc %d set: %u\n", 127, B.is_bit_set(127));
	printf("loc %d set: %u\n", 8, B.is_bit_set(8));
	printf("loc %d set: %u\n", 63, B.is_bit_set(63));
//	vector<inti> locs = B.get_all_locs_set(128);
//	for (const auto &l : locs) {
//		printf(" %u", l);
//	}
//	puts("");
}

int main() {
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
	pado();
	puts("Done!");
	return EXIT_SUCCESS;
}
