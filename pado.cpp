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
//#include "pado.20190130.tmp.candidates_bitmap.h"
//#include "pado_weighted.20190111.batch_process_vectorized_dist_query.h"
//#include "pado_weighted.20181228.batch_process.h"
//#include "pado_weighted.20190122.labels_correction.h"
//#include "pado_weighted.20190126.tmp.vectorized_more_part_not_only_DQ.h"
//#include "pado_weighted.20190128.vectorized_with_roots_labels_buffer.h"
//#include "pado_para.h"
//#include "pado_para.20181106.tmp.scalability.h"
//#include "pado_para.20181115.tmp.parallel_bp.h"
//#include "pado_para.20190129.candidates_que.h"
//#include "pado_para.20190205.vec_DQ_vec_BPC.h"
//#include "pado_weighted_para.20190129.parallel.h"
//#include "pado_weighted_para.20190131.no_temp_queue_but_atomic_opts.h"
//#include "pado.20190130.vectorization.h"
//#include "pado.20190201.vec_with_extra_label_array.h"
//#include "pado.20190203.bp_checking_vec.h"
//#include "pado.20190207.seq_3-level-label_DQ.h"
//#include "pado.20190208.vec_3-level-label_DQ.h"

#include "pado_unw_unv_unp.h"
#include "pado_unw_vec_unp.h"
#include "pado_weighted_unv_unp.h"

using namespace PADO;

void pado(const char filename[])
{
	//printf("Reading...\n"); fflush(stdout);//test
//	WeightedGraph G(filename);
	Graph G(filename);
	//printf("Ranking...\n"); fflush(stdout);//test
	vector<idi> rank = G.make_rank();
	vector<idi> rank2id = G.id_transfer(rank);
	//printf("Labeling...\n"); fflush(stdout);//test

	//WeightedVertexCentricPLL VCPLL(G);
	//VCPLL.switch_labels_to_old_id(rank2id, rank);

//	NUM_THREADS = 40;
//	omp_set_num_threads(NUM_THREADS);
////	WeightedVertexCentricPLL VCPLL(G);
//	ParaVertexCentricPLL VCPLL(G);
	VertexCentricPLL<1024> VCPLL(G);
//	VCPLL.switch_labels_to_old_id(rank2id, rank);


//	for (inti t_num = 1; t_num <= 32; t_num *= 2) {
//		NUM_THREADS = t_num;
//		omp_set_num_threads(NUM_THREADS);
//		ParaVertexCentricPLL *VCPLL = new ParaVertexCentricPLL(G);
//		VCPLL->switch_labels_to_old_id(rank2id, rank);
//		delete VCPLL;
//		puts("");
//	}
//	{
//		NUM_THREADS = 40;
//		omp_set_num_threads(NUM_THREADS);
//		ParaVertexCentricPLL *VCPLL = new ParaVertexCentricPLL(G);
//		VCPLL->switch_labels_to_old_id(rank2id, rank);
//		delete VCPLL;
//		puts("");
//	}

	idi num_v = rank.size();
	VCPLL.order_labels(rank2id, rank);

//	// Test Input
//	{
//		idi a;
//		idi b;
//		while (std::cin >> a >> b) {
//			inti d = VCPLL.query_distance(a, b, num_v);
//			if (255 == d) {
//				d = 2147483647;
//			}
//			printf("%u\n", VCPLL.query_distance(a, b, num_v));
//		}
//	}

	// Benchmark
	{
		const inti num_queries = 1000000;
		vector< pair<idi, idi> > queries(num_queries);
		for (inti i = 0; i < num_queries; ++i) {
			queries[i].first = rand() % num_v;
			queries[i].second = rand() % num_v;
		}
		inti sum = 0;
		double query_time = -WallTimer::get_time_mark();
		for (inti i = 0; i < num_queries; ++i) {
			sum += VCPLL.query_distance(queries[i].first, queries[i].second, num_v);
//			VCPLL.query_distance(queries[i].first, queries[i].second, num_v);
		}
		query_time += WallTimer::get_time_mark();
		printf("test_sum: %u\n", sum);//test
		printf("query_time: %f seconds mean: %f microseconds\n", query_time, query_time / num_queries * 1E6);
		printf("label_length_larger_than_16: %'llu %.2f%%\n",
						VCPLL.length_larger_than_16.first,
						100.0 * VCPLL.length_larger_than_16.first / VCPLL.length_larger_than_16.second);
	}
}

void usage_print()
{
	fprintf(stderr,
			"Usage: ./pado <input_file> [-w 0|1] [-v 0|1] [-p 0|1]\n"
			"\t-w: 0 means unweighted graph (default), 1 means weighted graph\n"
			"\t-v: 0 means sequential version (default), 1 means AVX512 version (needs CPU support)\n"
			"\t-p: 0 means single-thread version (default), 1 means multithread version\n");
}

int main(int argc, char *argv[])
{
	if (argc < 2) {
		usage_print();
		exit(EXIT_FAILURE);
	}
	bool is_weighted = false;
	bool is_vectorized = false;
	bool is_multithread = false;
	int opt;
	while ((opt = getopt(argc, argv, "w:v:p:")) != -1) {
		switch(opt) {
		case 'w':
			if (strtoul(optarg, NULL, 0)) {
				is_weighted = true;
			}
			break;
		case 'v':
			if (strtoul(optarg, NULL, 0)) {
				is_vectorized = true;
			}
			break;
		case 'p':
			if (strtoul(optarg, NULL, 0)) {
				is_multithread = true;
			}
			break;
		default:
			fprintf(stderr, "Error: unknown opt %c.\n", opt);
			exit(EXIT_FAILURE);
		}
	}
	setvbuf(stdout, NULL, _IONBF, 0); //  Set stdout no buffer
//	pado(argv[1]);
	if (!is_weighted) {
		// Unweighted
		if (!is_vectorized) {
			// No vectorization
			if (!is_multithread) {
				// Single Thread
				Graph G(argv[1]);
				vector<idi> rank = G.make_rank();
				vector<idi> rank2id = G.id_transfer(rank);
				VertexCentricPLL<1024> VCPLL(G);
				VCPLL.switch_labels_to_old_id(rank2id, rank);
			} else {
				// Multithread
				// TODO
			}
		} else {
			// Vectorization
			if (!is_multithread) {
				// Single Thread
				Graph G(argv[1]);
				vector<idi> rank = G.make_rank();
				vector<idi> rank2id = G.id_transfer(rank);
				VertexCentricPLLVec<1024> VCPLLvec(G);
				VCPLLvec.switch_labels_to_old_id(rank2id, rank);
			} else {
				// Multithread
				// TODO
			}
		}
	} else {
		// Weighted
		if (!is_vectorized) {
			// No vectorization
			if (!is_multithread) {
				// Single Thread
				WeightedGraph G(argv[1]);
				vector<idi> rank = G.make_rank();
				vector<idi> rank2id = G.id_transfer(rank);
				WeightedVertexCentricPLL<512> VCPLL(G);
				VCPLL.switch_labels_to_old_id(rank2id, rank);
			} else {
				// Multithread
				// TODO
			}
		} else {
			// Vectorization
			if (!is_multithread) {
				// Single Thread
				// TODO
			} else {
				// Multithread
				// TODO
			}
		}
	}
	//printf("Done!\n");
	return EXIT_SUCCESS;
}
