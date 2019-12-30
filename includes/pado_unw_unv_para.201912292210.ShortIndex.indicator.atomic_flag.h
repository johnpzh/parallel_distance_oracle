/*
 * pado.h
 *
 *  Created on: Sep 4, 2018
 *      Author: Zhen Peng
 */

#ifndef INCLUDES_PADO_UNW_PARA_UNV_H_
#define INCLUDES_PADO_UNW_PARA_UNV_H_

#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <iostream>
#include <limits.h>
#include <xmmintrin.h>
#include <bitset>
#include <cmath>
#include <atomic>
#include "globals.h"
#include "graph.h"
#include <omp.h>

using std::vector;
using std::unordered_map;
using std::map;
using std::bitset;
using std::stable_sort;
using std::min;
using std::fill;

namespace PADO {
//inti NUM_THREADS = 4;
//const inti BATCH_SIZE = 1024; // The size for regular batch and bit array.
//const inti BITPARALLEL_SIZE = 50;
//const inti THRESHOLD_PARALLEL = 80;



//// Batch based processing, 09/11/2018
template<inti BATCH_SIZE = 1024>
class ParaVertexCentricPLL {
private:
    static const inti BITPARALLEL_SIZE = 50;
    idi num_v_ = 0;
    const inti THRESHOLD_PARALLEL = 80;
    // Structure for the type of label
    struct IndexType {
        struct Batch {
            idi batch_id; // Batch ID
            idi start_index; // Index to the array distances where the batch starts
            inti size; // Number of distances element in this batch

            Batch(idi batch_id_, idi start_index_, inti size_) :
                    batch_id(batch_id_), start_index(start_index_), size(size_)
            {
                ;
            }
        };

        struct DistanceIndexType {
            idi start_index; // Index to the array vertices where the same-ditance vertices start
            inti size; // Number of the same-distance vertices
            smalli dist; // The real distance

            DistanceIndexType(idi start_index_, inti size_, smalli dist_) :
                    start_index(start_index_), size(size_), dist(dist_)
            {
                ;
            }
        };

        smalli bp_dist[BITPARALLEL_SIZE];
        uint64_t bp_sets[BITPARALLEL_SIZE][2];  // [0]: S^{-1}, [1]: S^{0}

        vector<Batch> batches; // Batch info
        vector<DistanceIndexType> distances; // Distance info
        vector<idi> vertices; // Vertices in the label, preresented as temperory ID
    }; //__attribute__((aligned(64)));

    // Structure for the type of temporary label
    struct ShortIndex {
        // I use BATCH_SIZE + 1 bit for indicator bit array.
        // The v.indicator[BATCH_SIZE] is set if in current batch v has got any new labels already.
        // In this way, it helps update_label_indices() and can be reset along with other indicator elements.
//        bitset<BATCH_SIZE + 1> indicator; // Global indicator, indicator[r] (0 <= r < BATCH_SIZE) is set means root r once selected as candidate already
//        std::vector<std::atomic_bool> indicator;
        std::vector<uint8_t> indicator = std::vector<uint8_t>(BATCH_SIZE + 1, 0);

        // Use a queue to store candidates
        vector<inti> candidates_que = vector<inti>(BATCH_SIZE);
        inti end_candidates_que = 0;
        vector<uint8_t> is_candidate = vector<uint8_t>(BATCH_SIZE, 0);

//        ShortIndex()
//        {
//            indicator.resize(BATCH_SIZE + 1);
//            indicator_reset();
//        }

        void indicator_reset()
        {
            const idi bound = indicator.size();
            std::fill(indicator.begin(), indicator.end(), 0);
//#pragma omp parallel for
//            for (idi i = 0; i < bound; ++i) {
//                indicator[i].store(false, std::memory_order_relaxed);
//            }
        }

    }; //__attribute__((aligned(64)));

    // Structure of the public ordered index for distance queries.
    struct IndexOrdered {
        weighti bp_dist[BITPARALLEL_SIZE];
        uint64_t bp_sets[BITPARALLEL_SIZE][2]; // [0]: S^{-1}, [1]: S^{0}

        vector<idi> label_id;
        vector<weighti> label_dists;
    };

    vector<IndexType> L;
    vector<IndexOrdered> Index; // Ordered labels for original vertex ID

    void construct(const Graph &G);

    inline void bit_parallel_labeling(
            const Graph &G,
            vector<IndexType> &L,
            vector<uint8_t> &used_bp_roots);
//	inline void bit_parallel_labeling(
//				const Graph &G,
//				vector<IndexType> &L,
//				vector<bool> &used_bp_roots);

    inline void batch_process(
            const Graph &G,
            idi b_id,
            idi roots_start, // start id of roots
            inti roots_size, // how many roots in the batch
            vector<IndexType> &L,
            const vector<uint8_t> &used_bp_roots,
            vector<idi> &active_queue,
            idi &end_active_queue,
            vector<idi> &candidate_queue,
            idi &end_candidate_queue,
            vector<ShortIndex> &short_index,
            vector<vector<smalli> > &dist_matrix,
            vector<uint8_t> &got_candidates,
            vector<uint8_t> &is_active,
            vector<idi> &once_candidated_queue,
            idi &end_once_candidated_queue,
            vector<uint8_t> &once_candidated);
//	inline void batch_process(
//			const Graph &G,
//			idi b_id,
//			idi root_start,
//			inti roots_size,
//			vector<IndexType> &L,
//			const vector<bool> &used_bp_roots);


    inline void initialize(
            vector<ShortIndex> &short_index,
            vector<vector<smalli> > &dist_matrix,
            vector<idi> &active_queue,
            idi &end_active_queue,
            vector<idi> &once_candidated_queue,
            idi &end_once_candidated_queue,
//				vector<bool> &once_candidated,
            vector<uint8_t> &once_candidated,
            idi b_id,
            idi roots_start,
            inti roots_size,
            vector<IndexType> &L,
            const vector<uint8_t> &used_bp_roots);

    inline void push_labels(
            idi v_head,
            idi roots_start,
            const Graph &G,
            const vector<IndexType> &L,
            vector<ShortIndex> &short_index,
//				vector<idi> &candidate_queue,
//				idi &end_candidate_queue,
            vector<idi> &tmp_candidate_queue,
            idi &size_tmp_candidate_queue,
            const idi offset_tmp_queue,
//				idi &offset_tmp_candidate_queue,
//				vector<bool> &got_candidates,
            vector<uint8_t> &got_candidates,
            vector<idi> &once_candidated_queue,
            idi &end_once_candidated_queue,
//				vector<bool> &once_candidated,
            vector<uint8_t> &once_candidated,
            const vector<uint8_t> &used_bp_roots,
            smalli iter);

    inline bool distance_query(
            idi cand_root_id,
            idi v_id,
            idi roots_start,
            const vector<IndexType> &L,
            const vector<vector<smalli> > &dist_matrix,
            smalli iter);

    inline void insert_label_only(
            idi cand_root_id,
            idi v_id,
            idi roots_start,
            inti roots_size,
            vector<IndexType> &L,
            vector<vector<smalli> > &dist_matrix,
            smalli iter);

    inline void update_label_indices(
            idi v_id,
            idi inserted_count,
            vector<IndexType> &L,
            vector<ShortIndex> &short_index,
            idi b_id,
            smalli iter);

    inline void reset_at_end(
            idi roots_start,
            inti roots_size,
            vector<IndexType> &L,
            vector<vector<smalli> > &dist_matrix);

    // Some parallel interfaces
    inline idi prefix_sum_for_offsets(
            vector<idi> &offsets);

    template<typename T>
    inline void collect_into_queue(
            vector<T> &tmp_queue,
            vector<idi> &offsets_tmp_queue, // the locations in tmp_queue for writing from tmp_queue
            vector<idi> &offsets_queue, // the locations in queue for writing into queue.
            idi num_elements, // total number of elements which need to be added from tmp_queue to queue
            vector<T> &queue,
            idi &end_queue);

    template<typename T, typename Int>
    inline void TS_enqueue(
            vector<T> &queue,
            Int &end_queue,
            const T &e);

    // Test only
//	uint64_t normal_hit_count = 0;
//	uint64_t bp_hit_count = 0;
//	uint64_t total_check_count = 0;
//	double initializing_time = 0;
//	double candidating_time = 0;
//	double adding_time = 0;
//	double distance_query_time = 0;
//	double init_index_time = 0;
//	double init_dist_matrix_time = 0;
//	double init_start_reset_time = 0;
//	double init_indicators_time = 0;

//#ifdef PROFILE
//	vector<double> thds_adding_time = vector<double>(80, 0.0);
//	vector<uint64_t> thds_adding_count = vector<uint64_t>(80, 0);
//	L2CacheMissRate cache_miss;
//#endif
//    vector<ShortIndex> tmp_short_index;
//    vector<ShortIndex> now_short_index;
    // End test



public:
    ParaVertexCentricPLL() = default;

    ParaVertexCentricPLL(const Graph &G);

    weighti query(
            idi u,
            idi v);

    void print();

    void switch_labels_to_old_id(
            const vector<idi> &rank2id,
            const vector<idi> &rank);

    void store_index_to_file(
            const char *filename,
            const vector<idi> &rank);

    void load_index_from_file(
            const char *filename);

    void order_labels(
            const vector<idi> &rank2id,
            const vector<idi> &rank);

    weighti query_distance(
            idi a,
            idi b);

}; // class ParaVertexCentricPLL

template<inti BATCH_SIZE>
const inti ParaVertexCentricPLL<BATCH_SIZE>::BITPARALLEL_SIZE;

template<inti BATCH_SIZE>
ParaVertexCentricPLL<BATCH_SIZE>::ParaVertexCentricPLL(const Graph &G)
{
    construct(G);
}

template<inti BATCH_SIZE>
inline void ParaVertexCentricPLL<BATCH_SIZE>::bit_parallel_labeling(
        const Graph &G,
        vector<IndexType> &L,
        vector<uint8_t> &used_bp_roots) // CAS needs array
{
    idi num_v = G.get_num_v();
    idi num_e = G.get_num_e();

    if (num_v <= BITPARALLEL_SIZE) {
//	if (true) {}
        // Sequential version
        std::vector<weighti> tmp_d(num_v); // distances from the root to every v
        std::vector<std::pair<uint64_t, uint64_t> > tmp_s(num_v); // first is S_r^{-1}, second is S_r^{0}
        std::vector<idi> que(num_v); // active queue
        std::vector<std::pair<idi, idi> > sibling_es(
                num_e); // siblings, their distances to the root are equal (have difference of 0)
        std::vector<std::pair<idi, idi> > child_es(
                num_e); // child and father, their distances to the root have difference of 1.
        idi r = 0; // root r
        for (inti i_bpspt = 0; i_bpspt < BITPARALLEL_SIZE; ++i_bpspt) {
            while (r < num_v && used_bp_roots[r]) {
                ++r;
            }
            if (r == num_v) {
                for (idi v = 0; v < num_v; ++v) {
                    L[v].bp_dist[i_bpspt] = SMALLI_MAX;
                }
                continue;
            }
            used_bp_roots[r] = 1;

            fill(tmp_d.begin(), tmp_d.end(), SMALLI_MAX);
            fill(tmp_s.begin(), tmp_s.end(), std::make_pair(0, 0));

            idi que_t0 = 0, que_t1 = 0, que_h = 0;
            que[que_h++] = r;
            tmp_d[r] = 0;
            que_t1 = que_h;

            int ns = 0; // number of selected neighbor, default 64
            // the edge of one vertex in G is ordered decreasingly to rank, lower rank first, so here need to traverse edges backward
            // There was a bug cost countless time: the unsigned iterator i might decrease to zero and then flip to the INF.
            //		idi i_bound = G.vertices[r] - 1;
            //		idi i_start = i_bound + G.out_degrees[r];
            //		for (idi i = i_start; i > i_bound; --i) {}
            idi d_i_bound = G.out_degrees[r];
            idi i_start = G.vertices[r] + d_i_bound - 1;
            for (idi d_i = 0; d_i < d_i_bound; ++d_i) {
                idi i = i_start - d_i;
                idi v = G.out_edges[i];
                if (!used_bp_roots[v]) {
                    used_bp_roots[v] = 1;
                    // Algo3:line4: for every v in S_r, (dist[v], S_r^{-1}[v], S_r^{0}[v]) <- (1, {v}, empty_set)
                    que[que_h++] = v;
                    tmp_d[v] = 1;
                    tmp_s[v].first = 1ULL << ns;
                    if (++ns == 64) break;
                }
            }

            for (weighti d = 0; que_t0 < que_h; ++d) {
                idi num_sibling_es = 0, num_child_es = 0;

                for (idi que_i = que_t0; que_i < que_t1; ++que_i) {
                    idi v = que[que_i];
                    idi i_start = G.vertices[v];
                    idi i_bound = i_start + G.out_degrees[v];
                    for (idi i = i_start; i < i_bound; ++i) {
                        idi tv = G.out_edges[i];
                        weighti td = d + 1;

                        if (d > tmp_d[tv]) { ;
                        } else if (d == tmp_d[tv]) {
                            if (v < tv) { // ??? Why need v < tv !!! Because it's a undirected graph.
                                sibling_es[num_sibling_es].first = v;
                                sibling_es[num_sibling_es].second = tv;
                                ++num_sibling_es;
//								tmp_s[v].second |= tmp_s[tv].first;
//								tmp_s[tv].second |= tmp_s[v].first;
                            }
                        } else { // d < tmp_d[tv]
                            if (tmp_d[tv] == SMALLI_MAX) {
                                que[que_h++] = tv;
                                tmp_d[tv] = td;
                            }
                            child_es[num_child_es].first = v;
                            child_es[num_child_es].second = tv;
                            ++num_child_es;
//							tmp_s[tv].first  |= tmp_s[v].first;
//							tmp_s[tv].second |= tmp_s[v].second;
                        }
                    }
                }

                for (idi i = 0; i < num_sibling_es; ++i) {
                    idi v = sibling_es[i].first, w = sibling_es[i].second;
                    tmp_s[v].second |= tmp_s[w].first;
                    tmp_s[w].second |= tmp_s[v].first;
                }
                for (idi i = 0; i < num_child_es; ++i) {
                    idi v = child_es[i].first, c = child_es[i].second;
                    tmp_s[c].first |= tmp_s[v].first;
                    tmp_s[c].second |= tmp_s[v].second;
                }

                que_t0 = que_t1;
                que_t1 = que_h;
            }

            for (idi v = 0; v < num_v; ++v) {
                L[v].bp_dist[i_bpspt] = tmp_d[v];
                L[v].bp_sets[i_bpspt][0] = tmp_s[v].first; // S_r^{-1}
                L[v].bp_sets[i_bpspt][1] = tmp_s[v].second &
                                           ~tmp_s[v].first; // Only need those r's neighbors who are not already in S_r^{-1}
            }
        }
    } else {
        // Parallel version: Naive parallel enqueue
        std::vector<weighti> tmp_d(num_v); // distances from the root to every v
        std::vector<std::pair<uint64_t, uint64_t> > tmp_s(num_v); // first is S_r^{-1}, second is S_r^{0}
        std::vector<idi> que(num_v); // active queue
        std::vector<std::pair<idi, idi> > sibling_es(
                num_e); // siblings, their distances to the root are equal (have difference of 0)
        std::vector<std::pair<idi, idi> > child_es(
                num_e); // child and father, their distances to the root have difference of 1.
        idi r = 0; // root r
        for (inti i_bpspt = 0; i_bpspt < BITPARALLEL_SIZE; ++i_bpspt) {
            while (r < num_v && used_bp_roots[r]) {
                ++r;
            }
            if (r == num_v) {
                for (idi v = 0; v < num_v; ++v) {
                    L[v].bp_dist[i_bpspt] = SMALLI_MAX;
                }
                continue;
            }
            used_bp_roots[r] = 1;

            fill(tmp_d.begin(), tmp_d.end(), SMALLI_MAX);
            fill(tmp_s.begin(), tmp_s.end(), std::make_pair(0, 0));

            idi que_t0 = 0, que_t1 = 0, que_h = 0;
            que[que_h++] = r;
            tmp_d[r] = 0;
            que_t1 = que_h;

            int ns = 0; // number of selected neighbor, default 64
            // the edge of one vertex in G is ordered decreasingly to rank, lower rank first, so here need to traverse edges backward
            // There was a bug cost countless time: the unsigned iterator i might decrease to zero and then flip to the INF.
            //		idi i_bound = G.vertices[r] - 1;
            //		idi i_start = i_bound + G.out_degrees[r];
            //		for (idi i = i_start; i > i_bound; --i) {}
            idi d_i_bound = G.out_degrees[r];
            idi i_start = G.vertices[r] + d_i_bound - 1;
            for (idi d_i = 0; d_i < d_i_bound; ++d_i) {
                idi i = i_start - d_i;
                idi v = G.out_edges[i];
                if (!used_bp_roots[v]) {
                    used_bp_roots[v] = 1;
                    // Algo3:line4: for every v in S_r, (dist[v], S_r^{-1}[v], S_r^{0}[v]) <- (1, {v}, empty_set)
                    que[que_h++] = v;
                    tmp_d[v] = 1;
                    tmp_s[v].first = 1ULL << ns;
                    if (++ns == 64) break;
                }
            }

            for (weighti d = 0; que_t0 < que_h; ++d) {
                idi num_sibling_es = 0, num_child_es = 0;

                for (idi que_i = que_t0; que_i < que_t1; ++que_i) {
                    idi v = que[que_i];
                    idi i_start = G.vertices[v];
                    idi i_bound = i_start + G.out_degrees[v];
                    for (idi i = i_start; i < i_bound; ++i) {
                        idi tv = G.out_edges[i];
                        weighti td = d + 1;

                        if (d > tmp_d[tv]) { ;
                        } else if (d == tmp_d[tv]) {
                            if (v < tv) { // ??? Why need v < tv !!! Because it's a undirected graph.
                                sibling_es[num_sibling_es].first = v;
                                sibling_es[num_sibling_es].second = tv;
                                ++num_sibling_es;
//								tmp_s[v].second |= tmp_s[tv].first;
//								tmp_s[tv].second |= tmp_s[v].first;
                            }
                        } else { // d < tmp_d[tv]
                            if (tmp_d[tv] == SMALLI_MAX) {
                                que[que_h++] = tv;
                                tmp_d[tv] = td;
                            }
                            child_es[num_child_es].first = v;
                            child_es[num_child_es].second = tv;
                            ++num_child_es;
//							tmp_s[tv].first  |= tmp_s[v].first;
//							tmp_s[tv].second |= tmp_s[v].second;
                        }
                    }
                }

                for (idi i = 0; i < num_sibling_es; ++i) {
                    idi v = sibling_es[i].first, w = sibling_es[i].second;
                    tmp_s[v].second |= tmp_s[w].first;
                    tmp_s[w].second |= tmp_s[v].first;
                }
                for (idi i = 0; i < num_child_es; ++i) {
                    idi v = child_es[i].first, c = child_es[i].second;
                    tmp_s[c].first |= tmp_s[v].first;
                    tmp_s[c].second |= tmp_s[v].second;
                }

                que_t0 = que_t1;
                que_t1 = que_h;
            }

#pragma omp parallel for
            for (idi v = 0; v < num_v; ++v) {
                L[v].bp_dist[i_bpspt] = tmp_d[v];
//				L[v].bp_sets_0[i_bpspt] = tmp_s[v].first; // S_r^{-1}
//				L[v].bp_sets_1[i_bpspt] = tmp_s[v].second & ~tmp_s[v].first; // Only need those r's neighbors who are not already in S_r^{-1}
                L[v].bp_sets[i_bpspt][0] = tmp_s[v].first; // S_r^{-1}
                L[v].bp_sets[i_bpspt][1] = tmp_s[v].second &
                                           ~tmp_s[v].first; // Only need those r's neighbors who are not already in S_r^{-1}
            }
        }
    }
}
//inline void ParaVertexCentricPLL::bit_parallel_labeling(
//			const Graph &G,
//			vector<IndexType> &L,
//			vector<uint8_t> &used_bp_roots)
//{
//	idi num_v = G.get_num_v();
//	idi num_e = G.get_num_e();
//
////	std::vector<smalli> tmp_d(num_v); // distances from the root to every v
//	smalli *tmp_d = (smalli *) malloc(num_v * sizeof(smalli));
//	std::vector<std::pair<uint64_t, uint64_t> > tmp_s(num_v); // first is S_r^{-1}, second is S_r^{0}
//	std::vector<idi> que(num_v); // active queue
//	std::vector< std::pair<idi, idi> > sibling_es(num_e); // siblings, their distances to the root are equal (have difference of 0)
//	std::vector< std::pair<idi, idi> > child_es(num_e); // child and father, their distances to the root have difference of 1.
//
//	idi r = 0; // root r
//	for (inti i_bpspt = 0; i_bpspt < BITPARALLEL_SIZE; ++i_bpspt) {
//		while (r < num_v && used_bp_roots[r]) {
//			++r;
//		}
//		if (r == num_v) {
//			for (idi v = 0; v < num_v; ++v) {
//				L[v].bp_dist[i_bpspt] = SMALLI_MAX;
//			}
//			continue;
//		}
//		used_bp_roots[r] = 1;
//
////		fill(tmp_d.begin(), tmp_d.end(), SMALLI_MAX);
//		memset(tmp_d, (uint8_t) -1, num_v * sizeof(smalli));
//		fill(tmp_s.begin(), tmp_s.end(), std::make_pair(0, 0));
//
//		idi que_t0 = 0, que_t1 = 0, que_h = 0;
//		que[que_h++] = r;
//		tmp_d[r] = 0;
//		que_t1 = que_h;
//
//		int ns = 0; // number of selected neighbor, default 64
//		// the edge of one vertex in G is ordered decreasingly to rank, lower rank first, so here need to traverse edges backward
//		idi i_bound = G.vertices[r] - 1;
//		idi i_start = i_bound + G.out_degrees[r];
//		for (idi i = i_start; i > i_bound; --i) {
//			idi v = G.out_edges[i];
//			if (!used_bp_roots[v]) {
//				used_bp_roots[v] = 1;
//				// Algo3:line4: for every v in S_r, (dist[v], S_r^{-1}[v], S_r^{0}[v]) <- (1, {v}, empty_set)
//				que[que_h++] = v;
//				tmp_d[v] = 1;
//				tmp_s[v].first = 1ULL << ns;
//				if (++ns == 64) break;
//			}
//		}
//
//		for (smalli d = 0; que_t0 < que_h; ++d) {
//			idi num_sibling_es = 0, num_child_es = 0;
//
//			// For parallel adding to que
//			idi que_size = que_t1 - que_t0;
//			vector<idi> offsets_tmp_queue(que_size);
//#pragma omp parallel for
//			for (idi i_q = 0; i_q < que_size; ++i_q) {
//				offsets_tmp_queue[i_q] = G.out_degrees[que[que_t0 + i_q]];
//			}
//			idi num_neighbors = prefix_sum_for_offsets(offsets_tmp_queue);
//			vector<idi> tmp_que(num_neighbors);
//			vector<idi> sizes_tmp_que(que_size, 0);
//			// For parallel adding to sibling_es
//			vector< pair<idi, idi> > tmp_sibling_es(num_neighbors);
//			vector<idi> sizes_tmp_sibling_es(que_size, 0);
//			// For parallel adding to child_es
//			vector< pair<idi, idi> > tmp_child_es(num_neighbors);
//			vector<idi> sizes_tmp_child_es(que_size, 0);
//
//#pragma omp parallel for
//			for (idi que_i = que_t0; que_i < que_t1; ++que_i) {
//				idi tmp_que_i = que_i - que_t0; // location in the tmp_que
//				idi v = que[que_i];
//				idi i_start = G.vertices[v];
//				idi i_bound = i_start + G.out_degrees[v];
//				for (idi i = i_start; i < i_bound; ++i) {
//					idi tv = G.out_edges[i];
//					smalli td = d + 1;
//
//					if (d > tmp_d[tv]) {
//						;
//					}
//					else if (d == tmp_d[tv]) {
//						if (v < tv) { // ??? Why need v < tv !!! Because it's a undirected graph.
//							idi &size_in_group = sizes_tmp_sibling_es[tmp_que_i];
//							tmp_sibling_es[offsets_tmp_queue[tmp_que_i] + size_in_group].first = v;
//							tmp_sibling_es[offsets_tmp_queue[tmp_que_i] + size_in_group].second = tv;
//							++size_in_group;
////							sibling_es[num_sibling_es].first  = v;
////							sibling_es[num_sibling_es].second = tv;
////							++num_sibling_es;
//						}
//					} else { // d < tmp_d[tv]
//						if (tmp_d[tv] == SMALLI_MAX) {
//							if (CAS(tmp_d + tv, SMALLI_MAX, td)) { // tmp_d[tv] = td
//								tmp_que[offsets_tmp_queue[tmp_que_i] + sizes_tmp_que[tmp_que_i]++] = tv;
//							}
//						}
////						if (tmp_d[tv] == SMALLI_MAX) {
////							que[que_h++] = tv;
////							tmp_d[tv] = td;
////						}
//						idi &size_in_group = sizes_tmp_child_es[tmp_que_i];
//						tmp_child_es[offsets_tmp_queue[tmp_que_i] + size_in_group].first = v;
//						tmp_child_es[offsets_tmp_queue[tmp_que_i] + size_in_group].second = tv;
//						++size_in_group;
////						child_es[num_child_es].first  = v;
////						child_es[num_child_es].second = tv;
////						++num_child_es;
//					}
//				}
//			}
//
//			// From tmp_sibling_es to sibling_es
//			idi total_sizes_tmp_queue = prefix_sum_for_offsets(sizes_tmp_sibling_es);
//			collect_into_queue(
//						tmp_sibling_es,
//						offsets_tmp_queue,
//						sizes_tmp_sibling_es,
//						total_sizes_tmp_queue,
//						sibling_es,
//						num_sibling_es);
//
//#pragma omp parallel for
//			for (idi i = 0; i < num_sibling_es; ++i) {
//				idi v = sibling_es[i].first, w = sibling_es[i].second;
//				__sync_or_and_fetch(&tmp_s[v].second, tmp_s[w].first);
//				__sync_or_and_fetch(&tmp_s[w].second, tmp_s[v].first);
////				tmp_s[v].second |= tmp_s[w].first;
////				tmp_s[w].second |= tmp_s[v].first;
//			}
//
//			// From tmp_child_es to child_es
//			total_sizes_tmp_queue = prefix_sum_for_offsets(sizes_tmp_child_es);
//			collect_into_queue(
//						tmp_child_es,
//						offsets_tmp_queue,
//						sizes_tmp_child_es,
//						total_sizes_tmp_queue,
//						child_es,
//						num_child_es);
//
//#pragma omp parallel for
//			for (idi i = 0; i < num_child_es; ++i) {
//				idi v = child_es[i].first, c = child_es[i].second;
//				__sync_or_and_fetch(&tmp_s[c].first, tmp_s[v].first);
//				__sync_or_and_fetch(&tmp_s[c].second, tmp_s[v].second);
////				tmp_s[c].first  |= tmp_s[v].first;
////				tmp_s[c].second |= tmp_s[v].second;
//			}
//
//			// From tmp_que to que
//			total_sizes_tmp_queue = prefix_sum_for_offsets(sizes_tmp_que);
//			collect_into_queue(
//						tmp_que,
//						offsets_tmp_queue,
//						sizes_tmp_que,
//						total_sizes_tmp_queue,
//						que,
//						que_h);
//
//			que_t0 = que_t1;
//			que_t1 = que_h;
//		}
//
//#pragma omp parallel for
//		for (idi v = 0; v < num_v; ++v) {
//			L[v].bp_dist[i_bpspt] = tmp_d[v];
//			L[v].bp_sets[i_bpspt][0] = tmp_s[v].first; // S_r^{-1}
//			L[v].bp_sets[i_bpspt][1] = tmp_s[v].second & ~tmp_s[v].first; // Only need those r's neighbors who are not already in S_r^{-1}
//		}
//	}
//
//	free(tmp_d);
//}


// Function for initializing at the begin of a batch
// For a batch, initialize the temporary labels and real labels of roots;
// traverse roots' labels to initialize distance buffer;
// unset flag arrays is_active and got_labels
template<inti BATCH_SIZE>
inline void ParaVertexCentricPLL<BATCH_SIZE>::initialize(
        vector<ShortIndex> &short_index,
        vector<vector<smalli> > &dist_matrix,
        vector<idi> &active_queue,
        idi &end_active_queue,
        vector<idi> &once_candidated_queue,
        idi &end_once_candidated_queue,
//			vector<bool> &once_candidated,
        vector<uint8_t> &once_candidated,
        idi b_id,
        idi roots_start,
        inti roots_size,
        vector<IndexType> &L,
        const vector<uint8_t> &used_bp_roots)
{
    idi roots_bound = roots_start + roots_size;
//	init_start_reset_time -= WallTimer::get_time_mark();
    // TODO: parallel enqueue
    {
        // active_queue
        for (idi r_real_id = roots_start; r_real_id < roots_bound; ++r_real_id) {
            if (!used_bp_roots[r_real_id]) {
                active_queue[end_active_queue++] = r_real_id;
            }
        }
    }
//	init_start_reset_time += WallTimer::get_time_mark();
//	init_index_time -= WallTimer::get_time_mark();
    // Short_index
    {
//		init_indicators_time -= WallTimer::get_time_mark();
        if (end_once_candidated_queue >= THRESHOLD_PARALLEL) {
#pragma omp parallel for
            for (idi v_i = 0; v_i < end_once_candidated_queue; ++v_i) {
                idi v = once_candidated_queue[v_i];
//                short_index[v].indicator.reset();
                short_index[v].indicator_reset();
                once_candidated[v] = 0;
            }
        } else {
            for (idi v_i = 0; v_i < end_once_candidated_queue; ++v_i) {
                idi v = once_candidated_queue[v_i];
//                short_index[v].indicator.reset();
                short_index[v].indicator_reset();
                once_candidated[v] = 0;
            }
        }
//#pragma omp parallel for
//		for (idi v_i = 0; v_i < end_once_candidated_queue; ++v_i) {
//			idi v = once_candidated_queue[v_i];
//			short_index[v].indicator.reset();
//			once_candidated[v] = 0;
//		}
        end_once_candidated_queue = 0;
        if (roots_size >= THRESHOLD_PARALLEL) {
#pragma omp parallel for
            for (idi v = roots_start; v < roots_bound; ++v) {
                if (!used_bp_roots[v]) {
//                    short_index[v].indicator.set(v - roots_start);
//                    short_index[v].indicator.set(BATCH_SIZE); // v got labels
                    short_index[v].indicator[v - roots_start] = 1;
                    short_index[v].indicator[BATCH_SIZE] = 1; // v got labels
                }
            }
        } else {
            for (idi v = roots_start; v < roots_bound; ++v) {
                if (!used_bp_roots[v]) {
//                    short_index[v].indicator.set(v - roots_start);
//                    short_index[v].indicator.set(BATCH_SIZE); // v got labels
                    short_index[v].indicator[v - roots_start] = 1;
                    short_index[v].indicator[BATCH_SIZE] = 1; // v got labels
                }
            }
        }
//		for (idi v = roots_start; v < roots_bound; ++v) {
//			if (!used_bp_roots[v]) {
//				short_index[v].indicator.set(v - roots_start);
//				short_index[v].indicator.set(BATCH_SIZE); // v got labels
//			}
//		}
//		init_indicators_time += WallTimer::get_time_mark();
    }
//
    // Real Index
    {
        if (roots_size >= THRESHOLD_PARALLEL) {
#pragma omp parallel for
            for (idi r_id = 0; r_id < roots_size; ++r_id) {
                if (used_bp_roots[r_id + roots_start]) {
                    continue;
                }
                IndexType &Lr = L[r_id + roots_start];
                Lr.batches.push_back(IndexType::Batch(
                        b_id, // Batch ID
                        Lr.distances.size(), // start_index
                        1)); // size
                Lr.distances.push_back(IndexType::DistanceIndexType(
                        Lr.vertices.size(), // start_index
                        1, // size
                        0)); // dist
                Lr.vertices.push_back(r_id);
            }
        } else {
            for (idi r_id = 0; r_id < roots_size; ++r_id) {
                if (used_bp_roots[r_id + roots_start]) {
                    continue;
                }
                IndexType &Lr = L[r_id + roots_start];
                Lr.batches.push_back(IndexType::Batch(
                        b_id, // Batch ID
                        Lr.distances.size(), // start_index
                        1)); // size
                Lr.distances.push_back(IndexType::DistanceIndexType(
                        Lr.vertices.size(), // start_index
                        1, // size
                        0)); // dist
                Lr.vertices.push_back(r_id);
            }
        }
//		for (idi r_id = 0; r_id < roots_size; ++r_id) {
//			if (used_bp_roots[r_id + roots_start]) {
//				continue;
//			}
//			IndexType &Lr = L[r_id + roots_start];
//			Lr.batches.push_back(IndexType::Batch(
//												b_id, // Batch ID
//												Lr.distances.size(), // start_index
//												1)); // size
//			Lr.distances.push_back(IndexType::DistanceIndexType(
//												Lr.vertices.size(), // start_index
//												1, // size
//												0)); // dist
//			Lr.vertices.push_back(r_id);
//		}
    }
//	init_index_time += WallTimer::get_time_mark();
//	init_dist_matrix_time -= WallTimer::get_time_mark();
    // Dist_matrix
    {
        if (roots_size >= THRESHOLD_PARALLEL) {
// schedule dynamic is slower
#pragma omp parallel for
            for (idi r_id = 0; r_id < roots_size; ++r_id) {
                if (used_bp_roots[r_id + roots_start]) {
                    continue;
                }
                IndexType &Lr = L[r_id + roots_start];
                inti b_i_bound = Lr.batches.size();
                _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
                _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
                _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
                for (inti b_i = 0; b_i < b_i_bound; ++b_i) {
                    idi id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
                    idi dist_start_index = Lr.batches[b_i].start_index;
                    idi dist_bound_index = dist_start_index + Lr.batches[b_i].size;
                    // Traverse dist_matrix
                    for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                        idi v_start_index = Lr.distances[dist_i].start_index;
                        idi v_bound_index = v_start_index + Lr.distances[dist_i].size;
                        smalli dist = Lr.distances[dist_i].dist;
                        for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                            dist_matrix[r_id][Lr.vertices[v_i] + id_offset] = dist;
                        }
                    }
                }
            }
        } else {
            inti b_i_bound;
            idi id_offset;
            idi dist_start_index;
            idi dist_bound_index;
            idi v_start_index;
            idi v_bound_index;
            smalli dist;
            for (idi r_id = 0; r_id < roots_size; ++r_id) {
                if (used_bp_roots[r_id + roots_start]) {
                    continue;
                }
                IndexType &Lr = L[r_id + roots_start];
                b_i_bound = Lr.batches.size();
                _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
                _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
                _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
                for (inti b_i = 0; b_i < b_i_bound; ++b_i) {
                    id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
                    dist_start_index = Lr.batches[b_i].start_index;
                    dist_bound_index = dist_start_index + Lr.batches[b_i].size;
                    // Traverse dist_matrix
                    for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                        v_start_index = Lr.distances[dist_i].start_index;
                        v_bound_index = v_start_index + Lr.distances[dist_i].size;
                        dist = Lr.distances[dist_i].dist;
                        for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                            dist_matrix[r_id][Lr.vertices[v_i] + id_offset] = dist;
                        }
                    }
                }
            }
        }
//		inti b_i_bound;
//		idi id_offset;
//		idi dist_start_index;
//		idi dist_bound_index;
//		idi v_start_index;
//		idi v_bound_index;
//		smalli dist;
//		for (idi r_id = 0; r_id < roots_size; ++r_id) {
//			if (used_bp_roots[r_id + roots_start]) {
//				continue;
//			}
//			IndexType &Lr = L[r_id + roots_start];
//			b_i_bound = Lr.batches.size();
//			_mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
//			_mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
//			_mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
//			for (inti b_i = 0; b_i < b_i_bound; ++b_i) {
//				id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
//				dist_start_index = Lr.batches[b_i].start_index;
//				dist_bound_index = dist_start_index + Lr.batches[b_i].size;
//				// Traverse dist_matrix
//				for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
//					v_start_index = Lr.distances[dist_i].start_index;
//					v_bound_index = v_start_index + Lr.distances[dist_i].size;
//					dist = Lr.distances[dist_i].dist;
//					for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
//						dist_matrix[r_id][Lr.vertices[v_i] + id_offset] = dist;
//					}
//				}
//			}
//		}
    }
//	init_dist_matrix_time += WallTimer::get_time_mark();
}

// Function that pushes v_head's labels to v_head's every neighbor
template<inti BATCH_SIZE>
inline void ParaVertexCentricPLL<BATCH_SIZE>::push_labels(
        idi v_head,
        idi roots_start,
        const Graph &G,
        const vector<IndexType> &L,
        vector<ShortIndex> &short_index,
//				vector<idi> &candidate_queue,
//				idi &end_candidate_queue,
        vector<idi> &tmp_candidate_queue,
        idi &size_tmp_candidate_queue,
        const idi offset_tmp_queue,
//				idi &offset_tmp_queue,
//				vector<bool> &got_candidates,
        vector<uint8_t> &got_candidates,
//				vector<idi> &once_candidated_queue,
//				idi &end_once_candidated_queue,
        vector<idi> &tmp_once_candidated_queue,
        idi &size_tmp_once_candidated_queue,
//				vector<bool> &once_candidated,
        vector<uint8_t> &once_candidated,
        const vector<uint8_t> &used_bp_roots,
        smalli iter)
{
    const IndexType &Lv = L[v_head];
    // These 2 index are used for traversing v_head's last inserted labels
    idi l_i_start = Lv.distances.rbegin()->start_index;
    idi l_i_bound = l_i_start + Lv.distances.rbegin()->size;
    // Traverse v_head's every neighbor v_tail
    idi e_i_start = G.vertices[v_head];
    idi e_i_bound = e_i_start + G.out_degrees[v_head];
    for (idi e_i = e_i_start; e_i < e_i_bound; ++e_i) {
        idi v_tail = G.out_edges[e_i];

        if (used_bp_roots[v_head]) {
            continue;
        }

        if (v_tail < roots_start) { // v_tail has higher rank than any roots, then no roots can push new labels to it.
            return;
        }
//		if (v_tail <= Lv.vertices[l_i_start] + roots_start) { // v_tail has higher rank than any v_head's labels
//			return;
//		} // This condition cannot be used anymore since v_head's last inserted labels are not ordered from higher rank to lower rank now, because v_head's candidate set is a queue now rather than a bitmap. For a queue, its order of candidates are not ordered by ranks.
        const IndexType &L_tail = L[v_tail];
        _mm_prefetch(&L_tail.bp_dist[0], _MM_HINT_T0);
        _mm_prefetch(&L_tail.bp_sets[0][0], _MM_HINT_T0);
        // Traverse v_head's last inserted labels
        for (idi l_i = l_i_start; l_i < l_i_bound; ++l_i) {
            inti label_root_id = Lv.vertices[l_i];
            idi label_real_id = label_root_id + roots_start;
            if (v_tail <= label_real_id) {
                // v_tail has higher rank than all remaining labels
                // For candidates_que, this is not true any more!
//				break;
                continue;
            }
            ShortIndex &SI_v_tail = short_index[v_tail];
//            if (SI_v_tail.indicator[label_root_id]) {
//                // The label is already selected before
//                continue;
//            }
//            // Record label_root_id as once selected by v_tail
//            SI_v_tail.indicator.set(label_root_id);
            {// Deal with race condition
                if (!PADO::CAS(SI_v_tail.indicator.data() + label_root_id, static_cast<uint8_t>(0), static_cast<uint8_t>(1))) {
                    // The label is already selected before
                    continue;
                }
            }

//            {//test
//                // Check v_tail's indicator
//                if (!SI_v_tail.indicator[label_root_id]) {
//                    printf("L%u: B%u short_index[%u].indicator[%u]: %u which should be 1.\n", __LINE__,
//                           roots_start / BATCH_SIZE, v_tail, label_real_id, (idi) SI_v_tail.indicator[label_root_id]);
//                }
//            }
            // Add into once_candidated_queue
            if (!once_candidated[v_tail]) {
                // If v_tail is not in the once_candidated_queue yet, add it in
                if (CAS(&once_candidated[v_tail], (uint8_t) 0, (uint8_t) 1)) {
                    tmp_once_candidated_queue[offset_tmp_queue + size_tmp_once_candidated_queue++] = v_tail;
                }
            } // CHANGED!

            // Bit Parallel Checking: if label_real_id to v_tail has shorter distance already
//			++total_check_count;
            const IndexType &L_label = L[label_real_id];
            bool no_need_add = false;
            _mm_prefetch(&L_label.bp_dist[0], _MM_HINT_T0);
            _mm_prefetch(&L_label.bp_sets[0][0], _MM_HINT_T0);
            for (inti i = 0; i < BITPARALLEL_SIZE; ++i) {
                inti td = L_label.bp_dist[i] + L_tail.bp_dist[i];
                if (td - 2 <= iter) {
                    td +=
                            (L_label.bp_sets[i][0] & L_tail.bp_sets[i][0]) ? -2 :
                            ((L_label.bp_sets[i][0] & L_tail.bp_sets[i][1]) |
                             (L_label.bp_sets[i][1] & L_tail.bp_sets[i][0]))
                            ? -1 : 0;
                    if (td <= iter) {
                        no_need_add = true;
//		        	++bp_hit_count;
                        break;
                    }
                }
            }
            if (no_need_add) {
                continue;
            }

            // Record vertex label_root_id as v_tail's candidates label
//			SI_v_tail.candidates.set(label_root_id);
//			if (!SI_v_tail.is_candidate[label_root_id]) {
//				SI_v_tail.is_candidate[label_root_id] = true;
//				SI_v_tail.candidates_que[SI_v_tail.end_candidates_que++] = label_root_id;
//			}
            if (!SI_v_tail.is_candidate[label_root_id]) {
                if (CAS(&SI_v_tail.is_candidate[label_root_id], (uint8_t) 0, (uint8_t) 1)) {
                    TS_enqueue(SI_v_tail.candidates_que, SI_v_tail.end_candidates_que, label_root_id);
//                    {
//                        SI_v_tail.indicator.set(label_root_id);
//                    }
//                    {//test
//                        // Check v_tail's indicator
//                        if (!SI_v_tail.indicator[label_root_id]) {
//                            printf("L%u: T%u: B%u: l_i: %u iter: %u "
//                                   "short_index[%u].indicator[%u]: %u which should be 1.\n",
//                                   __LINE__, omp_get_thread_num(), roots_start / BATCH_SIZE, l_i, iter,
//                                   v_tail, label_real_id, (idi) SI_v_tail.indicator[label_root_id]);
//                        }
//                    }
                }
            }

            // Add into candidate_queue
            if (!got_candidates[v_tail]) {
                // If v_tail is not in candidate_queue, add it in (prevent duplicate)
                if (CAS(&got_candidates[v_tail], (uint8_t) 0, (uint8_t) 1)) {
                    tmp_candidate_queue[offset_tmp_queue + size_tmp_candidate_queue++] = v_tail;
                }
            }
        }
    }

//	printf("v_head: %u, size_tmp_candidate_queue: %u\n", v_head, size_tmp_candidate_queue);//test
}

// Function for distance query;
// traverse vertex v_id's labels;
// return the distance between v_id and cand_root_id based on existing labels.
// return false if shorter distance exists already, return true if the cand_root_id can be added into v_id's label.
template<inti BATCH_SIZE>
inline bool ParaVertexCentricPLL<BATCH_SIZE>::distance_query(
        idi cand_root_id,
        idi v_id,
        idi roots_start,
        const vector<IndexType> &L,
        const vector<vector<smalli> > &dist_matrix,
        smalli iter)
{
//	++total_check_count;
//	distance_query_time -= WallTimer::get_time_mark();

    idi cand_real_id = cand_root_id + roots_start;
    const IndexType &Lv = L[v_id];

    // Traverse v_id's all existing labels
    inti b_i_bound = Lv.batches.size();
    _mm_prefetch(&Lv.batches[0], _MM_HINT_T0);
    _mm_prefetch(&Lv.distances[0], _MM_HINT_T0);
    _mm_prefetch(&Lv.vertices[0], _MM_HINT_T0);
    _mm_prefetch(&dist_matrix[cand_root_id][0], _MM_HINT_T0);
    for (inti b_i = 0; b_i < b_i_bound; ++b_i) {
        idi id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
        idi dist_start_index = Lv.batches[b_i].start_index;
        idi dist_bound_index = dist_start_index + Lv.batches[b_i].size;
        // Traverse dist_matrix
        for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
            inti dist = Lv.distances[dist_i].dist;
            if (dist >= iter) { // In a batch, the labels' distances are increasingly ordered.
                // If the half path distance is already greater than their targeted distance, jump to next batch
                break;
            }
            idi v_start_index = Lv.distances[dist_i].start_index;
            idi v_bound_index = v_start_index + Lv.distances[dist_i].size;
//			_mm_prefetch(&dist_matrix[cand_root_id][0], _MM_HINT_T0);
            for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                idi v = Lv.vertices[v_i] + id_offset; // v is a label hub of v_id
                {//test
                    if (v == cand_real_id) {
                        printf("T%u: "
                               "In distance_query: v_id %u had got (%u, %u) in B%u, but is being pushed (%u, %u) in B%u again.\n",
                               omp_get_thread_num(),
                               v_id,
                               v, dist, Lv.batches[b_i].batch_id,
                               cand_real_id, iter, roots_start / BATCH_SIZE);
//                        printf("tmp_short_index[%u].indicator[%u]: %u "
//                               "now_short_index[%u].indicator[%u]: %u\n",
//                                v_id, cand_real_id,
//                               (idi) tmp_short_index[v_id].indicator[cand_root_id],
//                               v_id, cand_real_id,
//                               (idi) now_short_index[v_id].indicator[cand_root_id]);
                    }
                }
                if (v >= cand_real_id) {
                    // Vertex cand_real_id cannot have labels whose ranks are lower than it,
                    // in which case dist_matrix[cand_root_id][v] does not exit.
                    continue;
                }
                inti d_tmp = dist + dist_matrix[cand_root_id][v];
                {//test
                    if (v == cand_real_id) {
                        printf("d_tmp: %u dist: %u dist_matrix[%u][%u]: %u\n",
                               d_tmp,
                               dist,
                               cand_real_id, v, dist_matrix[cand_root_id][v]);
                    }
                }
                if (d_tmp <= iter) {
//					distance_query_time += WallTimer::get_time_mark();
//					++normal_hit_count;
                    return false;
                }
            }
        }
    }
//	distance_query_time += WallTimer::get_time_mark();
    return true;
}

// Function inserts candidate cand_root_id into vertex v_id's labels;
// update the distance buffer dist_matrix;
// but it only update the v_id's labels' vertices array;
template<inti BATCH_SIZE>
inline void ParaVertexCentricPLL<BATCH_SIZE>::insert_label_only(
        idi cand_root_id,
        idi v_id,
        idi roots_start,
        inti roots_size,
        vector<IndexType> &L,
        vector<vector<smalli> > &dist_matrix,
        smalli iter)
{
    L[v_id].vertices.push_back(cand_root_id);
    // Update the distance buffer if necessary
    idi v_root_id = v_id - roots_start;
    if (v_id >= roots_start && v_root_id < roots_size) {
        dist_matrix[v_root_id][cand_root_id + roots_start] = iter;
    }
}

// Function updates those index arrays in v_id's label only if v_id has been inserted new labels
template<inti BATCH_SIZE>
inline void ParaVertexCentricPLL<BATCH_SIZE>::update_label_indices(
        idi v_id,
        idi inserted_count,
        vector<IndexType> &L,
        vector<ShortIndex> &short_index,
        idi b_id,
        smalli iter)
{
    IndexType &Lv = L[v_id];
    // indicator[BATCH_SIZE + 1] is true, means v got some labels already in this batch
    if (short_index[v_id].indicator[BATCH_SIZE]) {
        // Increase the batches' last element's size because a new distance element need to be added
        ++(Lv.batches.rbegin()->size);
    } else {
//        short_index[v_id].indicator.set(BATCH_SIZE);
        short_index[v_id].indicator[BATCH_SIZE] = 1;
        // Insert a new Batch with batch_id, start_index, and size because a new distance element need to be added
        Lv.batches.push_back(IndexType::Batch(
                b_id,
                Lv.distances.size(),
                1));
    }
    // Insert a new distance element with start_index, size, and dist
    Lv.distances.push_back(IndexType::DistanceIndexType(
            Lv.vertices.size() - inserted_count,
            inserted_count,
            iter));
}

// Function to reset dist_matrix the distance buffer to INF
// Traverse every root's labels to reset its distance buffer elements to INF.
// In this way to reduce the cost of initialization of the next batch.
template<inti BATCH_SIZE>
inline void ParaVertexCentricPLL<BATCH_SIZE>::reset_at_end(
        idi roots_start,
        inti roots_size,
        vector<IndexType> &L,
        vector<vector<smalli> > &dist_matrix)
{
    if (roots_size >= THRESHOLD_PARALLEL) {
#pragma omp parallel for
        for (idi r_id = 0; r_id < roots_size; ++r_id) {
            IndexType &Lr = L[r_id + roots_start];
            inti b_i_bound = Lr.batches.size();
            _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
            _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
            _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
            for (inti b_i = 0; b_i < b_i_bound; ++b_i) {
                idi id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
                idi dist_start_index = Lr.batches[b_i].start_index;
                idi dist_bound_index = dist_start_index + Lr.batches[b_i].size;
                // Traverse dist_matrix
                for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                    idi v_start_index = Lr.distances[dist_i].start_index;
                    idi v_bound_index = v_start_index + Lr.distances[dist_i].size;
                    for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                        dist_matrix[r_id][Lr.vertices[v_i] + id_offset] = SMALLI_MAX;
                    }
                }
            }
        }
    } else {
        inti b_i_bound;
        idi id_offset;
        idi dist_start_index;
        idi dist_bound_index;
        idi v_start_index;
        idi v_bound_index;
        for (idi r_id = 0; r_id < roots_size; ++r_id) {
            IndexType &Lr = L[r_id + roots_start];
            b_i_bound = Lr.batches.size();
            _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
            _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
            _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
            for (inti b_i = 0; b_i < b_i_bound; ++b_i) {
                id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
                dist_start_index = Lr.batches[b_i].start_index;
                dist_bound_index = dist_start_index + Lr.batches[b_i].size;
                // Traverse dist_matrix
                for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                    v_start_index = Lr.distances[dist_i].start_index;
                    v_bound_index = v_start_index + Lr.distances[dist_i].size;
                    for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                        dist_matrix[r_id][Lr.vertices[v_i] + id_offset] = SMALLI_MAX;
                    }
                }
            }
        }
    }
//	inti b_i_bound;
//	idi id_offset;
//	idi dist_start_index;
//	idi dist_bound_index;
//	idi v_start_index;
//	idi v_bound_index;
//	for (idi r_id = 0; r_id < roots_size; ++r_id) {
//		IndexType &Lr = L[r_id + roots_start];
//		b_i_bound = Lr.batches.size();
//		_mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
//		_mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
//		_mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
//		for (inti b_i = 0; b_i < b_i_bound; ++b_i) {
//			id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
//			dist_start_index = Lr.batches[b_i].start_index;
//			dist_bound_index = dist_start_index + Lr.batches[b_i].size;
//			// Traverse dist_matrix
//			for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
//				v_start_index = Lr.distances[dist_i].start_index;
//				v_bound_index = v_start_index + Lr.distances[dist_i].size;
//				for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
//					dist_matrix[r_id][Lr.vertices[v_i] + id_offset] = SMALLI_MAX;
//				}
//			}
//		}
//	}
}

template<inti BATCH_SIZE>
inline void ParaVertexCentricPLL<BATCH_SIZE>::batch_process(
        const Graph &G,
        idi b_id,
        idi roots_start, // start id of roots
        inti roots_size, // how many roots in the batch
        vector<IndexType> &L,
        const vector<uint8_t> &used_bp_roots,
        vector<idi> &active_queue,
        idi &end_active_queue,
        vector<idi> &candidate_queue,
        idi &end_candidate_queue,
        vector<ShortIndex> &short_index,
        vector<vector<smalli> > &dist_matrix,
        vector<uint8_t> &got_candidates,
        vector<uint8_t> &is_active,
        vector<idi> &once_candidated_queue,
        idi &end_once_candidated_queue,
        vector<uint8_t> &once_candidated)
//inline void ParaVertexCentricPLL::batch_process(
//						const Graph &G,
//						idi b_id,
//						idi roots_start, // start id of roots
//						inti roots_size, // how many roots in the batch
//						vector<IndexType> &L,
//						const vector<bool> &used_bp_roots)
{

//	initializing_time -= WallTimer::get_time_mark();
//	static const idi num_v = G.get_num_v();
//	static vector<idi> active_queue(num_v);
//	static idi end_active_queue = 0;
//	static vector<idi> candidate_queue(num_v);
//	static idi end_candidate_queue = 0;
//	static vector<ShortIndex> short_index(num_v);
//	static vector< vector<smalli> > dist_matrix(roots_size, vector<smalli>(num_v, SMALLI_MAX));
//	static uint8_t *got_candidates = (uint8_t *) calloc(num_v, sizeof(uint8_t)); // need raw integer type to do CAS.
//	static uint8_t *is_active = (uint8_t *) calloc(num_v, sizeof(uint8_t));
//	static vector<idi> once_candidated_queue(num_v); // The vertex who got some candidates in this batch is in the once_candidated_queue.
//	static idi end_once_candidated_queue = 0;
//	static uint8_t *once_candidated = (uint8_t *) calloc(num_v, sizeof(uint8_t)); // need raw integer type to do CAS.

    // At the beginning of a batch, initialize the labels L and distance buffer dist_matrix;
//	printf("initializing...\n");//test
    initialize(
            short_index,
            dist_matrix,
            active_queue,
            end_active_queue,
            once_candidated_queue,
            end_once_candidated_queue,
            once_candidated,
            b_id,
            roots_start,
            roots_size,
            L,
            used_bp_roots);

    smalli iter = 0; // The iterator, also the distance for current iteration
//	initializing_time += WallTimer::get_time_mark();


    {//test
//        now_short_index.assign(short_index.begin(), short_index.end());
    }
    while (0 != end_active_queue) {
//		candidating_time -= WallTimer::get_time_mark();
        ++iter;
        {//test
//            tmp_short_index.swap(now_short_index);
        }

        // Pushing
//		printf("pushing...\n");//test
        {
            // Prepare for parallel processing the active_queue and adding to candidate_queue.
            // Every vertex's offset location in tmp_candidate_queue
            // It's used for every thread to write into tmp_candidate_queue and tmp_once_candidated_queue
            vector<idi> offsets_tmp_queue(end_active_queue);
#pragma omp parallel for
            for (idi i_queue = 0; i_queue < end_active_queue; ++i_queue) {
                // Traverse all active vertices, get their out degrees.
                offsets_tmp_queue[i_queue] = G.out_degrees[active_queue[i_queue]];
            }
            idi num_neighbors = prefix_sum_for_offsets(offsets_tmp_queue);
            // every thread writes to tmp_candidate_queue at its offset location
            vector<idi> tmp_candidate_queue(num_neighbors);
            // A vector to store the true number of pushed neighbors of every active vertex.
            vector<idi> sizes_tmp_candidate_queue(end_active_queue, 0);
            // similarly, every thread writes to tmp_once_candidated_queue at its offset location
            vector<idi> tmp_once_candidated_queue(num_neighbors);
            // And store the true number of new added once-candidated vertices.
            vector<idi> sizes_tmp_once_candidated_queue(end_active_queue, 0);

            // Traverse active vertices to push their labels as candidates
// schedule dynamic is slower
#pragma omp parallel for
//TODO: turn on OpenMP
            for (idi i_queue = 0; i_queue < end_active_queue; ++i_queue) {
                idi v_head = active_queue[i_queue];
                is_active[v_head] = 0; // reset is_active

                push_labels(
                        v_head,
                        roots_start,
                        G,
                        L,
                        short_index,
                        //					candidate_queue,
                        //					end_candidate_queue,
                        tmp_candidate_queue,
                        sizes_tmp_candidate_queue[i_queue],
                        offsets_tmp_queue[i_queue],
                        got_candidates,
                        //					once_candidated_queue,
                        //					end_once_candidated_queue,
                        tmp_once_candidated_queue,
                        sizes_tmp_once_candidated_queue[i_queue],
                        once_candidated,
                        used_bp_roots,
                        iter);
            }
            {//test
//                now_short_index.assign(short_index.begin(), short_index.end());
            }

            // According to sizes_tmp_candidate_queue, get the offset for inserting to the real queue
            idi total_new = prefix_sum_for_offsets(sizes_tmp_candidate_queue);
            // Collect all candidate vertices from tmp_candidate_queue into candidate_queue.
            collect_into_queue(
                    tmp_candidate_queue,
                    offsets_tmp_queue, // the locations in tmp_queue for writing from tmp_queue
                    sizes_tmp_candidate_queue, // the locations in queue for writing into queue.
                    total_new, // total number of elements which need to be added from tmp_queue to queue
                    candidate_queue,
                    end_candidate_queue);
            // Get the offset for inserting to the real queue.
            total_new = prefix_sum_for_offsets(sizes_tmp_once_candidated_queue);
            // Collect all once-candidated vertices from tmp_once_candidated_queue into once_candidated_queue
            collect_into_queue(
                    tmp_once_candidated_queue,
                    offsets_tmp_queue,
                    sizes_tmp_once_candidated_queue,
                    total_new,
                    once_candidated_queue,
                    end_once_candidated_queue);
            //		printf("end_candidate_queue: %u\n", end_candidate_queue); fflush(stdout);//test
            end_active_queue = 0; // Set the active_queue empty
        }

//		candidating_time += WallTimer::get_time_mark();
        if (end_candidate_queue == 0) {
            break;
        }
//		adding_time -= WallTimer::get_time_mark();
        // Adding
//		printf("adding...\n");//test
        {
//////////////////////////////////////////////////////////////////////////////////
// OpenMP Version
            // Prepare for parallel processing the candidate_queue and adding to active_queue.
            // Every vertex's offset location in tmp_active_queue is i_queue * roots_size
            // It's used for every thread to write into tmp_candidate_queue and tmp_once_candidated_queue
            vector<idi> offsets_tmp_queue(end_candidate_queue);
#pragma omp parallel for
            for (idi i_queue = 0; i_queue < end_candidate_queue; ++i_queue) {
                // Traverse all active vertices, get their out degrees.
                // A ridiculous bug here. The v_id will, if any, only add itself to the active queue.
                //offsets_tmp_queue[i_queue] = i_queue * roots_size;
                offsets_tmp_queue[i_queue] = i_queue;
            }
            // every thread writes to tmp_candidate_queue at its offset location
            vector<idi> tmp_active_queue(end_candidate_queue);
            // A vector to store the true number of pushed neighbors of every active vertex.
            vector<idi> sizes_tmp_active_queue(end_candidate_queue, 0);

            // Traverse vertices in the candidate_queue to insert labels
// Here schedule dynamic will be slower
//#ifdef PROFILE
//			cache_miss.measure_start();
//#endif
#pragma omp parallel for schedule(dynamic)
            for (idi i_queue = 0; i_queue < end_candidate_queue; ++i_queue) {
//#ifdef PROFILE
//				inti tid = omp_get_thread_num();
//				thds_adding_time[tid] -= WallTimer::get_time_mark();
//#endif

                idi v_id = candidate_queue[i_queue];
                inti inserted_count = 0; //recording number of v_id's truly inserted candidates
                got_candidates[v_id] = 0; // reset got_candidates
                inti bound_cand_i = short_index[v_id].end_candidates_que;
                for (inti cand_i = 0; cand_i < bound_cand_i; ++cand_i) {
                    inti cand_root_id = short_index[v_id].candidates_que[cand_i];
//                    {//test
//                        // Check v_id's indicator
//                        if (!short_index[v_id].indicator[cand_root_id]) {
//                            printf("L%u: T%u: B%u: iter: %u "
//                                   "short_index[%u].indicator[%u]: %u which should be 1.\n",
//                                   __LINE__, omp_get_thread_num(), b_id, iter,
//                                   v_id, cand_root_id + roots_start, (idi) short_index[v_id].indicator[cand_root_id]);
//                        }
//                    }
                    short_index[v_id].is_candidate[cand_root_id] = 0; // Reset is_candidate
                    // Only insert cand_root_id into v_id's label if its distance to v_id is shorter than existing distance
                    if (distance_query(
                            cand_root_id,
                            v_id,
                            roots_start,
                            L,
                            dist_matrix,
                            iter)) {
                        if (!is_active[v_id]) {
                            is_active[v_id] = 1;
                            tmp_active_queue[offsets_tmp_queue[i_queue] + sizes_tmp_active_queue[i_queue]++] = v_id;
                        }
//						if (!be_active) {
//							be_active = true;
//						}
                        //					if (!is_active[v_id]) {
                        //						is_active[v_id] = true;
                        //						active_queue[end_active_queue++] = v_id;
                        //					}
                        ++inserted_count;
                        // The candidate cand_root_id needs to be added into v_id's label
                        insert_label_only(
                                cand_root_id,
                                v_id,
                                roots_start,
                                roots_size,
                                L,
                                dist_matrix,
                                iter);
                        {//test
////                            // Check v_id's indicator
//                            if (!short_index[v_id].indicator[cand_root_id]) {
//                                printf("L:%u T%u: B%u iter: %u "
//                                       "short_index[%u].indicator[%u]: %u which should be 1.\n",
//                                       __LINE__, omp_get_thread_num(), b_id, iter,
//                                       v_id, cand_root_id + roots_start,
//                                       (idi) short_index[v_id].indicator[cand_root_id]);
//                            }

                            // Traverse all v_id's labels and check if cand_root_id is there
                            const IndexType &Lv = L[v_id];
                            inti b_i_bound = Lv.batches.size();
                            _mm_prefetch(&Lv.batches[0], _MM_HINT_T0);
                            _mm_prefetch(&Lv.distances[0], _MM_HINT_T0);
                            _mm_prefetch(&Lv.vertices[0], _MM_HINT_T0);
                            _mm_prefetch(&dist_matrix[cand_root_id][0], _MM_HINT_T0);
                            for (inti b_i = 0; b_i < b_i_bound; ++b_i) {
                                idi id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
                                idi dist_start_index = Lv.batches[b_i].start_index;
                                idi dist_bound_index = dist_start_index + Lv.batches[b_i].size;
                                // Traverse dist_matrix
                                for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                                    inti dist = Lv.distances[dist_i].dist;
                                    idi v_start_index = Lv.distances[dist_i].start_index;
                                    idi v_bound_index = v_start_index + Lv.distances[dist_i].size;
                                    for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                                        idi v = Lv.vertices[v_i] + id_offset; // v is a label hub of v_id
                                        if (v == cand_root_id + roots_start) {
                                            printf("! T%u: "
                                                   "v_id %u already got (%u, %u), rather than (%u, %u)\n",
                                                   omp_get_thread_num(),
                                                   v_id, v, dist, cand_root_id + roots_start, iter);
//                                            exit(-1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                short_index[v_id].end_candidates_que = 0;
//				if (be_active) {
//					if (CAS(&is_active[v_id], (uint8_t) 0, (uint8_t) 1)) {
//						tmp_active_queue[offsets_tmp_queue[i_queue] + sizes_tmp_active_queue[i_queue]++] = v_id;
//					}
//				}
                if (0 != inserted_count) {
                    // Update other arrays in L[v_id] if new labels were inserted in this iteration
                    update_label_indices(
                            v_id,
                            inserted_count,
                            L,
                            short_index,
                            b_id,
                            iter);
                }
            }

            // According to sizes_tmp_active_queue, get the offset for inserting to the real queue
            idi total_new = prefix_sum_for_offsets(sizes_tmp_active_queue);
            // Collect all candidate vertices from tmp_candidate_queue into candidate_queue.
            collect_into_queue(
                    tmp_active_queue,
                    offsets_tmp_queue, // the locations in tmp_queue for writing from tmp_queue
                    sizes_tmp_active_queue, // the locations in queue for writing into queue.
                    total_new, // total number of elements which need to be added from tmp_queue to queue
                    active_queue,
                    end_active_queue);
            end_candidate_queue = 0; // Set the candidate_queue empty
//////////////////////////////////////////////////////////////////////////////////
////// Sequential version
//            for (idi i_queue = 0; i_queue < end_candidate_queue; ++i_queue) {
//                idi v_id = candidate_queue[i_queue];
//                inti inserted_count = 0; //recording number of v_id's truly inserted candidates
//                got_candidates[v_id] = false; // reset got_candidates
//                // Traverse v_id's all candidates
//                inti bound_cand_i = short_index[v_id].end_candidates_que;
//                for (inti cand_i = 0; cand_i < bound_cand_i; ++cand_i) {
//                    inti cand_root_id = short_index[v_id].candidates_que[cand_i];
//                    short_index[v_id].is_candidate[cand_root_id] = false;
//                    // Only insert cand_root_id into v_id's label if its distance to v_id is shorter than existing distance
//                    if ( distance_query(
//                            cand_root_id,
//                            v_id,
//                            roots_start,
//                            L,
//                            dist_matrix,
//                            iter) ) {
//                        if (!is_active[v_id]) {
//                            is_active[v_id] = true;
//                            active_queue[end_active_queue++] = v_id;
//                        }
//                        ++inserted_count;
//                        // The candidate cand_root_id needs to be added into v_id's label
//                        insert_label_only(
//                                cand_root_id,
//                                v_id,
//                                roots_start,
//                                roots_size,
//                                L,
//                                dist_matrix,
//                                iter);
//                    }
//                }
//                short_index[v_id].end_candidates_que = 0;
////			}
//                if (0 != inserted_count) {
//                    // Update other arrays in L[v_id] if new labels were inserted in this iteration
//                    update_label_indices(
//                            v_id,
//                            inserted_count,
//                            L,
//                            short_index,
//                            b_id,
//                            iter);
//                }
//            }
//            end_candidate_queue = 0; // Set the candidate_queue empty
//////////////////////////////////////////////////////////////////////////////////////
        }
//		adding_time += WallTimer::get_time_mark();
    }


    // Reset the dist_matrix
//	initializing_time -= WallTimer::get_time_mark();
//	init_dist_matrix_time -= WallTimer::get_time_mark();
    reset_at_end(
            roots_start,
            roots_size,
            L,
            dist_matrix);
//	init_dist_matrix_time += WallTimer::get_time_mark();
//	initializing_time += WallTimer::get_time_mark();


//	double total_time = time_can + time_add;
//	printf("Candidating time: %f (%f%%)\n", time_can, time_can / total_time * 100);
//	printf("Adding time: %f (%f%%)\n", time_add, time_add / total_time * 100);
}


template<inti BATCH_SIZE>
void ParaVertexCentricPLL<BATCH_SIZE>::construct(const Graph &G)
{
//	initializing_time -= WallTimer::get_time_mark();

    idi num_v = G.get_num_v();
    num_v_ = num_v;
    L.resize(num_v);
    idi remainer = num_v % BATCH_SIZE;
    idi b_i_bound = num_v / BATCH_SIZE;
//	uint8_t *used_bp_roots = (uint8_t *) calloc(num_v, sizeof(uint8_t));
    vector<uint8_t> used_bp_roots(num_v, 0);

    vector<idi> active_queue(num_v);
    idi end_active_queue = 0;
    vector<idi> candidate_queue(num_v);
    idi end_candidate_queue = 0;
    vector<ShortIndex> short_index(num_v);
//    vector<ShortIndex> short_index; short_index.resize(num_v);
    vector<vector<smalli> > dist_matrix(BATCH_SIZE, vector<smalli>(num_v, SMALLI_MAX));
//	uint8_t *got_candidates = (uint8_t *) calloc(num_v, sizeof(uint8_t)); // need raw integer type to do CAS.
//	uint8_t *is_active = (uint8_t *) calloc(num_v, sizeof(uint8_t)); // need raw integer type to do CAS.
    vector<uint8_t> got_candidates(num_v, 0);
    vector<uint8_t> is_active(num_v, 0);
    vector<idi> once_candidated_queue(
            num_v); // The vertex who got some candidates in this batch is in the once_candidated_queue.
    idi end_once_candidated_queue = 0;
//	uint8_t *once_candidated = (uint8_t *) calloc(num_v, sizeof(uint8_t)); // need raw integer type to do CAS.
    vector<uint8_t> once_candidated(num_v, 0);

//	initializing_time += WallTimer::get_time_mark();
    double time_labeling = -WallTimer::get_time_mark();

    //double bp_labeling_time = -WallTimer::get_time_mark();
//	printf("BP labeling...\n"); //test
    bit_parallel_labeling(
            G,
            L,
            used_bp_roots);
    //bp_labeling_time += WallTimer::get_time_mark();


    for (idi b_i = 0; b_i < b_i_bound; ++b_i) {
//		printf("b_i: %u\n", b_i);//test
        batch_process(
                G,
                b_i,
                b_i * BATCH_SIZE,
                BATCH_SIZE,
                L,
                used_bp_roots,
                active_queue,
                end_active_queue,
                candidate_queue,
                end_candidate_queue,
                short_index,
                dist_matrix,
                got_candidates,
                is_active,
                once_candidated_queue,
                end_once_candidated_queue,
                once_candidated);
//		batch_process(
//				G,
//				b_i,
//				b_i * BATCH_SIZE,
//				BATCH_SIZE,
//				L,
//				used_bp_roots);
    }
    if (remainer != 0) {
//		printf("b_i: %u the last batch\n", b_i_bound);//test
        batch_process(
                G,
                b_i_bound,
                b_i_bound * BATCH_SIZE,
                remainer,
                L,
                used_bp_roots,
                active_queue,
                end_active_queue,
                candidate_queue,
                end_candidate_queue,
                short_index,
                dist_matrix,
                got_candidates,
                is_active,
                once_candidated_queue,
                end_once_candidated_queue,
                once_candidated);
//		batch_process(
//				G,
//				b_i_bound,
//				b_i_bound * BATCH_SIZE,
//				remainer,
//				L,
//				used_bp_roots);
    }
    time_labeling += WallTimer::get_time_mark();

//	free(got_candidates);
//	free(is_active);
//	free(once_candidated);
//	free(used_bp_roots);

    // Test
    printf("Threads: %u Batch_size: %u\n", NUM_THREADS, BATCH_SIZE);
    //printf("BP_labeling: %.2f %.2f%%\n", bp_labeling_time, bp_labeling_time / time_labeling * 100);
    printf("BP_Roots_Size: %u\n", BITPARALLEL_SIZE);
//	printf("Initializing: %.2f %.2f%%\n", initializing_time, initializing_time / time_labeling * 100);
//		printf("\tinit_start_reset_time: %f (%f%%)\n", init_start_reset_time, init_start_reset_time / initializing_time * 100);
//		printf("\tinit_index_time: %f (%f%%)\n", init_index_time, init_index_time / initializing_time * 100);
//			printf("\t\tinit_indicators_time: %f (%f%%)\n", init_indicators_time, init_indicators_time / init_index_time * 100);
//		printf("\tinit_dist_matrix_time: %f (%f%%)\n", init_dist_matrix_time, init_dist_matrix_time / initializing_time * 100);
//	printf("Candidating: %.2f %.2f%%\n", candidating_time, candidating_time / time_labeling * 100);
//	printf("Adding: %.2f %.2f%%\n", adding_time, adding_time / time_labeling * 100);
//		printf("\tdistance_query_time: %f (%f%%)\n", distance_query_time, distance_query_time / adding_time * 100);
//		printf("\ttotal_check_count: %llu\n", total_check_count);
//		printf("\tbp_hit_count (to total_check): %llu (%f%%)\n",
//						bp_hit_count,
//						bp_hit_count * 100.0 / total_check_count);
//		printf("\tnormal_hit_count (to total_check, to normal_check): %llu (%f%%, %f%%)\n",
//						normal_hit_count,
//						normal_hit_count * 100.0 / total_check_count,
//						normal_hit_count * 100.0 / (total_check_count - bp_hit_count));

#ifdef PROFILE
    uint64_t total_thds_adding_count = 0;
    double total_thds_adding_time = 0;
    for (inti tid = 0; tid < NUM_THREADS; ++tid) {
        total_thds_adding_count += thds_adding_count[tid];
        total_thds_adding_time += thds_adding_time[tid];
    }
    printf("Threads_adding_count:");
    for (inti tid = 0; tid < NUM_THREADS; ++tid) {
        printf(" %lu(%.2f%%)", thds_adding_count[tid], thds_adding_count[tid] * 100.0 / total_thds_adding_count);
    } puts("");
    printf("Threads_adding_time:");
    for (inti tid = 0; tid < NUM_THREADS; ++tid) {
        printf(" %f(%.2f%%)", thds_adding_time[tid], thds_adding_time[tid] * 100.0 / total_thds_adding_time);
    } puts("");
    //printf("Threads_adding_average_time:");
    //for (inti tid = 0; tid < NUM_THREADS; ++tid) {
    //	printf(" %f", thds_adding_time[tid] / thds_adding_count[tid]);
    //} puts("");

    cache_miss.print();
#endif

    printf("Total_labeling_time: %.2f seconds\n", time_labeling);
    // End test
}

// Function to get the prefix sum of elements in offsets
template<inti BATCH_SIZE>
inline idi ParaVertexCentricPLL<BATCH_SIZE>::prefix_sum_for_offsets(
        vector<idi> &offsets)
{
    idi size_offsets = offsets.size();
    if (1 == size_offsets) {
        idi tmp = offsets[0];
        offsets[0] = 0;
        return tmp;
    } else if (size_offsets < 2048) {
        idi offset_sum = 0;
        idi size = size_offsets;
        for (idi i = 0; i < size; ++i) {
            idi tmp = offsets[i];
            offsets[i] = offset_sum;
            offset_sum += tmp;
        }
        return offset_sum;
    } else {
        // Parallel Prefix Sum, based on Guy E. Blelloch's Prefix Sums and Their Applications
        idi last_element = offsets[size_offsets - 1];
        //	idi size = 1 << ((idi) log2(size_offsets - 1) + 1);
        idi size = 1 << ((idi) log2(size_offsets));
        //	vector<idi> nodes(size, 0);
        idi tmp_element = offsets[size - 1];
        //#pragma omp parallel for
        //	for (idi i = 0; i < size_offsets; ++i) {
        //		nodes[i] = offsets[i];
        //	}

        // Up-Sweep (Reduce) Phase
        idi log2size = log2(size);
        for (idi d = 0; d < log2size; ++d) {
            idi by = 1 << (d + 1);
#pragma omp parallel for
            for (idi k = 0; k < size; k += by) {
                offsets[k + (1 << (d + 1)) - 1] += offsets[k + (1 << d) - 1];
            }
        }

        // Down-Sweep Phase
        offsets[size - 1] = 0;
        for (idi d = log2(size) - 1; d != (idi) -1; --d) {
            idi by = 1 << (d + 1);
#pragma omp parallel for
            for (idi k = 0; k < size; k += by) {
                idi t = offsets[k + (1 << d) - 1];
                offsets[k + (1 << d) - 1] = offsets[k + (1 << (d + 1)) - 1];
                offsets[k + (1 << (d + 1)) - 1] += t;
            }
        }

        //#pragma omp parallel for
        //	for (idi i = 0; i < size_offsets; ++i) {
        //		offsets[i] = nodes[i];
        //	}
        if (size != size_offsets) {
            idi tmp_sum = offsets[size - 1] + tmp_element;
            for (idi i = size; i < size_offsets; ++i) {
                idi t = offsets[i];
                offsets[i] = tmp_sum;
                tmp_sum += t;
            }
        }

        return offsets[size_offsets - 1] + last_element;
    }
//	// Get the offset as the prefix sum of out degrees
//	idi offset_sum = 0;
//	idi size = offsets.size();
//	for (idi i = 0; i < size; ++i) {
//		idi tmp = offsets[i];
//		offsets[i] = offset_sum;
//		offset_sum += tmp;
//	}
//	return offset_sum;

//// Parallel Prefix Sum, based on Guy E. Blelloch's Prefix Sums and Their Applications
//	idi size_offsets = offsets.size();
//	idi last_element = offsets[size_offsets - 1];
////	idi size = 1 << ((idi) log2(size_offsets - 1) + 1);
//	idi size = 1 << ((idi) log2(size_offsets));
////	vector<idi> nodes(size, 0);
//	idi tmp_element = offsets[size - 1];
////#pragma omp parallel for
////	for (idi i = 0; i < size_offsets; ++i) {
////		nodes[i] = offsets[i];
////	}
//
//	// Up-Sweep (Reduce) Phase
//	idi log2size = log2(size);
//	for (idi d = 0; d < log2size; ++d) {
//		idi by = 1 << (d + 1);
//#pragma omp parallel for
//		for (idi k = 0; k < size; k += by) {
//			offsets[k + (1 << (d + 1)) - 1] += offsets[k + (1 << d) - 1];
//		}
//	}
//
//	// Down-Sweep Phase
//	offsets[size - 1] = 0;
//	for (idi d = log2(size) - 1; d != (idi) -1 ; --d) {
//		idi by = 1 << (d + 1);
//#pragma omp parallel for
//		for (idi k = 0; k < size; k += by) {
//			idi t = offsets[k + (1 << d) - 1];
//			offsets[k + (1 << d) - 1] = offsets[k + (1 << (d + 1)) - 1];
//			offsets[k + (1 << (d + 1)) - 1] += t;
//		}
//	}
//
////#pragma omp parallel for
////	for (idi i = 0; i < size_offsets; ++i) {
////		offsets[i] = nodes[i];
////	}
//	if (size != offsets.size()) {
//		idi tmp_sum = offsets[size - 1] + tmp_element;
//		idi i_bound = offsets.size();
//		for (idi i = size; i < i_bound; ++i) {
//			idi t = offsets[i];
//			offsets[i] = tmp_sum;
//			tmp_sum += t;
//		}
//	}
//
//	return offsets[size_offsets - 1] + last_element;
}

// Collect elements in the tmp_queue into the queue
template<inti BATCH_SIZE>
template<typename T>
inline void ParaVertexCentricPLL<BATCH_SIZE>::collect_into_queue(
//					vector<idi> &tmp_queue,
        vector<T> &tmp_queue,
        vector<idi> &offsets_tmp_queue, // the locations in tmp_queue for writing from tmp_queue
        vector<idi> &offsets_queue, // the locations in queue for writing into queue.
        idi num_elements, // total number of elements which need to be added from tmp_queue to queue
//					vector<idi> &queue,
        vector<T> &queue,
        idi &end_queue)
{
    if (0 == num_elements) {
        return;
    }
    idi i_bound = offsets_tmp_queue.size();
#pragma omp parallel for
    for (idi i = 0; i < i_bound; ++i) {
        idi i_q_start = end_queue + offsets_queue[i];
        idi i_q_bound;
        if (i_bound - 1 != i) {
            i_q_bound = end_queue + offsets_queue[i + 1];
        } else {
            i_q_bound = end_queue + num_elements;
        }
        if (i_q_start == i_q_bound) {
            // If the group has no elements to be added, then continue to the next group
            continue;
        }
        idi end_tmp = offsets_tmp_queue[i];
        for (idi i_q = i_q_start; i_q < i_q_bound; ++i_q) {
            queue[i_q] = tmp_queue[end_tmp++];
        }
    }
    end_queue += num_elements;
}

// Function: thread-save enqueue. The queue has enough size already. An index points the end of the queue.
template<inti BATCH_SIZE>
template<typename T, typename Int>
inline void ParaVertexCentricPLL<BATCH_SIZE>::TS_enqueue(
        vector<T> &queue,
        Int &end_queue,
        const T &e)
{
    volatile Int old_i = end_queue;
    volatile Int new_i = old_i + 1;
    while (!CAS(&end_queue, old_i, new_i)) {
        old_i = end_queue;
        new_i = old_i + 1;
    }
    queue[old_i] = e;
}

template<inti BATCH_SIZE>
void ParaVertexCentricPLL<BATCH_SIZE>::store_index_to_file(
        const char *filename,
        const vector<idi> &rank)
{
    // TODO: fout comment out
//	std::ofstream fout(filename);
//	if (!fout.is_open()) {
//		fprintf(stderr, "Error: cannot open file %s\n", filename);
//		exit(EXIT_FAILURE);
//	}

//	std::string txt_filename = std::string(filename) + ".txt";//test
//	std::ofstream txt_out(txt_filename.c_str());

    // Store into file the number of vertices and the number of bit-parallel roots.
    uint64_t labels_count = 0;
//	fout.write((char *) &num_v_, sizeof(num_v_));
//	fout.write((char *) &BITPARALLEL_SIZE, sizeof(BITPARALLEL_SIZE));
    for (idi v_id = 0; v_id < num_v_; ++v_id) {
        idi v_rank = rank[v_id];
        const IndexType &Lv = L[v_rank];
        idi size_labels = Lv.vertices.size();
        labels_count += size_labels;
//		// Store Bit-parallel Labels into file.
//		for (inti b_i = 0; b_i < BITPARALLEL_SIZE; ++b_i) {
//			weighti d = Lv.bp_dist[b_i];
//			uint64_t s0 = Lv.bp_sets[b_i][0];
//			uint64_t s1 = Lv.bp_sets[b_i][1];
//			fout.write((char *) &d, sizeof(d));
//			fout.write((char *) &s0, sizeof(s0));
//			fout.write((char *) &s1, sizeof(s1));
//		}

        vector<std::pair<idi, weighti> > ordered_labels;
        // Traverse v_id's all existing labels
        for (inti b_i = 0; b_i < Lv.batches.size(); ++b_i) {
            idi id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
            idi dist_start_index = Lv.batches[b_i].start_index;
            idi dist_bound_index = dist_start_index + Lv.batches[b_i].size;
            // Traverse dist_matrix
            for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                idi v_start_index = Lv.distances[dist_i].start_index;
                idi v_bound_index = v_start_index + Lv.distances[dist_i].size;
                weighti dist = Lv.distances[dist_i].dist;
                for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                    idi tail = Lv.vertices[v_i] + id_offset;
                    ordered_labels.push_back(std::make_pair(tail, dist));
                }
            }
        }
        // Sort
        sort(ordered_labels.begin(), ordered_labels.end());
//		// Store into file
//		fout.write((char *) &size_labels, sizeof(size_labels));

        for (idi l_i = 0; l_i < size_labels; ++l_i) {
            idi l = ordered_labels[l_i].first;
            weighti d = ordered_labels[l_i].second;
//			fout.write((char *) &l, sizeof(l));
//			fout.write((char *) &d, sizeof(d));

//            {//test
//                txt_out << v_id << " " << v_rank << ": " << l << " " << (idi) d << std::endl;
//            }
        }
    }

    printf("Label_size: %'lu mean: %f\n", labels_count, static_cast<double>(labels_count) / num_v_);
//	fout.close();
}

template<inti BATCH_SIZE>
void ParaVertexCentricPLL<BATCH_SIZE>::load_index_from_file(
        const char *filename)
{
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    idi num_v;
    // Load from file the number of vertices and the number of bit-parallel roots.
    fin.read((char *) &num_v, sizeof(num_v));
    fin.read((char *) &BITPARALLEL_SIZE, sizeof(BITPARALLEL_SIZE));
    num_v_ = num_v;
    Index.resize(num_v);
    uint64_t labels_count = 0;
    // Load labels for every vertex
    for (idi v_id = 0; v_id < num_v; ++v_id) {
        IndexOrdered &Iv = Index[v_id];
        // Load Bit-parallel Labels from file.
        for (inti b_i = 0; b_i < BITPARALLEL_SIZE; ++b_i) {
            fin.read((char *) &Iv.bp_dist[b_i], sizeof(Iv.bp_dist[b_i]));
            fin.read((char *) &Iv.bp_sets[b_i][0], sizeof(Iv.bp_sets[b_i][0]));
            fin.read((char *) &Iv.bp_sets[b_i][1], sizeof(Iv.bp_sets[b_i][1]));
        }

        // Normal Labels
        // Load Labels from file.
        idi size_labels;
        fin.read((char *) &size_labels, sizeof(size_labels));
        labels_count += size_labels;
        Iv.label_id.resize(size_labels + 1);
        Iv.label_dists.resize(size_labels + 1);
        for (idi l_i = 0; l_i < size_labels; ++l_i) {
            fin.read((char *) &Iv.label_id[l_i], sizeof(Iv.label_id[l_i]));
            fin.read((char *) &Iv.label_dists[l_i], sizeof(Iv.label_dists[l_i]));
        }
        Iv.label_id[size_labels] = num_v; // Sentinel
        Iv.label_dists[size_labels] = (weighti) -1; // Sentinel
    }
    printf("Label_size_loaded: %'lu mean: %f\n", labels_count, static_cast<double>(labels_count) / num_v);
    fin.close();
}


template<inti BATCH_SIZE>
void ParaVertexCentricPLL<BATCH_SIZE>::order_labels(
        const vector<idi> &rank2id,
        const vector<idi> &rank)
{
    idi num_v = rank.size();
    vector<vector<pair < idi, weighti> > > ordered_L(num_v);
    idi labels_count = 0;
    Index.resize(num_v);

    // Traverse the L, put them into Index (ordered labels)
    for (idi v_id = 0; v_id < num_v; ++v_id) {
        idi new_v = rank2id[v_id];
        IndexOrdered &Iv = Index[new_v];
        const IndexType &Lv = L[v_id];
        auto &OLv = ordered_L[new_v];
        // Bit-parallel Labels
        memcpy(&Iv.bp_dist, &Lv.bp_dist, BITPARALLEL_SIZE * sizeof(weighti));
        for (inti b_i = 0; b_i < BITPARALLEL_SIZE; ++b_i) {
            memcpy(&Iv.bp_sets[b_i], &Lv.bp_sets[b_i], 2 * sizeof(uint64_t));
        }

        // Normal Labels
        // Traverse v_id's all existing labels
        for (inti b_i = 0; b_i < Lv.batches.size(); ++b_i) {
            idi id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
            idi dist_start_index = Lv.batches[b_i].start_index;
            idi dist_bound_index = dist_start_index + Lv.batches[b_i].size;
            // Traverse dist_matrix
            for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                idi v_start_index = Lv.distances[dist_i].start_index;
                idi v_bound_index = v_start_index + Lv.distances[dist_i].size;
                inti dist = Lv.distances[dist_i].dist;
                for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                    idi tail = Lv.vertices[v_i] + id_offset;
//					idi new_tail = rank2id[tail];
//					new_L[new_v].push_back(make_pair(new_tail, dist));
                    OLv.push_back(std::make_pair(tail, dist));
                }
            }
        }
        // Sort
        sort(OLv.begin(), OLv.end());
        // Store into Index
        inti size_labels = OLv.size();
        labels_count += size_labels;
        Iv.label_id.resize(size_labels + 1); // Adding one for Sentinel
        Iv.label_dists.resize(size_labels + 1); // Adding one for Sentinel
        for (inti l_i = 0; l_i < size_labels; ++l_i) {
            Iv.label_id[l_i] = OLv[l_i].first;
            Iv.label_dists[l_i] = OLv[l_i].second;
        }
        Iv.label_id[size_labels] = num_v; // Sentinel
        Iv.label_dists[size_labels] = WEIGHTI_MAX; // Sentinel
    }
    printf("Label_size: %u mean: %f\n", labels_count, static_cast<double>(labels_count) / num_v);
//	// Test
//	{
//		puts("Asserting...");
//		for (idi v_id = 0; v_id < num_v; ++v_id) {
//			const IndexType &Lv = L[v_id];
//			const IndexOrdered &Iv = Index[rank2id[v_id]];
//			// Bit-parallel Labels
//			for (inti b_i = 0; b_i < BITPARALLEL_SIZE; ++b_i) {
//				assert(Lv.bp_dist[b_i] == Iv.bp_dist[b_i]);
//				assert(Lv.bp_sets[b_i][0] == Iv.bp_sets[b_i][0]);
//				assert(Lv.bp_sets[b_i][1] == Iv.bp_sets[b_i][1]);
//			}
//			// Normal Labels
//			assert(Lv.vertices.size() == Iv.label_id.size());
//			assert(Lv.vertices.size() == Iv.label_dists.size());
////			{
////				inti bound_i = Iv.label_id.size() > 10 ? 10 : Iv.label_id.size();
////				printf("V %u:", rank2id[v_id]);
////				for (inti i = 0; i < bound_i; ++i) {
////					printf(" (%u, %u)", Iv.label_id[i], Iv.label_dists[i]);
////				}
////				puts("");
////			}
//
//		}
//		puts("Asserted.");
//	}
}

template<inti BATCH_SIZE>
weighti ParaVertexCentricPLL<BATCH_SIZE>::query_distance(
        idi a,
        idi b)
{
    idi num_v = num_v_;
    if (a >= num_v || b >= num_v) {
        return a == b ? 0 : WEIGHTI_MAX;
    }

//	// A is shorter than B
//	IndexOrdered &Ia = (Index[a].label_id.size() < Index[b].label_id.size()) ? Index[a] : Index[b];
//	IndexOrdered &Ib = (Index[a].label_id.size() < Index[b].label_id.size()) ? Index[b] : Index[a];

//	// A is longer than B
//	IndexOrdered &Ia = (Index[a].label_id.size() > Index[b].label_id.size()) ? Index[a] : Index[b];
//	IndexOrdered &Ib = (Index[a].label_id.size() > Index[b].label_id.size()) ? Index[b] : Index[a];

    IndexOrdered &Ia = Index[a];
    IndexOrdered &Ib = Index[b];

//	const IndexOrdered &Ia = Index[a];
//	const IndexOrdered &Ib = Index[b];
    inti d = WEIGHTI_MAX;

    _mm_prefetch(&Ia.label_id[0], _MM_HINT_T0);
    _mm_prefetch(&Ib.label_id[0], _MM_HINT_T0);
    _mm_prefetch(&Ia.label_dists[0], _MM_HINT_T0);
    _mm_prefetch(&Ib.label_dists[0], _MM_HINT_T0);

    // Bit-Parallel Labels
    for (int i = 0; i < BITPARALLEL_SIZE; ++i) {
        int td = Ia.bp_dist[i] + Ib.bp_dist[i];
        if (td - 2 <= d) {
            td +=
                    (Ia.bp_sets[i][0] & Ib.bp_sets[i][0]) ? -2 :
                    ((Ia.bp_sets[i][0] & Ib.bp_sets[i][1]) | (Ia.bp_sets[i][1] & Ib.bp_sets[i][0]))
                    ? -1 : 0;

            if (td < d) {
                d = td;
            }
        }
    }

    // Normal Labels (ordered)
//	// Vectorizaed Version
//	vector<idi> &A = Ia.label_id;
//	vector<idi> &B = Ib.label_id;
//	idi len_B = B.size() - 1;
////	idi len_B = B.size();
//	idi bound_b_base_i = len_B - (len_B % NUM_P_INT);
//	idi a_i = 0;
//	idi b_base_i = 0;
//	idi len_A = A.size() - 1;
////	idi len_A = A.size();
//	++length_larger_than_16.second;
//	if (len_B >= 16) {
//		++length_larger_than_16.first;
//	}
//	while (a_i < len_A && b_base_i < bound_b_base_i) {
//		int a = A[a_i];
//		__m512i a_v = _mm512_set1_epi32(a);
//
//		// Packed b
//		__m512i b_v = _mm512_loadu_epi32(&B[b_base_i]); // @suppress("Function cannot be resolved")
//		__mmask16 is_equal_m = _mm512_cmpeq_epi32_mask(a_v, b_v);
//		if (is_equal_m) {
////			if (a == num_v) {
////				break;  // Sentinel
////			}
//			inti td = Ia.label_dists[a_i] + Ib.label_dists[b_base_i + (idi) (log2(is_equal_m))];
//			if (td < d) {
//				d = td;
//			}
//
//			// Advance index
//			if (is_equal_m & (__mmask16) 0x8000) {
//				++a_i;
//				b_base_i += NUM_P_INT;
//			} else {
//				a_i += (a < B[b_base_i + NUM_P_INT - 1]) ? 1 : 0;
//				b_base_i += (B[b_base_i + NUM_P_INT - 1] < a) ? NUM_P_INT : 0;
//			}
//		} else {
//			// Advance index
//			a_i += (a < B[b_base_i + NUM_P_INT - 1]) ? 1 : 0;
//			b_base_i += (B[b_base_i + NUM_P_INT - 1] < a) ? NUM_P_INT : 0;
//		}
//	}
//	while (a_i < len_A && b_base_i < len_B) {
//		if (A[a_i] == B[b_base_i]) {
////			if (a == num_v) {
////				break;  // Sentinel
////			}
//			inti td = Ia.label_dists[a_i] + Ib.label_dists[b_base_i];
//			if (td < d) {
//				d = td;
//			}
//
//			// Advance index
//			++a_i;
//			++b_base_i;
//		} else {
//			// Advance index
//			a_i += (A[a_i] < B[b_base_i]) ? 1 : 0;
//			b_base_i += (B[b_base_i] < A[a_i]) ? 1 : 0;
//		}
//	}

    // Sequential Version
    for (idi i1 = 0, i2 = 0;;) {
        idi v1 = Ia.label_id[i1], v2 = Ib.label_id[i2];
        if (v1 == v2) {
            if (v1 == num_v) {
                break;  // Sentinel
            }
            inti td = Ia.label_dists[i1] + Ib.label_dists[i2];
            if (td < d) {
                d = td;
            }
            ++i1;
            ++i2;
        } else {
            i1 += v1 < v2 ? 1 : 0;
            i2 += v1 > v2 ? 1 : 0;
        }
    }

    if (d >= WEIGHTI_MAX - 2) {
        d = WEIGHTI_MAX;
    }
    return d;
}

template<inti BATCH_SIZE>
void ParaVertexCentricPLL<BATCH_SIZE>::switch_labels_to_old_id(
        const vector<idi> &rank2id,
        const vector<idi> &rank)
{
    idi label_sum = 0;
    idi test_label_sum = 0;

//	idi num_v = rank2id.size();
    idi num_v = rank.size();
    vector<vector<pair < idi, weighti> > > new_L(num_v);
//	for (idi r = 0; r < num_v; ++r) {
//		idi v = rank2id[r];
//		const IndexType &Lr = L[r];
//		IndexType &Lv = new_L[v];
//		idi size = Lr.get_size();
//		label_sum += size;
//		for (idi li = 0; li < size; ++li) {
//			idi l = Lr.get_label_ith_v(li);
//			idi new_l = rank2id[l];
//			Lv.add_label_seq(new_l, Lr.get_label_ith_d(li));
//		}
//	}
//	L = new_L;
    for (idi v_id = 0; v_id < num_v; ++v_id) {
        idi new_v = rank2id[v_id];
        const IndexType &Lv = L[v_id];
        // Traverse v_id's all existing labels
        for (inti b_i = 0; b_i < Lv.batches.size(); ++b_i) {
            idi id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
            idi dist_start_index = Lv.batches[b_i].start_index;
            idi dist_bound_index = dist_start_index + Lv.batches[b_i].size;
            // Traverse dist_matrix
            for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                label_sum += Lv.distances[dist_i].size;
                idi v_start_index = Lv.distances[dist_i].start_index;
                idi v_bound_index = v_start_index + Lv.distances[dist_i].size;
                inti dist = Lv.distances[dist_i].dist;
                for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                    idi tail = Lv.vertices[v_i] + id_offset;
//					idi new_tail = rank2id[tail];
//					new_L[new_v].push_back(make_pair(new_tail, dist));
                    new_L[new_v].push_back(std::make_pair(tail, dist));
                    ++test_label_sum;
                }
            }
        }
    }
    printf("Label sum: %u %u mean: %f\n", label_sum, test_label_sum, label_sum * 1.0 / num_v);

//	// Try to print
//	for (idi v = 0; v < num_v; ++v) {
//		const auto &Lv = new_L[v];
//		idi size = Lv.size();
//		printf("Vertex %u (Size %u):", v, size);
//		for (idi i = 0; i < size; ++i) {
//			printf(" (%u, %d)", Lv[i].first, Lv[i].second);
//			fflush(stdout);
//		}
//		puts("");
//	}

//	// Try query
//	idi u;
//	idi v;
//	while (std::cin >> u >> v) {
//		weighti dist = WEIGHTI_MAX;
//		// Bit Parallel Check
//		const IndexType &idx_u = L[rank[u]];
//		const IndexType &idx_v = L[rank[v]];
//
//		for (inti i = 0; i < BITPARALLEL_SIZE; ++i) {
//			int td = idx_v.bp_dist[i] + idx_u.bp_dist[i];
//			if (td - 2 <= dist) {
//				td +=
//					(idx_v.bp_sets[i][0] & idx_u.bp_sets[i][0]) ? -2 :
//					((idx_v.bp_sets[i][0] & idx_u.bp_sets[i][1])
//							| (idx_v.bp_sets[i][1] & idx_u.bp_sets[i][0]))
//							? -1 : 0;
//				if (td < dist) {
//					dist = td;
//				}
//			}
//		}
//
//		// Normal Index Check
//		const auto &Lu = new_L[u];
//		const auto &Lv = new_L[v];
////		unsorted_map<idi, weighti> markers;
//		map<idi, weighti> markers;
//		for (idi i = 0; i < Lu.size(); ++i) {
//			markers[Lu[i].first] = Lu[i].second;
//		}
//		for (idi i = 0; i < Lv.size(); ++i) {
//			const auto &tmp_l = markers.find(Lv[i].first);
//			if (tmp_l == markers.end()) {
//				continue;
//			}
//			int d = tmp_l->second + Lv[i].second;
//			if (d < dist) {
//				dist = d;
//			}
//		}
//		if (dist == 255) {
//			printf("2147483647\n");
//		} else {
//			printf("%u\n", dist);
//		}
//	}
}

}
#endif /* INCLUDES_PADO_H_ */
