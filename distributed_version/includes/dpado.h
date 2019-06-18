//
// Created by Zhen Peng on 5/28/19.
//

#ifndef PADO_DPADO_H
#define PADO_DPADO_H

#include <vector>
//#include <unordered_map>
#include <map>
#include <algorithm>
#include <iostream>
#include <limits.h>
//#include <xmmintrin.h>
#include <immintrin.h>
#include <bitset>
#include <math.h>
#include <fstream>
#include "globals.h"
#include "dglobals.h"
//#include "graph.h"
#include "dgraph.h"


namespace PADO {

template <VertexID BATCH_SIZE = 1024, VertexID BITPARALLEL_SIZE = 50>
class DistBVCPLL {
private:
//    static const VertexID BITPARALLEL_SIZE = 50;
    VertexID num_v = 0;
    VertexID num_masters = 0;
    int host_id = 0;
    int num_hosts = 0;
    MPI_Datatype V_ID_Type;

    // Structure for the type of label
    struct IndexType {
        struct Batch {
            VertexID batch_id; // Batch ID
            VertexID start_index; // Index to the array distances where the batch starts
            VertexID size; // Number of distances element in this batch

            Batch() = default;
            Batch(VertexID batch_id_, VertexID start_index_, VertexID size_):
                    batch_id(batch_id_), start_index(start_index_), size(size_)
            { }
        };

        struct DistanceIndexType {
            VertexID start_index; // Index to the array vertices where the same-ditance vertices start
            VertexID size; // Number of the same-distance vertices
            UnweightedDist dist; // The real distance

            DistanceIndexType() = default;
            DistanceIndexType(VertexID start_index_, VertexID size_, UnweightedDist dist_):
                    start_index(start_index_), size(size_), dist(dist_)
            { }
        };
//        // Bit-parallel Labels
//        UnweightedDist bp_dist[BITPARALLEL_SIZE];
//        uint64_t bp_sets[BITPARALLEL_SIZE][2];  // [0]: S^{-1}, [1]: S^{0}

        std::vector<Batch> batches; // Batch info
        std::vector<DistanceIndexType> distances; // Distance info
        std::vector<VertexID> vertices; // Vertices in the label, presented as temporary ID

//        // Clean up all labels
//        void cleanup()
//        {
//            std::vector<Batch>().swap(batches);
//            std::vector<DistanceIndexType>().swap(distances);
//            std::vector<VertexID>().swap(vertices);
//        }
    }; //__attribute__((aligned(64)));

    // Structure for the type of temporary label
    struct ShortIndex {
        // I use BATCH_SIZE + 1 bit for indicator bit array.
        // The v.indicator[BATCH_SIZE] is set if in current batch v has got any new labels already.
        // In this way, when do initialization, only initialize those short_index[v] whose indicator[BATCH_SIZE] is set.
        // It is also used for inserting new labels because the label structure.
        std::bitset<BATCH_SIZE + 1> indicator; // Global indicator, indicator[r] (0 <= r < BATCH_SIZE) is set means root r once selected as candidate already
//		bitset<BATCH_SIZE> candidates; // Candidates one iteration, candidates[r] is set means root r is candidate in this iteration

        // Use a queue to store candidates
        std::vector<VertexID> candidates_que = std::vector<VertexID>(BATCH_SIZE);
        VertexID end_candidates_que = 0;
        std::vector<bool> is_candidate = std::vector<bool>(BATCH_SIZE, false);

    }; //__attribute__((aligned(64)));

    // Structure of the public ordered index for distance queries.
    struct IndexOrdered {
//        UnweightedDist bp_dist[BITPARALLEL_SIZE];
//        uint64_t bp_sets[BITPARALLEL_SIZE][2]; // [0]: S^{-1}, [1]: S^{0}

        std::vector<VertexID> label_id;
        std::vector<UnweightedDist> label_dists;
    };

    std::vector<IndexType> L;
    std::vector<IndexOrdered> Index; // Ordered labels for original vertex ID

//    void construct(const DistGraph &G);
//    inline void bit_parallel_labeling(
//            const DistGraph &G,
//            std::vector<IndexType> &L,
//            std::vector<bool> &used_bp_roots);
//    inline bool bit_parallel_checking(
//            VertexID v_id,
//            VertexID w_id,
//            const std::vector<IndexType> &L,
//            UnweightedDist iter);
    inline void batch_process(
            const DistGraph &G,
            VertexID b_id,
            VertexID roots_start,
            VertexID roots_size,
//            std::vector<IndexType> &L,
            const std::vector<bool> &used_bp_roots,
            std::vector<VertexID> &active_queue,
            VertexID &end_active_queue,
            std::vector<VertexID> &got_candidates_queue,
            VertexID &end_got_candidates_queue,
            std::vector<ShortIndex> &short_index,
            std::vector< std::vector<UnweightedDist> > &dist_table,
            std::vector< std::vector<VertexID> > &recved_dist_table,
            std::vector<bool> &got_candidates,
            std::vector<bool> &is_active,
            std::vector<VertexID> &once_candidated_queue,
            VertexID &end_once_candidated_queue,
            std::vector<bool> &once_candidated);
    inline VertexID initialization(
            const DistGraph &G,
            std::vector<ShortIndex> &short_index,
            std::vector< std::vector<UnweightedDist> > &dist_table,
            std::vector< std::vector<VertexID> > &recved_dist_table,
            std::vector<VertexID> &active_queue,
            VertexID &end_active_queue,
            std::vector<VertexID> &once_candidated_queue,
            VertexID &end_once_candidated_queue,
            std::vector<bool> &once_candidated,
            VertexID b_id,
            VertexID roots_start,
            VertexID roots_size,
            std::vector<VertexID> &roots_master_local,
            const std::vector<bool> &used_bp_roots);
    inline void sync_masters_2_mirrors(
            const DistGraph &G,
            const std::vector<VertexID> &active_queue,
            VertexID end_active_queue,
            std::vector<MPI_Request> &requests_send);
    inline void push_label(
            VertexID v_head_global,
            VertexID label_root_id,
            VertexID roots_start,
            const DistGraph &G,
            std::vector<ShortIndex> &short_index,
            std::vector<VertexID> &got_candidates_queue,
            VertexID &end_got_candidates_queue,
            std::vector<bool> &got_candidates,
            std::vector<VertexID> &once_candidated_queue,
            VertexID &end_once_candidated_queue,
            std::vector<bool> &once_candidated,
            const std::vector<bool> &used_bp_roots,
            UnweightedDist iter);
    inline void local_push_labels(
            VertexID v_head_local,
            VertexID roots_start,
            const DistGraph &G,
            std::vector<ShortIndex> &short_index,
            std::vector<VertexID> &got_candidates_queue,
            VertexID &end_got_candidates_queue,
            std::vector<bool> &got_candidates,
            std::vector<VertexID> &once_candidated_queue,
            VertexID &end_once_candidated_queue,
            std::vector<bool> &once_candidated,
            const std::vector<bool> &used_bp_roots,
            UnweightedDist iter);
//    inline bool check_pushed_labels(
//            const DistGraph &G,
//            VertexID v_tail_global,
//            VertexID label_root_id,
//            VertexID roots_start,
//            std::vector<ShortIndex> &short_index,
//            std::vector<VertexID> &once_candidated_queue,
//            VertexID &end_once_candidated_queue,
//            std::vector<bool> &once_candidated,
//            UnweightedDist iter);
//    inline void sync_potential_candidates(
//            const std::vector< std::vector< std::pair<VertexID, VertexID> > > &buffer_send_list,
//            const DistGraph &G,
//            std::vector<ShortIndex> &short_index,
//            std::vector<VertexID> &got_candidates_queue,
//            VertexID &end_got_candidates_queue,
//            std::vector<bool> &got_candidates,
//            std::vector<VertexID> &once_candidated_queue,
//            VertexID &end_once_candidated_queue,
//            std::vector<bool> &once_candidated,
//            VertexID roots_start,
//            UnweightedDist iter);
    inline bool distance_query(
            VertexID cand_root_id,
            VertexID v_id,
            VertexID roots_start,
//            const std::vector<IndexType> &L,
            const std::vector< std::vector<UnweightedDist> > &dist_table,
            UnweightedDist iter);
    inline void insert_label_only(
            VertexID cand_root_id,
            VertexID v_id,
            VertexID roots_start,
            VertexID roots_size,
//            std::vector<IndexType> &L,
            const DistGraph &G,
            std::vector< std::vector<UnweightedDist> > &dist_table,
            std::vector< std::pair<VertexID, VertexID> > &buffer_send,
            UnweightedDist iter);
    inline void update_label_indices(
            VertexID v_id,
            VertexID inserted_count,
//            std::vector<IndexType> &L,
            std::vector<ShortIndex> &short_index,
            VertexID b_id,
            UnweightedDist iter);
    inline void reset_at_end(
            const DistGraph &G,
            VertexID roots_start,
//            VertexID roots_size,
            const std::vector<VertexID> &roots_master_local,
            std::vector< std::vector<UnweightedDist> > &dist_table,
            std::vector< std::vector<VertexID> > &recved_dist_table);

    inline void test_queries_normal_index(std::vector< std::vector< std::pair<VertexID, UnweightedDist> > > new_L);

    // Test only
//	uint64_t normal_hit_count = 0;
//	uint64_t bp_hit_count = 0;
//	uint64_t total_check_count = 0;
//	uint64_t normal_check_count = 0;
//	uint64_t total_candidates_num = 0;
//	uint64_t set_candidates_num = 0;
//	double initializing_time = 0;
//	double candidating_time = 0;
//	double adding_time = 0;
//	double distance_query_time = 0;
//	double init_index_time = 0;
//	double init_dist_matrix_time = 0;
//	double init_start_reset_time = 0;
//	double init_indicators_time = 0;
    //L2CacheMissRate cache_miss;

//	TotalInstructsExe candidating_ins_count;
//	TotalInstructsExe adding_ins_count;
//	TotalInstructsExe bp_labeling_ins_count;
//	TotalInstructsExe bp_checking_ins_count;
//	TotalInstructsExe dist_query_ins_count;
    // End test



public:
    std::pair<uint64_t, uint64_t> length_larger_than_16 = std::make_pair(0, 0);
    DistBVCPLL() = default;
    explicit DistBVCPLL(const DistGraph &G);

//    void print();
    void switch_labels_to_old_id(
            const std::vector<VertexID> &rank2id);
//            const std::vector<VertexID> &rank);
    void store_index_to_file(
            const char *filename,
            const std::vector<VertexID> &rank);
    void load_index_from_file(
            const char *filename);
    void order_labels(
            const std::vector<VertexID> &rank2id);
//            const std::vector<VertexID> &rank);
    UnweightedDist query_distance(
            VertexID a,
            VertexID b);
    UnweightedDist dist_distance_query_pair(
            VertexID a_global,
            VertexID b_global,
            const DistGraph &G);
}; // class DistBVCPLL

//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//const VertexID DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::BITPARALLEL_SIZE;

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::DistBVCPLL(const DistGraph &G)
{
    num_v = G.num_v;
    assert(num_v >= BATCH_SIZE);
    num_masters = G.num_masters;
    host_id = G.host_id;
    num_hosts = G.num_hosts;
    V_ID_Type = G.V_ID_Type;
//    L.resize(num_v);
    L.resize(num_masters);
    VertexID remainer = num_v % BATCH_SIZE;
    VertexID b_i_bound = num_v / BATCH_SIZE;
    std::vector<bool> used_bp_roots(num_v, false);
    //cache_miss.measure_start();
    double time_labeling = -WallTimer::get_time_mark();

//	double bp_labeling_time = -WallTimer::get_time_mark();
//	bp_labeling_ins_count.measure_start();
//    bit_parallel_labeling(
//            G,
////            L,
//            used_bp_roots);
//	bp_labeling_ins_count.measure_stop();
//	bp_labeling_time += WallTimer::get_time_mark();

    std::vector<VertexID> active_queue(num_masters); // Any vertex v who is active should be put into this queue.
//    std::vector<VertexID> active_queue(num_v); // Any vertex v who is active should be put into this queue.
    VertexID end_active_queue = 0;
    std::vector<bool> is_active(num_masters, false);// is_active[v] is true means vertex v is in the active queue.
//    std::vector<bool> is_active(num_v, false);// is_active[v] is true means vertex v is in the active queue.
    std::vector<VertexID> got_candidates_queue(num_masters); // Any vertex v who got candidates should be put into this queue.
//    std::vector<VertexID> got_candidates_queue(num_v); // Any vertex v who got candidates should be put into this queue.
    VertexID end_got_candidates_queue = 0;
    std::vector<bool> got_candidates(num_masters, false); // got_candidates[v] is true means vertex v is in the queue got_candidates_queue
//    std::vector<bool> got_candidates(num_v, false); // got_candidates[v] is true means vertex v is in the queue got_candidates_queue
    std::vector<ShortIndex> short_index(num_masters);
//    std::vector<ShortIndex> short_index(num_v);
    std::vector< std::vector<UnweightedDist> > dist_table(BATCH_SIZE, std::vector<UnweightedDist>(num_v, MAX_UNWEIGHTED_DIST));

    std::vector<VertexID> once_candidated_queue(num_masters); // if short_index[v].indicator.any() is true, v is in the queue.
//    std::vector<VertexID> once_candidated_queue(num_v); // if short_index[v].indicator.any() is true, v is in the queue.
    // Used mainly for resetting short_index[v].indicator.
    VertexID end_once_candidated_queue = 0;
    std::vector<bool> once_candidated(num_masters, false);
//    std::vector<bool> once_candidated(num_v, false);
    std::vector< std::vector<VertexID> > recved_dist_table(BATCH_SIZE);

    //printf("b_i_bound: %u\n", b_i_bound);//test
    for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
        if (0 == host_id) {
            printf("b_i: %u\n", b_i);//test
        }

        batch_process(
                G,
                b_i,
                b_i * BATCH_SIZE,
                BATCH_SIZE,
//                L,
                used_bp_roots,
                active_queue,
                end_active_queue,
                got_candidates_queue,
                end_got_candidates_queue,
                short_index,
                dist_table,
                recved_dist_table,
                got_candidates,
                is_active,
                once_candidated_queue,
                end_once_candidated_queue,
                once_candidated);
//        exit(EXIT_SUCCESS); //test
    }
    if (remainer != 0) {
        if (0 == host_id) {
            printf("b_i: %u\n", b_i_bound);//test
        }
        batch_process(
                G,
                b_i_bound,
                b_i_bound * BATCH_SIZE,
                remainer,
//                L,
                used_bp_roots,
                active_queue,
                end_active_queue,
                got_candidates_queue,
                end_got_candidates_queue,
                short_index,
                dist_table,
                recved_dist_table,
                got_candidates,
                is_active,
                once_candidated_queue,
                end_once_candidated_queue,
                once_candidated);
    }
    time_labeling += WallTimer::get_time_mark();
    //cache_miss.measure_stop();

    // Test
    setlocale(LC_NUMERIC, "");
    if (0 == host_id) {
        printf("BATCH_SIZE: %u\n", BATCH_SIZE);
        printf("BP_Size: %u\n", BITPARALLEL_SIZE);
    }

//	printf("BP_labeling: %f %.2f%%\n", bp_labeling_time, bp_labeling_time / time_labeling * 100);
//	printf("Initializing: %f %.2f%%\n", initializing_time, initializing_time / time_labeling * 100);
//		printf("\tinit_start_reset_time: %f (%f%%)\n", init_start_reset_time, init_start_reset_time / initializing_time * 100);
//		printf("\tinit_index_time: %f (%f%%)\n", init_index_time, init_index_time / initializing_time * 100);
//			printf("\t\tinit_indicators_time: %f (%f%%)\n", init_indicators_time, init_indicators_time / init_index_time * 100);
//		printf("\tinit_dist_matrix_time: %f (%f%%)\n", init_dist_matrix_time, init_dist_matrix_time / initializing_time * 100);
//	printf("Candidating: %f %.2f%%\n", candidating_time, candidating_time / time_labeling * 100);
//	printf("Adding: %f %.2f%%\n", adding_time, adding_time / time_labeling * 100);
//		printf("distance_query_time: %f %.2f%%\n", distance_query_time, distance_query_time / time_labeling * 100);
//		uint64_t total_check_count = bp_hit_count + normal_check_count;
//		printf("total_check_count: %'llu\n", total_check_count);
//		printf("bp_hit_count: %'llu %.2f%%\n",
//						bp_hit_count,
//						bp_hit_count * 100.0 / total_check_count);
//		printf("normal_check_count: %'llu %.2f%%\n", normal_check_count, normal_check_count * 100.0 / total_check_count);
//		printf("total_candidates_num: %'llu set_candidates_num: %'llu %.2f%%\n",
//							total_candidates_num,
//							set_candidates_num,
//							set_candidates_num * 100.0 / total_candidates_num);
//		printf("\tnormal_hit_count (to total_check, to normal_check): %llu (%f%%, %f%%)\n",
//						normal_hit_count,
//						normal_hit_count * 100.0 / total_check_count,
//						normal_hit_count * 100.0 / (total_check_count - bp_hit_count));
    //cache_miss.print();
//	printf("Candidating: "); candidating_ins_count.print();
//	printf("Adding: "); adding_ins_count.print();
//	printf("BP_Labeling: "); bp_labeling_ins_count.print();
//	printf("BP_Checking: "); bp_checking_ins_count.print();
//	printf("distance_query: "); dist_query_ins_count.print();

    printf("host_id: %u Local_labeling_time: %.2f seconds\n", host_id, time_labeling);
    double global_time_labeling;
    MPI_Allreduce(&time_labeling,
            &global_time_labeling,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == host_id) {
        printf("Global_labeling_time: %.2f seconds\n", global_time_labeling);
    }
    // End test
}

//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::bit_parallel_labeling(
//        const DistGraph &G,
//        std::vector<IndexType> &L,
//        std::vector<bool> &used_bp_roots)
//{
//    VertexID num_v = G.num_v;
//    EdgeID num_e = G.num_e;
//
//    std::vector<UnweightedDist> tmp_d(num_v); // distances from the root to every v
//    std::vector<std::pair<uint64_t, uint64_t> > tmp_s(num_v); // first is S_r^{-1}, second is S_r^{0}
//    std::vector<VertexID> que(num_v); // active queue
//    std::vector<std::pair<VertexID, VertexID> > sibling_es(num_e); // siblings, their distances to the root are equal (have difference of 0)
//    std::vector<std::pair<VertexID, VertexID> > child_es(num_e); // child and father, their distances to the root have difference of 1.
//
//    VertexID r = 0; // root r
//    for (VertexID i_bpspt = 0; i_bpspt < BITPARALLEL_SIZE; ++i_bpspt) {
//        while (r < num_v && used_bp_roots[r]) {
//            ++r;
//        }
//        if (r == num_v) {
//            for (VertexID v = 0; v < num_v; ++v) {
//                L[v].bp_dist[i_bpspt] = MAX_UNWEIGHTED_DIST;
//            }
//            continue;
//        }
//        used_bp_roots[r] = true;
//
//        fill(tmp_d.begin(), tmp_d.end(), MAX_UNWEIGHTED_DIST);
//        fill(tmp_s.begin(), tmp_s.end(), std::make_pair(0, 0));
//
//        VertexID que_t0 = 0, que_t1 = 0, que_h = 0;
//        que[que_h++] = r;
//        tmp_d[r] = 0;
//        que_t1 = que_h;
//
//        int ns = 0; // number of selected neighbor, default 64
//        // the edge of one vertex in G is ordered decreasingly to rank, lower rank first, so here need to traverse edges backward
//        // There was a bug cost countless time: the unsigned iterator i might decrease to zero and then flip to the INF.
////		VertexID i_bound = G.vertices[r] - 1;
////		VertexID i_start = i_bound + G.out_degrees[r];
////		for (VertexID i = i_start; i > i_bound; --i) {
//        //int i_bound = G.vertices[r];
//        //int i_start = i_bound + G.out_degrees[r] - 1;
//        //for (int i = i_start; i >= i_bound; --i) {
//        VertexID d_i_bound = G.out_degrees[r];
//        EdgeID i_start = G.vertices[r] + d_i_bound - 1;
//        for (VertexID d_i = 0; d_i < d_i_bound; ++d_i) {
//            EdgeID i = i_start - d_i;
//            VertexID v = G.out_edges[i];
//            if (!used_bp_roots[v]) {
//                used_bp_roots[v] = true;
//                // Algo3:line4: for every v in S_r, (dist[v], S_r^{-1}[v], S_r^{0}[v]) <- (1, {v}, empty_set)
//                que[que_h++] = v;
//                tmp_d[v] = 1;
//                tmp_s[v].first = 1ULL << ns;
//                if (++ns == 64) break;
//            }
//        }
//        //}
////		}
//
//        for (UnweightedDist d = 0; que_t0 < que_h; ++d) {
//            VertexID num_sibling_es = 0, num_child_es = 0;
//
//            for (VertexID que_i = que_t0; que_i < que_t1; ++que_i) {
//                VertexID v = que[que_i];
//                EdgeID i_start = G.vertices[v];
//                EdgeID i_bound = i_start + G.out_degrees[v];
//                for (EdgeID i = i_start; i < i_bound; ++i) {
//                    VertexID tv = G.out_edges[i];
//                    UnweightedDist td = d + 1;
//
//                    if (d > tmp_d[tv]) {
//                        ;
//                    }
//                    else if (d == tmp_d[tv]) {
//                        if (v < tv) { // ??? Why need v < tv !!! Because it's a undirected graph.
//                            sibling_es[num_sibling_es].first  = v;
//                            sibling_es[num_sibling_es].second = tv;
//                            ++num_sibling_es;
//                        }
//                    } else { // d < tmp_d[tv]
//                        if (tmp_d[tv] == MAX_UNWEIGHTED_DIST) {
//                            que[que_h++] = tv;
//                            tmp_d[tv] = td;
//                        }
//                        child_es[num_child_es].first  = v;
//                        child_es[num_child_es].second = tv;
//                        ++num_child_es;
//                    }
//                }
//            }
//
//            for (VertexID i = 0; i < num_sibling_es; ++i) {
//                VertexID v = sibling_es[i].first, w = sibling_es[i].second;
//                tmp_s[v].second |= tmp_s[w].first;
//                tmp_s[w].second |= tmp_s[v].first;
//            }
//            for (VertexID i = 0; i < num_child_es; ++i) {
//                VertexID v = child_es[i].first, c = child_es[i].second;
//                tmp_s[c].first  |= tmp_s[v].first;
//                tmp_s[c].second |= tmp_s[v].second;
//            }
//
//            que_t0 = que_t1;
//            que_t1 = que_h;
//        }
//
//        for (VertexID v = 0; v < num_v; ++v) {
//            L[v].bp_dist[i_bpspt] = tmp_d[v];
//            L[v].bp_sets[i_bpspt][0] = tmp_s[v].first; // S_r^{-1}
//            L[v].bp_sets[i_bpspt][1] = tmp_s[v].second & ~tmp_s[v].first; // Only need those r's neighbors who are not already in S_r^{-1}
//        }
//    }
//
//}

//// Function bit parallel checking:
//// return false if shortest distance exits in bp labels, return true if bp labels cannot cover the distance
//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//inline bool DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::bit_parallel_checking(
//        VertexID v_id,
//        VertexID w_id,
//        const std::vector<IndexType> &L,
//        UnweightedDist iter)
//{
//    // Bit Parallel Checking: if label_real_id to v_tail has shorter distance already
//    const IndexType &Lv = L[v_id];
//    const IndexType &Lw = L[w_id];
//
//    _mm_prefetch(&Lv.bp_dist[0], _MM_HINT_T0);
//    _mm_prefetch(&Lv.bp_sets[0][0], _MM_HINT_T0);
//    _mm_prefetch(&Lw.bp_dist[0], _MM_HINT_T0);
//    _mm_prefetch(&Lw.bp_sets[0][0], _MM_HINT_T0);
//    for (VertexID i = 0; i < BITPARALLEL_SIZE; ++i) {
//        VertexID td = Lv.bp_dist[i] + Lw.bp_dist[i]; // Use type VertexID in case of addition of two INF.
//        if (td - 2 <= iter) {
//            td +=
//                    (Lv.bp_sets[i][0] & Lw.bp_sets[i][0]) ? -2 :
//                    ((Lv.bp_sets[i][0] & Lw.bp_sets[i][1]) |
//                     (Lv.bp_sets[i][1] & Lw.bp_sets[i][0]))
//                    ? -1 : 0;
//            if (td <= iter) {
////				++bp_hit_count;
//                return false;
//            }
//        }
//    }
//    return true;
//}


// Function for initializing at the begin of a batch
// For a batch, initialize the temporary labels and real labels of roots;
// traverse roots' labels to initialize distance buffer;
// unset flag arrays is_active and got_labels
template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline VertexID DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::initialization(
        const DistGraph &G,
        std::vector<ShortIndex> &short_index,
        std::vector< std::vector<UnweightedDist> > &dist_table,
        std::vector< std::vector<VertexID> > &recved_dist_table,
        std::vector<VertexID> &active_queue,
        VertexID &end_active_queue,
        std::vector<VertexID> &once_candidated_queue,
        VertexID &end_once_candidated_queue,
        std::vector<bool> &once_candidated,
        VertexID b_id,
        VertexID roots_start,
        VertexID roots_size,
        std::vector<VertexID> &roots_master_local,
        const std::vector<bool> &used_bp_roots)
{
    //MPI_Datatype V_ID_Type = MPI_Instance::get_mpi_datatype<VertexID>();
    VertexID roots_bound = roots_start + roots_size;
    for (VertexID r_global = roots_start; r_global < roots_bound; ++r_global) {
        if (G.get_master_host_id(r_global) == host_id && !used_bp_roots[r_global]) {
            roots_master_local.push_back(G.get_local_vertex_id(r_global));
        }
    }
//	init_index_time -= WallTimer::get_time_mark();
    // Short_index
    {
//		init_indicators_time -= WallTimer::get_time_mark();
        for (VertexID v_i = 0; v_i < end_once_candidated_queue; ++v_i) {
            VertexID v_local = once_candidated_queue[v_i];
            short_index[v_local].indicator.reset();
            once_candidated[v_local] = false;
        }
        end_once_candidated_queue = 0;
        for (VertexID r_local : roots_master_local) {
            short_index[r_local].indicator.set(G.get_global_vertex_id(r_local) - roots_start); // v itself
            short_index[r_local].indicator.set(BATCH_SIZE); // v got labels
        }
//        for (VertexID v_i = 0; v_i < end_once_candidated_queue; ++v_i) {
//            VertexID v = once_candidated_queue[v_i];
//            short_index[v].indicator.reset();
//            once_candidated[v] = false;
//        }
//        end_once_candidated_queue = 0;
//        for (VertexID v = roots_start; v < roots_bound; ++v) {
//            if (!used_bp_roots[v]) {
//                short_index[v].indicator.set(v - roots_start); // v itself
//                short_index[v].indicator.set(BATCH_SIZE); // v got labels
//            }
//        }
//		init_indicators_time += WallTimer::get_time_mark();
    }
//
    // Real Index
    {
//		IndexType &Lr = nullptr;
        for (VertexID r_local : roots_master_local) {
            IndexType &Lr = L[r_local];
            Lr.batches.emplace_back(
                    b_id, // Batch ID
                    Lr.distances.size(), // start_index
                    1); // size
            Lr.distances.emplace_back(
                    Lr.vertices.size(), // start_index
                    1, // size
                    0); // dist
            Lr.vertices.push_back(G.get_global_vertex_id(r_local) - roots_start);
        }
//        for (VertexID r_id = 0; r_id < roots_size; ++r_id) {
//            if (used_bp_roots[r_id + roots_start]) {
//                continue;
//            }
//            IndexType &Lr = L[r_id + roots_start];
////            Lr.batches.push_back(IndexType::Batch(
////                    b_id, // Batch ID
////                    Lr.distances.size(), // start_index
////                    1)); // size
//            Lr.batches.emplace_back(
//                    b_id, // Batch ID
//                    Lr.distances.size(), // start_index
//                    1); // size
////            Lr.distances.push_back(IndexType::DistanceIndexType(
////                    Lr.vertices.size(), // start_index
////                    1, // size
////                    0)); // dist
//            Lr.distances.emplace_back(
//                    Lr.vertices.size(), // start_index
//                    1, // size
//                    0); // dist
//            Lr.vertices.push_back(r_id);
//        }
    }
//	init_index_time += WallTimer::get_time_mark();
//	init_dist_matrix_time -= WallTimer::get_time_mark();
	struct LabelTableUnit {
		VertexID root_id;
		VertexID label_global_id;
		UnweightedDist dist;
		LabelTableUnit() = default;
		LabelTableUnit(VertexID r, VertexID l, UnweightedDist d) :
			root_id(r), label_global_id(l), dist(d) {  }
	};
	std::vector<LabelTableUnit> buffer_send; // buffer for sending
    // Dist_matrix
    {
        // Deprecated Old method: unpack the IndexType structure before sending.
        for (VertexID r_local : roots_master_local) {
            // The distance table.
            IndexType &Lr = L[r_local];
            VertexID r_root_id = G.get_global_vertex_id(r_local) - roots_start;
            VertexID b_i_bound = Lr.batches.size();
            _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
            _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
            _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
            // Traverse batches array
            for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
                VertexID id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
                VertexID dist_start_index = Lr.batches[b_i].start_index;
                VertexID dist_bound_index = dist_start_index + Lr.batches[b_i].size;
                // Traverse distances array
                for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                    VertexID v_start_index = Lr.distances[dist_i].start_index;
                    VertexID v_bound_index = v_start_index + Lr.distances[dist_i].size;
                    UnweightedDist dist = Lr.distances[dist_i].dist;
                    // Traverse vertices array
                    for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                        // Write into the dist_table
                        dist_table[r_root_id][Lr.vertices[v_i] + id_offset] = dist; // distance table
						buffer_send.emplace_back(r_root_id, Lr.vertices[v_i] + id_offset, dist); // buffer for sending
                    }
                }
            }
        }
    }
//	init_dist_matrix_time += WallTimer::get_time_mark();
//    // Broadcast local roots labels
	std::vector<MPI_Request> requests_send(num_hosts - 1);
	{
		for (int loc = 0; loc < num_hosts - 1; ++loc) {
			int dest_host_id = G.buffer_send_list_loc_2_master_host_id(loc);
			MPI_Isend(buffer_send.data(),
					MPI_Instance::get_sending_size(buffer_send),
					MPI_CHAR,
					dest_host_id,
					SENDING_DIST_TABLE,
					MPI_COMM_WORLD,
					&requests_send[loc]);
		}
	}
//	std::vector<MPI_Request> requests_send((num_hosts - 1) * (1 + 4 * roots_master_local.size()));
//	VertexID end_requests_send = 0;
//    {
//        for (int loc = 0; loc < num_hosts - 1; ++loc) {
//            int dest_host_id = G.buffer_send_list_loc_2_master_host_id(loc);
//            // How many masters to be sent
//            VertexID num_root_masters = roots_master_local.size();
//            MPI_Isend(&num_root_masters,
//                    1,
//                    V_ID_Type,
//                    dest_host_id,
//                    SENDING_NUM_ROOT_MASTERS,
//                    MPI_COMM_WORLD,
//					&requests_send[end_requests_send++]);
//			printf("@%u host_id: %u send_to: %u num_root_masters: %u\n", __LINE__, host_id, dest_host_id, num_root_masters);//test
//            // For every root master
//            for (VertexID r_local : roots_master_local) {
//                // Which root
//                IndexType &Lr = L[r_local];
//                VertexID r_root_id = G.get_global_vertex_id(r_local) - roots_start;
//                MPI_Isend(&r_root_id,
//                        1,
//                        V_ID_Type,
//                        dest_host_id,
//                        SENDING_ROOT_ID,
//                        MPI_COMM_WORLD,
//						&requests_send[end_requests_send++]);
//				//printf("@%u host_id: %u send_to: %u r_root_id: %u\n", __LINE__, host_id, dest_host_id, r_root_id);//test
//                // The Batches array
//                MPI_Isend(Lr.batches.data(),
//                         MPI_Instance::get_sending_size(Lr.batches),
//                         MPI_CHAR,
//                         dest_host_id,
//                         SENDING_INDEXTYPE_BATCHES,
//                         MPI_COMM_WORLD,
//						 &requests_send[end_requests_send++]);
//				//printf("@%u host_id: %u send_to: %u batches.size(): %lu\n", __LINE__, host_id, dest_host_id, Lr.batches.size());//test
//                // The Distances array
//                MPI_Isend(Lr.distances.data(),
//                         MPI_Instance::get_sending_size(Lr.distances),
//                         MPI_CHAR,
//                         dest_host_id,
//                         SENDING_INDEXTYPE_DISTANCES,
//                         MPI_COMM_WORLD,
//						 &requests_send[end_requests_send++]);
//				//printf("@%u host_id: %u send_to: %u distances.size(): %lu\n", __LINE__, host_id, dest_host_id, Lr.distances.size());//test
//                // The Vertices arrray
//                MPI_Isend(Lr.vertices.data(),
//                         MPI_Instance::get_sending_size(Lr.vertices),
//                         MPI_CHAR,
//                         dest_host_id,
//                         SENDING_INDEXTYPE_VERTICES,
//                         MPI_COMM_WORLD,
//						 &requests_send[end_requests_send++]);
//				//printf("@%u host_id: %u send_to: %u vertices.size(): %lu\n", __LINE__, host_id, dest_host_id, Lr.vertices.size());//test
//            }
//        }
//		printf("@%u host_id: %d broadcast\n", __LINE__, host_id);
//		assert(end_requests_send == requests_send.size());
//    }

//    // Receive labels from every other host
	{
		std::vector<LabelTableUnit> buffer_recv;
		for (int h_i = 0; h_i < num_hosts - 1; ++h_i) {
			MPI_Instance::receive_dynamic_buffer_from_any(buffer_recv,
					num_hosts,
					SENDING_DIST_TABLE);
			if (buffer_recv.empty()) {
			    continue;
			}
			for (const auto &l : buffer_recv) {
				VertexID root_id = l.root_id;
				VertexID label_global_id = l.label_global_id;
				UnweightedDist dist = l.dist;
				dist_table[root_id][label_global_id] = dist;
                // Record the received label in recved_dist_table, for later reset
                recved_dist_table[root_id].push_back(label_global_id);
			}
		}
		MPI_Waitall(num_hosts - 1,
				requests_send.data(),
				MPI_STATUSES_IGNORE);
	}
//    {
//        for (int h_i = 0; h_i < num_hosts - 1; ++h_i) {
//            VertexID num_root_recieved;
//            MPI_Status status_recv;
//            MPI_Recv(&num_root_recieved,
//                    1,
//                    V_ID_Type,
//                    MPI_ANY_SOURCE,
//                    SENDING_NUM_ROOT_MASTERS,
//                    MPI_COMM_WORLD,
//                    &status_recv);
//            int source = status_recv.MPI_SOURCE;
//			printf("@%u host_id: %u recv_from: %u num_root_recieved: %u\n", __LINE__, host_id, source, num_root_recieved);//test
//            for (VertexID r_i = 0; r_i < num_root_recieved; ++r_i) {
//                // Receive the root_id
//                VertexID r_root_id;
//                MPI_Recv(&r_root_id,
//                        1,
//                        V_ID_Type,
//                        source,
//                        SENDING_ROOT_ID,
//                        MPI_COMM_WORLD,
//                        MPI_STATUS_IGNORE);
//				printf("@%u host_id: %u recv_from: %u r_root_id: %u\n", __LINE__, host_id, source, r_root_id);//test
//                // Receive Batches array
//                std::vector<typename IndexType::Batch> batches;
//				MPI_Instance::receive_dynamic_buffer_from_source(batches,
//                        num_hosts,
//                        source,
//                        SENDING_INDEXTYPE_BATCHES);
//				printf("@%u host_id: %u recv_from: %u batches.size(): %lu\n", __LINE__, host_id, source, batches.size());//test
//                // Receive Distances array
//                std::vector<typename IndexType::DistanceIndexType> distances;
//                MPI_Instance::receive_dynamic_buffer_from_source(distances,
//                        num_hosts,
//                        source,
//                        SENDING_INDEXTYPE_DISTANCES);
//				printf("@%u host_id: %u recv_from: %u distances.size(): %lu\n", __LINE__, host_id, source, distances.size());//test
//                // Receive Vertices array
//                std::vector<VertexID> vertices;
//                MPI_Instance::receive_dynamic_buffer_from_source(vertices,
//                        num_hosts,
//                        source,
//                        SENDING_INDEXTYPE_VERTICES);
//				printf("@%u host_id: %u recv_from: %u vertices.size(): %lu\n", __LINE__, host_id, source, vertices.size());//test
//                // Traverse labels to setup distance table
//                VertexID b_i_bound = batches.size();
//                // Traverse the batches array
//                for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
//                    VertexID id_offset = batches[b_i].batch_id * BATCH_SIZE;
//                    VertexID dist_start_index = batches[b_i].start_index;
//                    VertexID dist_bound_index = dist_start_index + batches[b_i].size;
//                    // Traverse distances array
//                    for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
//                        VertexID v_start_index = distances[dist_i].start_index;
//                        VertexID v_bound_index = v_start_index + distances[dist_i].size;
//                        UnweightedDist dist = distances[dist_i].dist;
//                        // Traverse vertices array
//                        for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
////                        VertexID label_global_id = Lr.vertices[v_i] + id_offset;
//                            dist_table[r_root_id][vertices[v_i] + id_offset] = dist; // distance table
////                        buffer_send.emplace_back(r_root_id, label_global_id, dist); // buffer for sending
//                        }
//                    }
//                }
//            }
//        }
//		printf("@%u host_id: %d received\n", __LINE__, host_id);
//    }
//	MPI_Waitall(requests_send.size(),
//			requests_send.data(),
//			MPI_STATUSES_IGNORE);

    // TODO: parallel enqueue
    // Active_queue
    VertexID global_num_actives = 0; // global number of active vertices.
    {
        for (VertexID r_local : roots_master_local) {
            active_queue[end_active_queue++] = r_local;
        }
//        printf("@%u host_id: %d end_active_queue: %u\n", __LINE__, host_id, end_active_queue); //test
//        for (VertexID r_real_id = roots_start; r_real_id < roots_bound; ++r_real_id) {
//            if (!used_bp_roots[r_real_id]) {
//                active_queue[end_active_queue++] = r_real_id;
//            }
//        }
        // Get the global number of active vertices;
        MPI_Allreduce(&end_active_queue,
                      &global_num_actives,
                      1,
                      V_ID_Type,
                      MPI_SUM,
                      MPI_COMM_WORLD);
    }

//    {// test
//        printf("@%u host_id: %u global_num_actives: %u\n", __LINE__, host_id, global_num_actives);
//    // Print the dist table
//        std::string filename = "output" + std::to_string(host_id) + ".txt";
//        FILE *fout = fopen(filename.c_str(), "w");
//		if (fout == nullptr) {
//			fprintf(stderr, "Error: cannot create file %s\n", filename.c_str());
//			exit(EXIT_FAILURE);
//		}
//		fprintf(fout, "host_id: %d size_roots_master_local: %lu\n", host_id, roots_master_local.size());
//		for (VertexID r = 0; r < dist_table.size(); ++r) {
//			fprintf(fout, "[%u,%u]: %u\n", r, r, dist_table[r][r]);
//		}
////        for (VertexID r = 0; r < dist_table.size(); ++r) {
////            for (VertexID v = 0; v < dist_table[r].size(); ++v) {
////                fprintf(fout, "[%u,%u]: %u\n", r, v, dist_table[r][v]);
////            }
////        }
//		fclose(fout);
//        exit(EXIT_SUCCESS);
//    }
    return global_num_actives;
}

//// Function: Vertex v_tail checks label (label_root_id, iter) received from its neighbor,
//// If the label is valid as candidate (need to be added into its candidate labels), return true; otherwise return false.
//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//inline bool DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::check_pushed_labels(
//        const DistGraph &G,
//        VertexID v_tail_global,
//        VertexID label_root_id,
//        VertexID roots_start,
//        std::vector<ShortIndex> &short_index,
//        std::vector<VertexID> &once_candidated_queue,
//        VertexID &end_once_candidated_queue,
//        std::vector<bool> &once_candidated,
//        UnweightedDist iter)
//{
//    VertexID v_tail_local = G.get_local_vertex_id(v_tail_global);
//    VertexID label_global_id = label_root_id + roots_start;
//    if (v_tail_global <= label_global_id) {
//        // v_tail_global has higher rank than all remaining labels
//        return false;
//    }
//    ShortIndex &SI_v_tail = short_index[v_tail_local];
//    if (SI_v_tail.indicator[label_root_id]) {
//        // The label is already selected before
//        return false;
//    }
//    // Record label_root_id as once selected by v_tail_global
//    SI_v_tail.indicator.set(label_root_id);
//    // Add into once_candidated_queue
//
//    if (!once_candidated[v_tail_local]) {
//        // If v_tail_global is not in the once_candidated_queue yet, add it in
//        once_candidated[v_tail_local] = true;
//        once_candidated_queue[end_once_candidated_queue++] = v_tail_local;
//    }
//
////            // Bit Parallel Checking: if label_global_id to v_tail_global has shorter distance already
////            //			++total_check_count;
////            const IndexType &L_label = L[label_global_id];
////
////            _mm_prefetch(&L_label.bp_dist[0], _MM_HINT_T0);
////            _mm_prefetch(&L_label.bp_sets[0][0], _MM_HINT_T0);
//////			bp_checking_ins_count.measure_start();
////            bool no_need_add = false;
////            for (VertexID i = 0; i < BITPARALLEL_SIZE; ++i) {
////                VertexID td = L_label.bp_dist[i] + L_tail.bp_dist[i];
////                if (td - 2 <= iter) {
////                    td +=
////                            (L_label.bp_sets[i][0] & L_tail.bp_sets[i][0]) ? -2 :
////                            ((L_label.bp_sets[i][0] & L_tail.bp_sets[i][1]) |
////                             (L_label.bp_sets[i][1] & L_tail.bp_sets[i][0]))
////                            ? -1 : 0;
////                    if (td <= iter) {
////                        no_need_add = true;
//////						++bp_hit_count;
////                        break;
////                    }
////                }
////            }
////            if (no_need_add) {
//////				bp_checking_ins_count.measure_stop();
////                continue;
////            }
//////			bp_checking_ins_count.measure_stop();
//    if (SI_v_tail.is_candidate[label_root_id]) {
//        return false;
//    }
//    SI_v_tail.is_candidate[label_root_id] = true;
//    SI_v_tail.candidates_que[SI_v_tail.end_candidates_que++] = label_root_id;
//    {// Just for the complain from the compiler
//        assert(iter >= iter);
//    }
//    return true;
//}

// Function: push v_head_global's newly added labels to its all neighbors.
template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::push_label(
        VertexID v_head_global,
        VertexID label_root_id,
        VertexID roots_start,
        const DistGraph &G,
        std::vector<ShortIndex> &short_index,
        std::vector<VertexID> &got_candidates_queue,
        VertexID &end_got_candidates_queue,
        std::vector<bool> &got_candidates,
        std::vector<VertexID> &once_candidated_queue,
        VertexID &end_once_candidated_queue,
        std::vector<bool> &once_candidated,
        const std::vector<bool> &used_bp_roots,
        UnweightedDist iter)
{
    VertexID label_global_id = label_root_id + roots_start;
    EdgeID e_i_start = G.vertices_idx[v_head_global];
    EdgeID e_i_bound = e_i_start + G.local_out_degrees[v_head_global];
    for (EdgeID e_i = e_i_start; e_i < e_i_bound; ++e_i) {
        VertexID v_tail_global = G.out_edges[e_i];
        if (used_bp_roots[v_tail_global]) {
            continue;
        }
        if (v_tail_global < roots_start) { // all remaining v_tail_global has higher rank than any roots, then no roots can push new labels to it.
            return;
        }

        VertexID v_tail_local = G.get_local_vertex_id(v_tail_global);
        if (v_tail_global <= label_global_id) {
            // remaining v_tail_global has higher rank than the label
            return;
        }
        ShortIndex &SI_v_tail = short_index[v_tail_local];
        if (SI_v_tail.indicator[label_root_id]) {
            // The label is already selected before
            continue;
        }
        // Record label_root_id as once selected by v_tail_global
        SI_v_tail.indicator.set(label_root_id);
        // Add into once_candidated_queue

        if (!once_candidated[v_tail_local]) {
            // If v_tail_global is not in the once_candidated_queue yet, add it in
            once_candidated[v_tail_local] = true;
            once_candidated_queue[end_once_candidated_queue++] = v_tail_local;
        }
//            // Bit Parallel Checking: if label_global_id to v_tail_global has shorter distance already
//            //			++total_check_count;
//            const IndexType &L_label = L[label_global_id];
//
//            _mm_prefetch(&L_label.bp_dist[0], _MM_HINT_T0);
//            _mm_prefetch(&L_label.bp_sets[0][0], _MM_HINT_T0);
////			bp_checking_ins_count.measure_start();
//            bool no_need_add = false;
//            for (VertexID i = 0; i < BITPARALLEL_SIZE; ++i) {
//                VertexID td = L_label.bp_dist[i] + L_tail.bp_dist[i];
//                if (td - 2 <= iter) {
//                    td +=
//                            (L_label.bp_sets[i][0] & L_tail.bp_sets[i][0]) ? -2 :
//                            ((L_label.bp_sets[i][0] & L_tail.bp_sets[i][1]) |
//                             (L_label.bp_sets[i][1] & L_tail.bp_sets[i][0]))
//                            ? -1 : 0;
//                    if (td <= iter) {
//                        no_need_add = true;
////						++bp_hit_count;
//                        break;
//                    }
//                }
//            }
//            if (no_need_add) {
////				bp_checking_ins_count.measure_stop();
//                continue;
//            }
////			bp_checking_ins_count.measure_stop();
        if (SI_v_tail.is_candidate[label_root_id]) {
            continue;
        }
        SI_v_tail.is_candidate[label_root_id] = true;
        SI_v_tail.candidates_que[SI_v_tail.end_candidates_que++] = label_root_id;

        if (!got_candidates[v_tail_local]) {
            // If v_tail_global is not in got_candidates_queue, add it in (prevent duplicate)
            got_candidates[v_tail_local] = true;
            got_candidates_queue[end_got_candidates_queue++] = v_tail_local;
        }
    }
    {// Just for the complain from the compiler
        assert(iter >= iter);
    }
}
// Function: pushes v_head's labels to v_head's every (master) neighbor
template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::local_push_labels(
        VertexID v_head_local,
        VertexID roots_start,
        const DistGraph &G,
        std::vector<ShortIndex> &short_index,
        std::vector<VertexID> &got_candidates_queue,
        VertexID &end_got_candidates_queue,
        std::vector<bool> &got_candidates,
        std::vector<VertexID> &once_candidated_queue,
        VertexID &end_once_candidated_queue,
        std::vector<bool> &once_candidated,
        const std::vector<bool> &used_bp_roots,
        UnweightedDist iter)
{
    // The data structure of a message
//    std::vector< LabelUnitType > buffer_recv;
    const IndexType &Lv = L[v_head_local];
    // These 2 index are used for traversing v_head's last inserted labels
    VertexID l_i_start = Lv.distances.rbegin() -> start_index;
    VertexID l_i_bound = l_i_start + Lv.distances.rbegin() -> size;
    // Traverse v_head's every neighbor v_tail
    VertexID v_head_global = G.get_global_vertex_id(v_head_local);
    EdgeID e_i_start = G.vertices_idx[v_head_global];
    EdgeID e_i_bound = e_i_start + G.local_out_degrees[v_head_global];
    for (EdgeID e_i = e_i_start; e_i < e_i_bound; ++e_i) {
        VertexID v_tail_global = G.out_edges[e_i];
        if (used_bp_roots[v_tail_global]) {
            continue;
        }
        if (v_tail_global < roots_start) { // v_tail_global has higher rank than any roots, then no roots can push new labels to it.
            return;
        }

        // Traverse v_head's last inserted labels
        for (VertexID l_i = l_i_start; l_i < l_i_bound; ++l_i) {
            VertexID label_root_id = Lv.vertices[l_i];
            VertexID label_global_id = label_root_id + roots_start;
            if (v_tail_global <= label_global_id) {
                // v_tail_global has higher rank than the label
                continue;
            }
            VertexID v_tail_local = G.get_local_vertex_id(v_tail_global);
            ShortIndex &SI_v_tail = short_index[v_tail_local];
            if (SI_v_tail.indicator[label_root_id]) {
                // The label is already selected before
                continue;
            }
            // Record label_root_id as once selected by v_tail_global
            SI_v_tail.indicator.set(label_root_id);
            // Add into once_candidated_queue

            if (!once_candidated[v_tail_local]) {
                // If v_tail_global is not in the once_candidated_queue yet, add it in
                once_candidated[v_tail_local] = true;
                once_candidated_queue[end_once_candidated_queue++] = v_tail_local;
            }

//            // Bit Parallel Checking: if label_global_id to v_tail_global has shorter distance already
//            //			++total_check_count;
//            const IndexType &L_label = L[label_global_id];
//
//            _mm_prefetch(&L_label.bp_dist[0], _MM_HINT_T0);
//            _mm_prefetch(&L_label.bp_sets[0][0], _MM_HINT_T0);
////			bp_checking_ins_count.measure_start();
//            bool no_need_add = false;
//            for (VertexID i = 0; i < BITPARALLEL_SIZE; ++i) {
//                VertexID td = L_label.bp_dist[i] + L_tail.bp_dist[i];
//                if (td - 2 <= iter) {
//                    td +=
//                            (L_label.bp_sets[i][0] & L_tail.bp_sets[i][0]) ? -2 :
//                            ((L_label.bp_sets[i][0] & L_tail.bp_sets[i][1]) |
//                             (L_label.bp_sets[i][1] & L_tail.bp_sets[i][0]))
//                            ? -1 : 0;
//                    if (td <= iter) {
//                        no_need_add = true;
////						++bp_hit_count;
//                        break;
//                    }
//                }
//            }
//            if (no_need_add) {
////				bp_checking_ins_count.measure_stop();
//                continue;
//            }
////			bp_checking_ins_count.measure_stop();
            if (SI_v_tail.is_candidate[label_root_id]) {
                continue;
            }
            SI_v_tail.is_candidate[label_root_id] = true;
            SI_v_tail.candidates_que[SI_v_tail.end_candidates_que++] = label_root_id;

            if (!got_candidates[v_tail_local]) {
                // If v_tail_global is not in got_candidates_queue, add it in (prevent duplicate)
                got_candidates[v_tail_local] = true;
                got_candidates_queue[end_got_candidates_queue++] = v_tail_local;
            }
        }
    }

    {
        assert(iter >= iter);
    }
}

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::sync_masters_2_mirrors(
        const DistGraph &G,
        const std::vector<VertexID> &active_queue,
        VertexID end_active_queue,
        std::vector<MPI_Request> &requests_send
)
{
    std::vector< std::pair<VertexID, VertexID> > buffer_send;
        // pair.first: Owener vertex ID of the label
        // pair.first: label vertex ID of the label
    // Prepare masters' newly added labels for sending
    for (VertexID i_q = 0; i_q < end_active_queue; ++i_q) {
        VertexID v_head_local = active_queue[i_q];
        VertexID v_head_global = G.get_global_vertex_id(v_head_local);
        const IndexType &Lv = L[v_head_local];
        // These 2 index are used for traversing v_head's last inserted labels
        VertexID l_i_start = Lv.distances.rbegin()->start_index;
        VertexID l_i_bound = l_i_start + Lv.distances.rbegin()->size;
        for (VertexID l_i = l_i_start; l_i < l_i_bound; ++l_i) {
            VertexID label_root_id = Lv.vertices[l_i];
            buffer_send.emplace_back(v_head_global, label_root_id);
        }
    }

    // Send messages
    for (int loc = 0; loc < num_hosts - 1; ++loc) {
        int dest_host_id = G.buffer_send_list_loc_2_master_host_id(loc);
        MPI_Isend(buffer_send.data(),
                MPI_Instance::get_sending_size(buffer_send),
                MPI_CHAR,
                dest_host_id,
                SENDING_MASTERS_TO_MIRRORS,
                MPI_COMM_WORLD,
                &requests_send[loc]);
    }
}

//// Function: send potential candidates to corresponding hosts from buffer_send_list; receive potential candidates
//// from other host; check those potential candidates and add valid ones to the candidates set.
//// The pair.first is the owner vertex of the potentail candidate.
//// THe pair.second is the label of the potential candidate.
//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::sync_potential_candidates(
//        const std::vector< std::vector< std::pair<VertexID, VertexID> > > &buffer_send_list,
//        const DistGraph &G,
//        std::vector<ShortIndex> &short_index,
//        std::vector<VertexID> &got_candidates_queue,
//        VertexID &end_got_candidates_queue,
//        std::vector<bool> &got_candidates,
//        std::vector<VertexID> &once_candidated_queue,
//        VertexID &end_once_candidated_queue,
//        std::vector<bool> &once_candidated,
//        VertexID roots_start,
//        UnweightedDist iter)
//{
//    // Send the messages to other hosts.
//    std::vector<MPI_Request> requests_send(num_hosts - 1);
//    for (int loc = 0; loc < num_hosts - 1; ++loc) {
//        int dest_host_id = G.buffer_send_list_loc_2_master_host_id(loc);
//        MPI_Isend(buffer_send_list[loc].data(),
//                  MPI_Instance::get_sending_size(buffer_send_list[loc]),
//                  MPI_CHAR,
//                  dest_host_id,
//                  SENDING_PUSHED_LABELS,
//                  MPI_COMM_WORLD,
//                  &requests_send[loc]);
////        printf("@%u host_id: %u send_to: %u size: %lu\n", __LINE__, host_id, dest_host_id, buffer_send_list[loc].size());//test
//    }
////    printf("@%u host_id: %u sent\n", __LINE__, host_id);//test
//
//    // Receive messages from other hosts.
//    std::vector<std::pair<VertexID, VertexID>> buffer_recv;
//    for (int h_i = 0; h_i < num_hosts - 1; ++h_i) {
//        // Receive labels
////        int source =
//        MPI_Instance::receive_dynamic_buffer_from_any(buffer_recv,
//                num_hosts,
//                SENDING_PUSHED_LABELS);
////        printf("@%u host_id: %u recv_from: %u size: %lu\n", __LINE__, host_id, source, buffer_recv.size());//test
//        // Check labels
//        if (buffer_recv.empty()) {
//            continue;
//        }
//        for (const auto &m : buffer_recv) {
//            VertexID v_tail_global = m.first;
//            VertexID label_root_id = m.second;
//            if (check_pushed_labels(
//                    G,
//                    v_tail_global,
//                    label_root_id,
//                    roots_start,
//                    short_index,
//                    once_candidated_queue,
//                    end_once_candidated_queue,
//                    once_candidated,
//                    iter)) {
//                // Add into got_candidates_queue
//                VertexID v_tail_local = G.get_local_vertex_id(v_tail_global);
//                if (!got_candidates[v_tail_local]) {
//                    // If v_tail_global is not in got_candidates_queue, add it in (prevent duplicate)
//                    got_candidates[v_tail_local] = true;
//                    got_candidates_queue[end_got_candidates_queue++] = v_tail_local;
//                }
//            }
//        }
//    }
////    printf("@%u host_id: %u received\n", __LINE__, host_id);//test
//    MPI_Waitall(num_hosts - 1,
//                requests_send.data(),
//                MPI_STATUSES_IGNORE);
//}


// Function for distance query;
// traverse vertex v_id's labels;
// return false if shorter distance exists already, return true if the cand_root_id can be added into v_id's label.
template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline bool DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::distance_query(
        VertexID cand_root_id,
        VertexID v_id_local,
        VertexID roots_start,
//        const std::vector<IndexType> &L,
        const std::vector< std::vector<UnweightedDist> > &dist_table,
        UnweightedDist iter)
{
//	++total_check_count;
//	++normal_check_count;
//	distance_query_time -= WallTimer::get_time_mark();
//	dist_query_ins_count.measure_start();

    VertexID cand_real_id = cand_root_id + roots_start;
    const IndexType &Lv = L[v_id_local];

    // Traverse v_id's all existing labels
    VertexID b_i_bound = Lv.batches.size();
    _mm_prefetch(&Lv.batches[0], _MM_HINT_T0);
    _mm_prefetch(&Lv.distances[0], _MM_HINT_T0);
    _mm_prefetch(&Lv.vertices[0], _MM_HINT_T0);
    //_mm_prefetch(&dist_table[cand_root_id][0], _MM_HINT_T0);
    for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
        VertexID id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
        VertexID dist_start_index = Lv.batches[b_i].start_index;
        VertexID dist_bound_index = dist_start_index + Lv.batches[b_i].size;
        // Traverse dist_table
        for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
            UnweightedDist dist = Lv.distances[dist_i].dist;
            if (dist >= iter) { // In a batch, the labels' distances are increasingly ordered.
                // If the half path distance is already greater than their targeted distance, jump to next batch
                break;
            }
            VertexID v_start_index = Lv.distances[dist_i].start_index;
            VertexID v_bound_index = v_start_index + Lv.distances[dist_i].size;
//            _mm_prefetch(&dist_table[cand_root_id][0], _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(dist_table[cand_root_id].data()), _MM_HINT_T0);
            for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                VertexID v = Lv.vertices[v_i] + id_offset; // v is a label hub of v_id
                if (v >= cand_real_id) {
                    // Vertex cand_real_id cannot have labels whose ranks are lower than it,
                    // in which case dist_table[cand_root_id][v] does not exist.
                    continue;
                }
                VertexID d_tmp = dist + dist_table[cand_root_id][v];
//                {//test
//                    if ((3 == iter && 17 == v_id_local && 7 == cand_root_id && 1 == host_id)
//                        || (3 == iter && 41 == v_id_local && 7 == cand_root_id && 0 == host_id)) {
//                        printf("DISTANCE_QUERY label(%u, %u) dist_table[%u][%u]: %u d_tmp: %u iter: %u\n", v, dist,
//                               cand_root_id, v, dist_table[cand_root_id][v], d_tmp, iter);//test
//                    }
//                }
                if (d_tmp <= iter) {
//					distance_query_time += WallTimer::get_time_mark();
//					dist_query_ins_count.measure_stop();
                    return false;
                }
            }
        }
    }
//	distance_query_time += WallTimer::get_time_mark();
//	dist_query_ins_count.measure_stop();
    return true;
}

// Function inserts candidate cand_root_id into vertex v_id's labels;
// update the distance buffer dist_table;
// but it only update the v_id's labels' vertices array;
template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::insert_label_only(
        VertexID cand_root_id,
        VertexID v_id_local,
        VertexID roots_start,
        VertexID roots_size,
//        std::vector<IndexType> &L,
        const DistGraph &G,
        std::vector< std::vector<UnweightedDist> > &dist_table,
        std::vector< std::pair<VertexID, VertexID> > &buffer_send,
        UnweightedDist iter)
{
    L[v_id_local].vertices.push_back(cand_root_id);
    // Update the distance buffer if v_id is a root
    VertexID v_id_global = G.get_global_vertex_id(v_id_local);
    VertexID v_root_id = v_id_global - roots_start;
    if (v_id_global >= roots_start && v_root_id < roots_size) {
        VertexID cand_real_id = cand_root_id + roots_start;
        dist_table[v_root_id][cand_real_id] = iter;
        // Put the update into the buffer_send for later sending
        buffer_send.emplace_back(v_root_id, cand_real_id);
    }
}

// Function updates those index arrays in v_id's label only if v_id has been inserted new labels
template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::update_label_indices(
        VertexID v_id_local,
        VertexID inserted_count,
//        std::vector<IndexType> &L,
        std::vector<ShortIndex> &short_index,
        VertexID b_id,
        UnweightedDist iter)
{
    IndexType &Lv = L[v_id_local];
    // indicator[BATCH_SIZE + 1] is true, means v got some labels already in this batch
    if (short_index[v_id_local].indicator[BATCH_SIZE]) {
        // Increase the batches' last element's size because a new distance element need to be added
        ++(Lv.batches.rbegin() -> size);
    } else {
        short_index[v_id_local].indicator.set(BATCH_SIZE);
        // Insert a new Batch with batch_id, start_index, and size because a new distance element need to be added
        Lv.batches.emplace_back(
                b_id, // batch id
                Lv.distances.size(), // start index
                1); // size
    }
    // Insert a new distance element with start_index, size, and dist
    Lv.distances.emplace_back(
            Lv.vertices.size() - inserted_count, // start index
            inserted_count, // size
            iter); // distance
}

// Function to reset dist_table the distance buffer to INF
// Traverse every root's labels to reset its distance buffer elements to INF.
// In this way to reduce the cost of initialization of the next batch.
template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::reset_at_end(
        const DistGraph &G,
        VertexID roots_start,
//        VertexID roots_size,
        const std::vector<VertexID> &roots_master_local,
        std::vector< std::vector<UnweightedDist> > &dist_table,
        std::vector< std::vector<VertexID> > &recved_dist_table)
{
//    VertexID b_i_bound;
//    VertexID id_offset;
//    VertexID dist_start_index;
//    VertexID dist_bound_index;
//    VertexID v_start_index;
//    VertexID v_bound_index;
//    for (VertexID r_id = 0; r_id < roots_size; ++r_id) {
    // Reset dist_table according to local masters' labels
    for (VertexID r_local_id : roots_master_local) {
        IndexType &Lr = L[r_local_id];
        VertexID r_root_id = G.get_global_vertex_id(r_local_id) - roots_start;
        VertexID b_i_bound = Lr.batches.size();
        _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
        _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
        _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
        for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
            VertexID id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
            VertexID dist_start_index = Lr.batches[b_i].start_index;
            VertexID dist_bound_index = dist_start_index + Lr.batches[b_i].size;
            // Traverse dist_table
            for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                VertexID v_start_index = Lr.distances[dist_i].start_index;
                VertexID v_bound_index = v_start_index + Lr.distances[dist_i].size;
                for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                    dist_table[r_root_id][Lr.vertices[v_i] + id_offset] = MAX_UNWEIGHTED_DIST;
                }
            }
        }

        // Cleanup for large graphs. 02/17/2019
//		Lr.cleanup();
    }
//    }
    // Reset dist_table according to received masters' labels from other hosts
    for (VertexID r_root_id = 0; r_root_id < BATCH_SIZE; ++r_root_id) {
        for (VertexID cand_real_id : recved_dist_table[r_root_id]) {
            dist_table[r_root_id][cand_real_id] = MAX_UNWEIGHTED_DIST;
        }
        recved_dist_table[r_root_id].clear();
    }
}

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::batch_process(
        const DistGraph &G,
        VertexID b_id,
        VertexID roots_start, // start id of roots
        VertexID roots_size, // how many roots in the batch
//        std::vector<IndexType> &L,
        const std::vector<bool> &used_bp_roots,
        std::vector<VertexID> &active_queue,
        VertexID &end_active_queue,
        std::vector<VertexID> &got_candidates_queue,
        VertexID &end_got_candidates_queue,
        std::vector<ShortIndex> &short_index,
        std::vector< std::vector<UnweightedDist> > &dist_table,
        std::vector< std::vector<VertexID> > &recved_dist_table,
        std::vector<bool> &got_candidates,
        std::vector<bool> &is_active,
        std::vector<VertexID> &once_candidated_queue,
        VertexID &end_once_candidated_queue,
        std::vector<bool> &once_candidated)
{
//	initializing_time -= WallTimer::get_time_mark();
    // At the beginning of a batch, initialize the labels L and distance buffer dist_table;
//	puts("Initializing...");//test
    std::vector<VertexID> roots_master_local; // Roots which belongs to this host.
    VertexID global_num_actives = initialization(G,
                                    short_index,
                                    dist_table,
                                    recved_dist_table,
                                    active_queue,
                                    end_active_queue,
                                    once_candidated_queue,
                                    end_once_candidated_queue,
                                    once_candidated,
                                    b_id,
                                    roots_start,
                                    roots_size,
                                    roots_master_local,
                                    used_bp_roots);
//	puts("Initial done.");//test

    UnweightedDist iter = 0; // The iterator, also the distance for current iteration
//	initializing_time += WallTimer::get_time_mark();


//    std::vector< std::vector<VertexID> > recved_dist_table(BATCH_SIZE);
    while (global_num_actives) {
//		candidating_time -= WallTimer::get_time_mark();
//		candidating_ins_count.measure_start();
        ++iter;
//		printf("iter: %u\n", iter);//test
        // Traverse active vertices to push their labels as candidates
//		puts("Pushing...");//test

        // Push newly added labels to local masters at first
        for (VertexID i_queue = 0; i_queue < end_active_queue; ++i_queue) {
            VertexID v_head_local = active_queue[i_queue];
            is_active[v_head_local] = false; // reset is_active
            if (!G.local_out_degrees[G.get_global_vertex_id(v_head_local)]) {
                continue;
            }
            local_push_labels(
                    v_head_local,
                    roots_start,
                    G,
                    short_index,
                    got_candidates_queue,
                    end_got_candidates_queue,
                    got_candidates,
                    once_candidated_queue,
                    end_once_candidated_queue,
                    once_candidated,
                    used_bp_roots,
                    iter);
        }
//        { // test
//            printf("@%u host_id: %u end_got_candidates_queue: %u\n", __LINE__, host_id, end_got_candidates_queue);//test
//            std::sort(got_candidates_queue.begin(), got_candidates_queue.begin() + end_got_candidates_queue);
//            for (VertexID i = 0; i < end_got_candidates_queue; ++i) {
//                printf("@%u host_id: %u got_candidates_queue[%u]: %u\n", __LINE__, host_id, i, got_candidates_queue[i]);
//            }
//        }

        // Send masters' newly added labels to other hosts
        std::vector<MPI_Request> requests_send(num_hosts - 1);
        sync_masters_2_mirrors(G,
                active_queue,
                end_active_queue,
                requests_send);
        // Receive messages from other hosts
        std::vector< std::pair<VertexID, VertexID> > buffer_recv;
        for (int loc = 0; loc < num_hosts - 1; ++loc) {
            MPI_Instance::receive_dynamic_buffer_from_any(buffer_recv,
                    num_hosts,
                    SENDING_MASTERS_TO_MIRRORS);
            if (buffer_recv.empty()) {
                continue;
            }
            for (const auto &m : buffer_recv) {
                VertexID v_head_global = m.first;
                if (!G.local_out_degrees[v_head_global]) {
                    continue;
                }
                VertexID label_root_id = m.second;
                push_label(
                        v_head_global,
                        label_root_id,
                        roots_start,
                        G,
                        short_index,
                        got_candidates_queue,
                        end_got_candidates_queue,
                        got_candidates,
                        once_candidated_queue,
                        end_once_candidated_queue,
                        once_candidated,
                        used_bp_roots,
                        iter);
            }
        }
        end_active_queue = 0;
        MPI_Waitall(num_hosts - 1,
                requests_send.data(),
                MPI_STATUSES_IGNORE);
        {// test
            VertexID global_end_got_candidates_queue;
            MPI_Allreduce(&end_got_candidates_queue,
                    &global_end_got_candidates_queue,
                    1,
                    V_ID_Type,
                    MPI_SUM,
                    MPI_COMM_WORLD);
            if (0 == host_id) {
                printf("iter %u @%u host_id: %u global_end_got_candidates_queue: %u\n", iter, __LINE__, host_id, global_end_got_candidates_queue);
            }
//            std::sort(got_candidates_queue.begin(), got_candidates_queue.begin() + end_got_candidates_queue);
//            for (VertexID i = 0; i < end_got_candidates_queue; ++i) {
//                printf("@%u host_id: %u got_candidates_queue[%u]: %u\n", __LINE__, host_id, i, G.get_global_vertex_id(got_candidates_queue[i]));
//            }
//            exit(EXIT_SUCCESS);
        }

///////////////////////////////////////////
//        std::vector< std::vector< std::pair<VertexID, VertexID> > > buffer_send_list(num_hosts - 1);
//        for (VertexID i_queue = 0; i_queue < end_active_queue; ++i_queue) {
//            VertexID v_head_local = active_queue[i_queue];
//            is_active[v_head_local] = false; // reset is_active
//
//            push_labels(
//                    v_head_local,
//                    roots_start,
//                    G,
////                    L,
//                    short_index,
//                    got_candidates_queue,
//                    end_got_candidates_queue,
//                    got_candidates,
//                    once_candidated_queue,
//                    end_once_candidated_queue,
//                    once_candidated,
//                    used_bp_roots,
//                    buffer_send_list,
//                    iter);
//        }
//        end_active_queue = 0; // Set the active_queue empty
//        // Send and Recieve and Check potential candidates
//        sync_potential_candidates(
//                buffer_send_list,
//                G,
//                short_index,
//                got_candidates_queue,
//                end_got_candidates_queue,
//                got_candidates,
//                once_candidated_queue,
//                end_once_candidated_queue,
//                once_candidated,
//                roots_start,
//                iter);
//
//        puts("Push done.");//test
//		candidating_ins_count.measure_stop();
//		candidating_time += WallTimer::get_time_mark();
//		adding_time -= WallTimer::get_time_mark();
//		adding_ins_count.measure_start();

        // Traverse vertices in the got_candidates_queue to insert labels
//		puts("Checking...");//test
        std::vector< std::pair<VertexID, VertexID> > buffer_send; // For sync elements in the dist_table
            // pair.first: root id
            // pair.second: label (global) id of the root
        for (VertexID i_queue = 0; i_queue < end_got_candidates_queue; ++i_queue) {
            VertexID v_id_local = got_candidates_queue[i_queue];
//            {//test
//                if (1 == b_id && 3 == iter && 41 == G.get_global_vertex_id(v_id_local)) {
//                    printf("Yes, host_id %u got v_id %u (local_id %u)\n", host_id, G.get_global_vertex_id(v_id_local), v_id_local);//test
//                }
//            }
            VertexID inserted_count = 0; //recording number of v_id's truly inserted candidates
            got_candidates[v_id_local] = false; // reset got_candidates
            // Traverse v_id's all candidates
            VertexID bound_cand_i = short_index[v_id_local].end_candidates_que;
            for (VertexID cand_i = 0; cand_i < bound_cand_i; ++cand_i) {
                VertexID cand_root_id = short_index[v_id_local].candidates_que[cand_i];
                short_index[v_id_local].is_candidate[cand_root_id] = false;
                // Only insert cand_root_id into v_id's label if its distance to v_id is shorter than existing distance
                if ( distance_query(
                        cand_root_id,
                        v_id_local,
                        roots_start,
//                        L,
                        dist_table,
                        iter) ) {
                    if (!is_active[v_id_local]) {
                        is_active[v_id_local] = true;
                        active_queue[end_active_queue++] = v_id_local;
//                        {//test
//                            if (1 == b_id && 3 == iter && 41 == G.get_global_vertex_id(v_id_local)) {
//                                printf("??? host_id %u v_id %u cand_root_id %u\n", host_id, G.get_global_vertex_id(v_id_local), cand_root_id);
//                            }
//                        }
//                        {//test
//                            if (2 == iter && 1 == G.get_global_vertex_id(v_id_local) && 0 == cand_root_id) {
//                                printf("@@@ host_id %u v_id %u got label (%u,%u)\n", host_id, 1, cand_root_id, 2);//test
//                            }
//                        }
                    }
                    ++inserted_count;
                    // The candidate cand_root_id needs to be added into v_id's label
                    insert_label_only(
                            cand_root_id,
                            v_id_local,
                            roots_start,
                            roots_size,
//                            L,
                            G,
                            dist_table,
                            buffer_send,
                            iter);
                }
            }
            short_index[v_id_local].end_candidates_que = 0;
            if (0 != inserted_count) {
                // Update other arrays in L[v_id] if new labels were inserted in this iteration
                update_label_indices(
                        v_id_local,
                        inserted_count,
//                        L,
                        short_index,
                        b_id,
                        iter);
            }
        }
        end_got_candidates_queue = 0; // Set the got_candidates_queue empty
//		puts("Check done.");//test
//		adding_ins_count.measure_stop();
//		adding_time += WallTimer::get_time_mark();

        // Sync the dist_table
        for (int loc = 0; loc < num_hosts - 1; ++loc) {
            // Send updated elements (' coordinates) in dist_table
            VertexID dest_host_id = G.buffer_send_list_loc_2_master_host_id(loc);
            MPI_Isend(buffer_send.data(),
                    MPI_Instance::get_sending_size(buffer_send),
                    MPI_CHAR,
                    dest_host_id,
                    SYNC_DIST_TABLE,
                    MPI_COMM_WORLD,
                    &requests_send[loc]);
        }
        for (int loc = 0; loc < num_hosts - 1; ++loc) {
            // Receive roots' new labels from other hosts
            MPI_Instance::receive_dynamic_buffer_from_any(buffer_recv,
                    num_hosts,
                    SYNC_DIST_TABLE);
            if (buffer_recv.empty()) {
                continue;
            }
            for (const auto &e : buffer_recv) {
                VertexID root_id = e.first;
                VertexID cand_real_id = e.second;
                dist_table[root_id][cand_real_id] = iter;
                // Record the received element, for future reset
                recved_dist_table[root_id].push_back(cand_real_id);
            }
        }
        MPI_Waitall(num_hosts - 1,
                requests_send.data(),
                MPI_STATUSES_IGNORE);

        // Sync the global_num_actives
        MPI_Allreduce(&end_active_queue,
                &global_num_actives,
                1,
                V_ID_Type,
                MPI_SUM,
                MPI_COMM_WORLD);
        {// test
//            if (1 == b_id && 3 == iter) {
            if (0 == host_id) {
                printf("iter: %u @%u host_id: %u global_num_actives: %u\n", iter, __LINE__, host_id, global_num_actives);//test
//                std::sort(active_queue.begin(), active_queue.begin() + end_active_queue);
//                for (VertexID i = 0; i < end_active_queue; ++i) {
//                    printf("iter: %u @%u host_id: %u active_queue[%u]: %u\n", iter, __LINE__, host_id, i, G.get_global_vertex_id(active_queue[i]));
//                }
//                exit(EXIT_SUCCESS);
            }
//            }

        }
    }

    // Reset the dist_table
//	initializing_time -= WallTimer::get_time_mark();
//	init_dist_matrix_time -= WallTimer::get_time_mark();
//	puts("Resetting...");//test
    reset_at_end(
            G,
            roots_start,
//            roots_size,
            roots_master_local,
            dist_table,
            recved_dist_table);
//	puts("Reset done.");//test
//	init_dist_matrix_time += WallTimer::get_time_mark();
//	initializing_time += WallTimer::get_time_mark();


//	double total_time = time_can + time_add;
//	printf("Candidating time: %f (%f%%)\n", time_can, time_can / total_time * 100);
//	printf("Adding time: %f (%f%%)\n", time_add, time_add / total_time * 100);
}


//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::construct(const DistGraph &G)
//{
//    num_v = G.num_v;
//    assert(num_v >= BATCH_SIZE);
//    num_masters = G.num_masters;
//    host_id = G.host_id;
//    num_hosts = G.num_hosts;
//    V_ID_Type = G.V_ID_Type;
////    L.resize(num_v);
//    L.resize(num_masters);
//    VertexID remainer = num_v % BATCH_SIZE;
//    VertexID b_i_bound = num_v / BATCH_SIZE;
//    std::vector<bool> used_bp_roots(num_v, false);
//    //cache_miss.measure_start();
//    double time_labeling = -WallTimer::get_time_mark();
//
////	double bp_labeling_time = -WallTimer::get_time_mark();
////	bp_labeling_ins_count.measure_start();
////    bit_parallel_labeling(
////            G,
//////            L,
////            used_bp_roots);
////	bp_labeling_ins_count.measure_stop();
////	bp_labeling_time += WallTimer::get_time_mark();
//
//    std::vector<VertexID> active_queue(num_masters); // Any vertex v who is active should be put into this queue.
////    std::vector<VertexID> active_queue(num_v); // Any vertex v who is active should be put into this queue.
//    VertexID end_active_queue = 0;
//    std::vector<bool> is_active(num_masters, false);// is_active[v] is true means vertex v is in the active queue.
////    std::vector<bool> is_active(num_v, false);// is_active[v] is true means vertex v is in the active queue.
//    std::vector<VertexID> got_candidates_queue(num_masters); // Any vertex v who got candidates should be put into this queue.
////    std::vector<VertexID> got_candidates_queue(num_v); // Any vertex v who got candidates should be put into this queue.
//    VertexID end_got_candidates_queue = 0;
//    std::vector<bool> got_candidates(num_masters, false); // got_candidates[v] is true means vertex v is in the queue got_candidates_queue
////    std::vector<bool> got_candidates(num_v, false); // got_candidates[v] is true means vertex v is in the queue got_candidates_queue
//    std::vector<ShortIndex> short_index(num_masters);
////    std::vector<ShortIndex> short_index(num_v);
//    std::vector< std::vector<UnweightedDist> > dist_table(BATCH_SIZE, std::vector<UnweightedDist>(num_v, MAX_UNWEIGHTED_DIST));
//
//
//    std::vector<VertexID> once_candidated_queue(num_masters); // if short_index[v].indicator.any() is true, v is in the queue.
////    std::vector<VertexID> once_candidated_queue(num_v); // if short_index[v].indicator.any() is true, v is in the queue.
//        // Used mainly for resetting short_index[v].indicator.
//    VertexID end_once_candidated_queue = 0;
//    std::vector<bool> once_candidated(num_masters, false);
////    std::vector<bool> once_candidated(num_v, false);
//
//    //printf("b_i_bound: %u\n", b_i_bound);//test
//    for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
//        printf("b_i: %u\n", b_i);//test
//        batch_process(
//                G,
//                b_i,
//                b_i * BATCH_SIZE,
//                BATCH_SIZE,
////                L,
//                used_bp_roots,
//                active_queue,
//                end_active_queue,
//                got_candidates_queue,
//                end_got_candidates_queue,
//                short_index,
//                dist_table,
//                got_candidates,
//                is_active,
//                once_candidated_queue,
//                end_once_candidated_queue,
//                once_candidated);
//    }
//    if (remainer != 0) {
//		printf("b_i: %u\n", b_i_bound);//test
//        batch_process(
//                G,
//                b_i_bound,
//                b_i_bound * BATCH_SIZE,
//                remainer,
////                L,
//                used_bp_roots,
//                active_queue,
//                end_active_queue,
//                got_candidates_queue,
//                end_got_candidates_queue,
//                short_index,
//                dist_table,
//                got_candidates,
//                is_active,
//                once_candidated_queue,
//                end_once_candidated_queue,
//                once_candidated);
//    }
//    time_labeling += WallTimer::get_time_mark();
//    //cache_miss.measure_stop();
//
//    // Test
//    setlocale(LC_NUMERIC, "");
//    printf("BATCH_SIZE: %u\n", BATCH_SIZE);
//    printf("BP_Size: %u\n", BITPARALLEL_SIZE);
////	printf("BP_labeling: %f %.2f%%\n", bp_labeling_time, bp_labeling_time / time_labeling * 100);
////	printf("Initializing: %f %.2f%%\n", initializing_time, initializing_time / time_labeling * 100);
////		printf("\tinit_start_reset_time: %f (%f%%)\n", init_start_reset_time, init_start_reset_time / initializing_time * 100);
////		printf("\tinit_index_time: %f (%f%%)\n", init_index_time, init_index_time / initializing_time * 100);
////			printf("\t\tinit_indicators_time: %f (%f%%)\n", init_indicators_time, init_indicators_time / init_index_time * 100);
////		printf("\tinit_dist_matrix_time: %f (%f%%)\n", init_dist_matrix_time, init_dist_matrix_time / initializing_time * 100);
////	printf("Candidating: %f %.2f%%\n", candidating_time, candidating_time / time_labeling * 100);
////	printf("Adding: %f %.2f%%\n", adding_time, adding_time / time_labeling * 100);
////		printf("distance_query_time: %f %.2f%%\n", distance_query_time, distance_query_time / time_labeling * 100);
////		uint64_t total_check_count = bp_hit_count + normal_check_count;
////		printf("total_check_count: %'llu\n", total_check_count);
////		printf("bp_hit_count: %'llu %.2f%%\n",
////						bp_hit_count,
////						bp_hit_count * 100.0 / total_check_count);
////		printf("normal_check_count: %'llu %.2f%%\n", normal_check_count, normal_check_count * 100.0 / total_check_count);
////		printf("total_candidates_num: %'llu set_candidates_num: %'llu %.2f%%\n",
////							total_candidates_num,
////							set_candidates_num,
////							set_candidates_num * 100.0 / total_candidates_num);
////		printf("\tnormal_hit_count (to total_check, to normal_check): %llu (%f%%, %f%%)\n",
////						normal_hit_count,
////						normal_hit_count * 100.0 / total_check_count,
////						normal_hit_count * 100.0 / (total_check_count - bp_hit_count));
//    //cache_miss.print();
////	printf("Candidating: "); candidating_ins_count.print();
////	printf("Adding: "); adding_ins_count.print();
////	printf("BP_Labeling: "); bp_labeling_ins_count.print();
////	printf("BP_Checking: "); bp_checking_ins_count.print();
////	printf("distance_query: "); dist_query_ins_count.print();
//
//    printf("Total_labeling_time: %.2f seconds\n", time_labeling);
//    // End test
//}

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::store_index_to_file(
        const char *filename,
        const std::vector<VertexID> &rank)
{
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    // Store into file the number of vertices and the number of bit-parallel roots.
    uint64_t labels_count = 0;
    VertexID num_bp_roots = BITPARALLEL_SIZE;
    fout.write(reinterpret_cast<char *>(&num_v), sizeof(num_v));
    fout.write(reinterpret_cast<char *>(&num_bp_roots), sizeof(num_bp_roots));
    for (VertexID v_id = 0; v_id < num_v; ++v_id) {
        VertexID v_rank = rank[v_id];
        const IndexType &Lv = L[v_rank];
        VertexID size_labels = Lv.vertices.size();
        labels_count += size_labels;
        // Store Bit-parallel Labels into file.
        for (VertexID b_i = 0; b_i < BITPARALLEL_SIZE; ++b_i) {
            UnweightedDist d = Lv.bp_dist[b_i];
            uint64_t s0 = Lv.bp_sets[b_i][0];
            uint64_t s1 = Lv.bp_sets[b_i][1];
            fout.write(reinterpret_cast<char *>(&d), sizeof(d));
            fout.write(reinterpret_cast<char *>(&s0), sizeof(s0));
            fout.write(reinterpret_cast<char *>(&s1), sizeof(s1));
        }

        std::vector< std::pair<VertexID, UnweightedDist> > ordered_labels;
        // Traverse v_id's all existing labels
        for (VertexID b_i = 0; b_i < Lv.batches.size(); ++b_i) {
            VertexID id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
            VertexID dist_start_index = Lv.batches[b_i].start_index;
            VertexID dist_bound_index = dist_start_index + Lv.batches[b_i].size;
            // Traverse dist_table
            for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                VertexID v_start_index = Lv.distances[dist_i].start_index;
                VertexID v_bound_index = v_start_index + Lv.distances[dist_i].size;
                UnweightedDist dist = Lv.distances[dist_i].dist;
                for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                    VertexID tail = Lv.vertices[v_i] + id_offset;
                    ordered_labels.emplace_back(tail, dist);
//                    ordered_labels.push_back(make_pair(tail, dist));
                }
            }
        }
        // Sort
        sort(ordered_labels.begin(), ordered_labels.end());
        // Store into file
        fout.write(reinterpret_cast<char *>(&size_labels), sizeof(size_labels));
        for (VertexID l_i = 0; l_i < size_labels; ++l_i) {
            VertexID l = ordered_labels[l_i].first;
            UnweightedDist d = ordered_labels[l_i].second;
            fout.write(reinterpret_cast<char *>(&l), sizeof(l));
            fout.write(reinterpret_cast<char *>(&d), sizeof(d));
        }
    }

    printf("Label_size: %'lu mean: %f\n", labels_count, static_cast<double>(labels_count) / num_v);
    fout.close();
}

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::load_index_from_file(const char *filename)
{
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
//    VertexID num_v;
    // Load from file the number of vertices and the number of bit-parallel roots.
    VertexID num_bp_roots;
    fin.read(reinterpret_cast<char *>(&num_v), sizeof(num_v));
    fin.read(reinterpret_cast<char *>(&num_bp_roots), sizeof(num_bp_roots));
//    num_v_ = num_v;
    Index.resize(num_v);
    uint64_t labels_count = 0;
    // Load labels for every vertex
    for (VertexID v_id = 0; v_id < num_v; ++v_id) {
        IndexOrdered &Iv = Index[v_id];
        // Load Bit-parallel Labels from file.
        for (VertexID b_i = 0; b_i < num_bp_roots; ++b_i) {
            fin.read(reinterpret_cast<char *>(&Iv.bp_dist[b_i]), sizeof(Iv.bp_dist[b_i]));
            fin.read(reinterpret_cast<char *>(&Iv.bp_sets[b_i][0]), sizeof(Iv.bp_sets[b_i][0]));
            fin.read(reinterpret_cast<char *>(&Iv.bp_sets[b_i][1]), sizeof(Iv.bp_sets[b_i][1]));
        }

        // Normal Labels
        // Load Labels from file.
        VertexID size_labels;
        fin.read(reinterpret_cast<char *>(&size_labels), sizeof(size_labels));
        labels_count += size_labels;
        Iv.label_id.resize(size_labels + 1);
        Iv.label_dists.resize(size_labels + 1);
        for (VertexID l_i = 0; l_i < size_labels; ++l_i) {
            fin.read(reinterpret_cast<char *>(&Iv.label_id[l_i]), sizeof(Iv.label_id[l_i]));
            fin.read(reinterpret_cast<char *>(&Iv.label_dists[l_i]), sizeof(Iv.label_dists[l_i]));
        }
        Iv.label_id[size_labels] = num_v; // Sentinel
        Iv.label_dists[size_labels] = (UnweightedDist) -1; // Sentinel
    }
    printf("Label_size_loaded: %'lu mean: %f\n", labels_count, static_cast<double>(labels_count) / num_v);
    fin.close();
}


template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::order_labels(
        const std::vector<VertexID> &rank2id)
//        const std::vector<VertexID> &rank)
{
//    VertexID num_v = rank.size();
    std::vector< std::vector< std::pair<VertexID, UnweightedDist> > > ordered_L(num_v);
    VertexID labels_count = 0;
    Index.resize(num_v);

    // Traverse the L, put them into Index (ordered labels)
    for (VertexID v_id = 0; v_id < num_v; ++v_id) {
        VertexID new_v = rank2id[v_id];
        IndexOrdered & Iv = Index[new_v];
        const IndexType &Lv = L[v_id];
        auto &OLv = ordered_L[new_v];
        // Bit-parallel Labels
        memcpy(&Iv.bp_dist, &Lv.bp_dist, BITPARALLEL_SIZE * sizeof(UnweightedDist));
        for (VertexID b_i = 0; b_i < BITPARALLEL_SIZE; ++b_i) {
            memcpy(&Iv.bp_sets[b_i], &Lv.bp_sets[b_i], 2 * sizeof(uint64_t));
        }

        // Normal Labels
        // Traverse v_id's all existing labels
        for (VertexID b_i = 0; b_i < Lv.batches.size(); ++b_i) {
            VertexID id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
            VertexID dist_start_index = Lv.batches[b_i].start_index;
            VertexID dist_bound_index = dist_start_index + Lv.batches[b_i].size;
            // Traverse dist_table
            for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                VertexID v_start_index = Lv.distances[dist_i].start_index;
                VertexID v_bound_index = v_start_index + Lv.distances[dist_i].size;
                UnweightedDist dist = Lv.distances[dist_i].dist;
                for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                    VertexID tail = Lv.vertices[v_i] + id_offset;
//					VertexID new_tail = rank2id[tail];
//					new_L[new_v].push_back(make_pair(new_tail, dist));
//                    OLv.push_back(make_pair(tail, dist));
                    OLv.emplace_back(tail, dist);
                }
            }
        }
        // Sort
        sort(OLv.begin(), OLv.end());
        // Store into Index
        VertexID size_labels = OLv.size();
        labels_count += size_labels;
        Iv.label_id.resize(size_labels + 1); // Adding one for Sentinel
        Iv.label_dists.resize(size_labels + 1); // Adding one for Sentinel
        for (VertexID l_i = 0; l_i < size_labels; ++l_i) {
            Iv.label_id[l_i] = OLv[l_i].first;
            Iv.label_dists[l_i] = OLv[l_i].second;
        }
        Iv.label_id[size_labels] = num_v; // Sentinel
        Iv.label_dists[size_labels] = MAX_UNWEIGHTED_DIST; // Sentinel
    }
    printf("Label_size: %u mean: %f\n", labels_count, static_cast<double>(labels_count) / num_v);
//	// Test
//	{
//		puts("Asserting...");
//		for (VertexID v_id = 0; v_id < num_v; ++v_id) {
//			const IndexType &Lv = L[v_id];
//			const IndexOrdered &Iv = Index[rank2id[v_id]];
//			// Bit-parallel Labels
//			for (VertexID b_i = 0; b_i < BITPARALLEL_SIZE; ++b_i) {
//				assert(Lv.bp_dist[b_i] == Iv.bp_dist[b_i]);
//				assert(Lv.bp_sets[b_i][0] == Iv.bp_sets[b_i][0]);
//				assert(Lv.bp_sets[b_i][1] == Iv.bp_sets[b_i][1]);
//			}
//			// Normal Labels
//			assert(Lv.vertices.size() == Iv.label_id.size());
//			assert(Lv.vertices.size() == Iv.label_dists.size());
////			{
////				VertexID bound_i = Iv.label_id.size() > 10 ? 10 : Iv.label_id.size();
////				printf("V %u:", rank2id[v_id]);
////				for (VertexID i = 0; i < bound_i; ++i) {
////					printf(" (%u, %u)", Iv.label_id[i], Iv.label_dists[i]);
////				}
////				puts("");
////			}
//
//		}
//		puts("Asserted.");
//	}
}

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
UnweightedDist DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::query_distance(
        VertexID a,
        VertexID b)
{
//    VertexID num_v = num_v_;
    if (a >= num_v || b >= num_v) {
        return a == b ? 0 : MAX_UNWEIGHTED_DIST;
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
    UnweightedDist d = MAX_UNWEIGHTED_DIST;

    _mm_prefetch(&Ia.label_id[0], _MM_HINT_T0);
    _mm_prefetch(&Ib.label_id[0], _MM_HINT_T0);
    _mm_prefetch(&Ia.label_dists[0], _MM_HINT_T0);
    _mm_prefetch(&Ib.label_dists[0], _MM_HINT_T0);

    // Bit-Parallel Labels
    for (VertexID i = 0; i < BITPARALLEL_SIZE; ++i) {
        VertexID td = Ia.bp_dist[i] + Ib.bp_dist[i];
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
//	std::vector<VertexID> &A = Ia.label_id;
//	std::vector<VertexID> &B = Ib.label_id;
//	VertexID len_B = B.size() - 1;
////	VertexID len_B = B.size();
//	VertexID bound_b_base_i = len_B - (len_B % NUM_P_INT);
//	VertexID a_i = 0;
//	VertexID b_base_i = 0;
//	VertexID len_A = A.size() - 1;
////	VertexID len_A = A.size();
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
//			VertexID td = Ia.label_dists[a_i] + Ib.label_dists[b_base_i + (VertexID) (log2(is_equal_m))];
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
//			VertexID td = Ia.label_dists[a_i] + Ib.label_dists[b_base_i];
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
    for (VertexID i1 = 0, i2 = 0; ; ) {
        VertexID v1 = Ia.label_id[i1], v2 = Ib.label_id[i2];
        if (v1 == v2) {
            if (v1 == num_v) {
                break;  // Sentinel
            }
            VertexID td = Ia.label_dists[i1] + Ib.label_dists[i2];
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

    if (d >= MAX_UNWEIGHTED_DIST - 2) {
        d = MAX_UNWEIGHTED_DIST;
    }
    return d;
}

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::test_queries_normal_index(
        std::vector< std::vector< std::pair<VertexID, UnweightedDist> > > new_L)
{
    // Try query
    VertexID u;
    VertexID v;
    while (std::cin >> u >> v) {
        assert(u < num_v && v < num_v);
        UnweightedDist dist = MAX_UNWEIGHTED_DIST;
//		// Bit Parallel Check
//		const IndexType &idx_u = L[rank[u]];
//		const IndexType &idx_v = L[rank[v]];
//
//		for (VertexID i = 0; i < BITPARALLEL_SIZE; ++i) {
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

        // Normal Index Check
        const auto &Lu = new_L[u];
        const auto &Lv = new_L[v];
//		unsorted_map<VertexID, UnweightedDist> markers;
        std::map<VertexID, UnweightedDist> markers;
        for (const auto &l : Lu) {
            markers[l.first] = l.second;
        }
//        for (VertexID i = 0; i < Lu.size(); ++i) {
//            markers[Lu[i].first] = Lu[i].second;
//        }
        for (const auto &l : Lv) {
            const auto &tmp_l = markers.find(l.first);
            if (tmp_l == markers.end()) {
                continue;
            }
            int d = tmp_l->second + l.second;
            if (d < dist) {
                dist = d;
            }
        }
//        for (VertexID i = 0; i < Lv.size(); ++i) {
//            const auto &tmp_l = markers.find(Lv[i].first);
//            if (tmp_l == markers.end()) {
//                continue;
//            }
//            int d = tmp_l->second + Lv[i].second;
//            if (d < dist) {
//                dist = d;
//            }
//        }
        if (dist == 255) {
            printf("2147483647\n");
        } else {
            printf("%u\n", dist);
        }
    }
}

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::switch_labels_to_old_id(
        const std::vector<VertexID> &rank2id)
//        const std::vector<VertexID> &rank)
{
    VertexID label_sum = 0;
    VertexID test_label_sum = 0;

//	VertexID num_v = rank2id.size();
//    VertexID num_v = rank.size();
    std::vector< std::vector< std::pair<VertexID, UnweightedDist> > > new_L(num_v);
//	for (VertexID r = 0; r < num_v; ++r) {
//		VertexID v = rank2id[r];
//		const IndexType &Lr = L[r];
//		IndexType &Lv = new_L[v];
//		VertexID size = Lr.get_size();
//		label_sum += size;
//		for (VertexID li = 0; li < size; ++li) {
//			VertexID l = Lr.get_label_ith_v(li);
//			VertexID new_l = rank2id[l];
//			Lv.add_label_seq(new_l, Lr.get_label_ith_d(li));
//		}
//	}
//	L = new_L;
    for (VertexID v_id = 0; v_id < num_v; ++v_id) {
        VertexID new_v = rank2id[v_id];
        const IndexType &Lv = L[v_id];
        // Traverse v_id's all existing labels
        for (VertexID b_i = 0; b_i < Lv.batches.size(); ++b_i) {
            VertexID id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
            VertexID dist_start_index = Lv.batches[b_i].start_index;
            VertexID dist_bound_index = dist_start_index + Lv.batches[b_i].size;
            // Traverse dist_table
            for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                label_sum += Lv.distances[dist_i].size;
                VertexID v_start_index = Lv.distances[dist_i].start_index;
                VertexID v_bound_index = v_start_index + Lv.distances[dist_i].size;
                UnweightedDist dist = Lv.distances[dist_i].dist;
                for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                    VertexID tail = Lv.vertices[v_i] + id_offset;
//					VertexID new_tail = rank2id[tail];
//					new_L[new_v].push_back(make_pair(new_tail, dist));
                    new_L[new_v].emplace_back(tail, dist);
//                    new_L[new_v].push_back(make_pair(tail, dist));
                    ++test_label_sum;
                }
            }
        }
    }
    printf("Label_sum: %u %u mean: %f\n", label_sum, test_label_sum, label_sum * 1.0 / num_v);

//	// Try to print
//	for (VertexID v = 0; v < num_v; ++v) {
//		const auto &Lv = new_L[v];
//		VertexID size = Lv.size();
//		printf("Vertex %u (Size %u):", v, size);
//		for (VertexID i = 0; i < size; ++i) {
//			printf(" (%u, %d)", Lv[i].first, Lv[i].second);
//			fflush(stdout);
//		}
//		puts("");
//	}

//	// Try query
    test_queries_normal_index(new_L);
//	VertexID u;
//	VertexID v;
//	while (std::cin >> u >> v) {
//		UnweightedDist dist = MAX_UNWEIGHTED_DIST;
//		// Bit Parallel Check
//		const IndexType &idx_u = L[rank[u]];
//		const IndexType &idx_v = L[rank[v]];
//
//		for (VertexID i = 0; i < BITPARALLEL_SIZE; ++i) {
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
////		unsorted_map<VertexID, UnweightedDist> markers;
//		map<VertexID, UnweightedDist> markers;
//		for (VertexID i = 0; i < Lu.size(); ++i) {
//			markers[Lu[i].first] = Lu[i].second;
//		}
//		for (VertexID i = 0; i < Lv.size(); ++i) {
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

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
UnweightedDist DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::dist_distance_query_pair(
        VertexID a_input,
        VertexID b_input,
        const DistGraph &G)
{
    VertexID a_global = G.rank[a_input];
    VertexID b_global = G.rank[b_input];
    VertexID a_host_id = G.get_master_host_id(a_global);
    VertexID b_host_id = G.get_master_host_id(b_global);
    UnweightedDist min_d = MAX_UNWEIGHTED_DIST;

    // Both local
    if (a_host_id == host_id && b_host_id == host_id) {
        std::map<VertexID, UnweightedDist> markers;
        // Traverse a's labels
        {
            VertexID a_local = G.get_local_vertex_id(a_global);
            IndexType &Lr = L[a_local];
            VertexID b_i_bound = Lr.batches.size();
            _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
            _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
            _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
            // Traverse batches array
            for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
                VertexID id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
                VertexID dist_start_index = Lr.batches[b_i].start_index;
                VertexID dist_bound_index = dist_start_index + Lr.batches[b_i].size;
                // Traverse distances array
                for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                    VertexID v_start_index = Lr.distances[dist_i].start_index;
                    VertexID v_bound_index = v_start_index + Lr.distances[dist_i].size;
                    UnweightedDist dist = Lr.distances[dist_i].dist;
                    // Traverse vertices array
                    for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                        VertexID label_id = Lr.vertices[v_i] + id_offset;
                        markers[label_id] = dist;
                    }
                }
            }
        }
        // Traverse b's labels
        {
            VertexID b_local = G.get_local_vertex_id(b_global);
            IndexType &Lr = L[b_local];
            VertexID b_i_bound = Lr.batches.size();
            _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
            _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
            _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
            // Traverse batches array
            for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
                VertexID id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
                VertexID dist_start_index = Lr.batches[b_i].start_index;
                VertexID dist_bound_index = dist_start_index + Lr.batches[b_i].size;
                // Traverse distances array
                for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                    VertexID v_start_index = Lr.distances[dist_i].start_index;
                    VertexID v_bound_index = v_start_index + Lr.distances[dist_i].size;
                    UnweightedDist dist = Lr.distances[dist_i].dist;
                    // Traverse vertices array
                    for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                        VertexID label_id = Lr.vertices[v_i] + id_offset;
                        const auto &tmp_l = markers.find(label_id);
                        if (tmp_l == markers.end()) {
                            continue;
                        }
                        int d = tmp_l->second + dist;
                        if (d < min_d) {
                            min_d = d;
                        }
                    }
                }
            }
        }
    } else {
        // Host b_host_id sends to host a_host_id, then host a_host_id do the query
        if (host_id == b_host_id) {
            std::vector< std::pair<VertexID, UnweightedDist> > buffer_send;
            VertexID b_local = G.get_local_vertex_id(b_global);
            IndexType &Lr = L[b_local];
            VertexID b_i_bound = Lr.batches.size();
            _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
            _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
            _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
            // Traverse batches array
            for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
                VertexID id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
                VertexID dist_start_index = Lr.batches[b_i].start_index;
                VertexID dist_bound_index = dist_start_index + Lr.batches[b_i].size;
                // Traverse distances array
                for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                    VertexID v_start_index = Lr.distances[dist_i].start_index;
                    VertexID v_bound_index = v_start_index + Lr.distances[dist_i].size;
                    UnweightedDist dist = Lr.distances[dist_i].dist;
                    // Traverse vertices array
                    for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                        VertexID label_id = Lr.vertices[v_i] + id_offset;
                        buffer_send.emplace_back(label_id, dist);
                    }
                }
            }
            MPI_Send(buffer_send.data(),
                    MPI_Instance::get_sending_size(buffer_send),
                    MPI_CHAR,
                    a_host_id,
                    SENDING_QUERY_LABELS,
                    MPI_COMM_WORLD);
        } else if (host_id == a_host_id) {
            std::map<VertexID, UnweightedDist> markers;
            // Traverse a's labels
            {
                VertexID a_local = G.get_local_vertex_id(a_global);
                IndexType &Lr = L[a_local];
                VertexID b_i_bound = Lr.batches.size();
                _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
                _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
                _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
                // Traverse batches array
                for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
                    VertexID id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
                    VertexID dist_start_index = Lr.batches[b_i].start_index;
                    VertexID dist_bound_index = dist_start_index + Lr.batches[b_i].size;
                    // Traverse distances array
                    for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                        VertexID v_start_index = Lr.distances[dist_i].start_index;
                        VertexID v_bound_index = v_start_index + Lr.distances[dist_i].size;
                        UnweightedDist dist = Lr.distances[dist_i].dist;
                        // Traverse vertices array
                        for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                            VertexID label_id = Lr.vertices[v_i] + id_offset;
                            markers[label_id] = dist;
                        }
                    }
                }
            }
            // Receive b's labels
            {
                std::vector<std::pair<VertexID, UnweightedDist> > buffer_recv;
                MPI_Instance::receive_dynamic_buffer_from_source(buffer_recv,
                                                                 num_hosts,
                                                                 b_host_id,
                                                                 SENDING_QUERY_LABELS);

                for (const auto &l : buffer_recv) {
                    VertexID label_id = l.first;
                    const auto &tmp_l = markers.find(label_id);
                    if (tmp_l == markers.end()) {
                        continue;
                    }
                    int d = tmp_l->second + l.second;
                    if (d < min_d) {
                        min_d = d;
                    }
                }
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE,
            &min_d,
            1,
            MPI_Instance::get_mpi_datatype<UnweightedDist>(),
            MPI_MIN,
            MPI_COMM_WORLD);
    return min_d;
}
}


#endif //PADO_DPADO_H
