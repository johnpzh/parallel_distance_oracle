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
        // Bit-parallel Labels
        UnweightedDist bp_dist[BITPARALLEL_SIZE];
        uint64_t bp_sets[BITPARALLEL_SIZE][2];  // [0]: S^{-1}, [1]: S^{0}

        std::vector<Batch> batches; // Batch info
        std::vector<DistanceIndexType> distances; // Distance info
        std::vector<VertexID> vertices; // Vertices in the label, presented as temporary ID

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

    // Type of Bit-Parallel Label
    struct BPLabelType {
        UnweightedDist bp_dist[BITPARALLEL_SIZE] = { 0 };
        uint64_t bp_sets[BITPARALLEL_SIZE][2] = { {0} }; // [0]: S^{-1}, [1]: S^{0}
    };

    VertexID num_v = 0;
    VertexID num_masters = 0;
    int host_id = 0;
    int num_hosts = 0;
    MPI_Datatype V_ID_Type;
    std::vector<IndexType> L;
    const VertexID UNIT_BUFFER_SIZE = (1U << 20U);
    std::vector<char> unit_buffer_send = std::vector<char>(UNIT_BUFFER_SIZE);
    std::vector<char> unit_buffer_recv = std::vector<char>(UNIT_BUFFER_SIZE);

    inline void bit_parallel_push_labels(
            const DistGraph &G,
            VertexID v_global,
            std::vector<VertexID> &tmp_que,
            VertexID &end_tmp_que,
            std::vector< std::pair<VertexID, VertexID> > &sibling_es,
            VertexID &num_sibling_es,
            std::vector< std::pair<VertexID, VertexID> > &child_es,
            VertexID &num_child_es,
            std::vector<UnweightedDist> &dists,
            UnweightedDist iter);
    inline void bit_parallel_labeling(
            const DistGraph &G,
//            std::vector<IndexType> &L,
            std::vector<uint8_t> &used_bp_roots);
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
            const std::vector<uint8_t> &used_bp_roots,
            std::vector<VertexID> &active_queue,
            VertexID &end_active_queue,
            std::vector<VertexID> &got_candidates_queue,
            VertexID &end_got_candidates_queue,
            std::vector<ShortIndex> &short_index,
            std::vector< std::vector<UnweightedDist> > &dist_table,
            std::vector< std::vector<VertexID> > &recved_dist_table,
            std::vector<BPLabelType> &bp_labels_table,
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
            std::vector<BPLabelType> &bp_labels_table,
            std::vector<VertexID> &active_queue,
            VertexID &end_active_queue,
            std::vector<VertexID> &once_candidated_queue,
            VertexID &end_once_candidated_queue,
            std::vector<bool> &once_candidated,
            VertexID b_id,
            VertexID roots_start,
            VertexID roots_size,
            std::vector<VertexID> &roots_master_local,
            const std::vector<uint8_t> &used_bp_roots);
    inline void sync_masters_2_mirrors(
            const DistGraph &G,
            const std::vector<VertexID> &active_queue,
            VertexID end_active_queue,
			std::vector< std::pair<VertexID, VertexID> > &buffer_send,
            std::vector<MPI_Request> &requests_send);
    inline void push_single_label(
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
            const std::vector<BPLabelType> &bp_labels_table,
            const std::vector<uint8_t> &used_bp_roots,
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
            const std::vector<BPLabelType> &bp_labels_table,
            const std::vector<uint8_t> &used_bp_roots,
            UnweightedDist iter);
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
            std::vector< std::vector<VertexID> > &recved_dist_table,
            std::vector<BPLabelType> &bp_labels_table);


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

    UnweightedDist dist_distance_query_pair(
            VertexID a_global,
            VertexID b_global,
            const DistGraph &G);
}; // class DistBVCPLL

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
    std::vector<uint8_t> used_bp_roots(num_v, 0);
    //cache_miss.measure_start();
    double time_labeling = -WallTimer::get_time_mark();

    bit_parallel_labeling(G,
            used_bp_roots);

    std::vector<VertexID> active_queue(num_masters); // Any vertex v who is active should be put into this queue.
    VertexID end_active_queue = 0;
    std::vector<bool> is_active(num_masters, false);// is_active[v] is true means vertex v is in the active queue.
    std::vector<VertexID> got_candidates_queue(num_masters); // Any vertex v who got candidates should be put into this queue.
    VertexID end_got_candidates_queue = 0;
    std::vector<bool> got_candidates(num_masters, false); // got_candidates[v] is true means vertex v is in the queue got_candidates_queue
    std::vector<ShortIndex> short_index(num_masters);
    std::vector< std::vector<UnweightedDist> > dist_table(BATCH_SIZE, std::vector<UnweightedDist>(num_v, MAX_UNWEIGHTED_DIST));
    std::vector<VertexID> once_candidated_queue(num_masters); // if short_index[v].indicator.any() is true, v is in the queue.
        // Used mainly for resetting short_index[v].indicator.
    VertexID end_once_candidated_queue = 0;
    std::vector<bool> once_candidated(num_masters, false);

//    std::vector<VertexID> active_queue(num_v); // Any vertex v who is active should be put into this queue.
//    VertexID end_active_queue = 0;
//    std::vector<bool> is_active(num_v, false);// is_active[v] is true means vertex v is in the active queue.
//    std::vector<VertexID> got_candidates_queue(num_v); // Any vertex v who got candidates should be put into this queue.
//    VertexID end_got_candidates_queue = 0;
//    std::vector<bool> got_candidates(num_v, false); // got_candidates[v] is true means vertex v is in the queue got_candidates_queue
//    std::vector<ShortIndex> short_index(num_v);
//    std::vector< std::vector<UnweightedDist> > dist_table(BATCH_SIZE, std::vector<UnweightedDist>(num_v, MAX_UNWEIGHTED_DIST));
//    std::vector<VertexID> once_candidated_queue(num_v); // if short_index[v].indicator.any() is true, v is in the queue.
//        // Used mainly for resetting short_index[v].indicator.
//    VertexID end_once_candidated_queue = 0;
//    std::vector<bool> once_candidated(num_v, false);

    std::vector< std::vector<VertexID> > recved_dist_table(BATCH_SIZE); // Some distances are from other hosts. This is used to reset the dist_table.
    std::vector<BPLabelType> bp_labels_table(BATCH_SIZE); // All roots' bit-parallel labels

    //printf("b_i_bound: %u\n", b_i_bound);//test
    for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
//        if (0 == host_id) {
//            printf("b_i: %u\n", b_i);//test
//        }

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
                bp_labels_table,
                got_candidates,
                is_active,
                once_candidated_queue,
                end_once_candidated_queue,
                once_candidated);
//        exit(EXIT_SUCCESS); //test
    }
    if (remainer != 0) {
//        if (0 == host_id) {
//            printf("b_i: %u\n", b_i_bound);//test
//        }
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
                bp_labels_table,
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

    // Total Number of Labels
    EdgeID local_num_labels = 0;
    for (VertexID v_global = 0; v_global < num_v; ++v_global) {
        if (G.get_master_host_id(v_global) != host_id) {
            continue;
        }
        local_num_labels += L[G.get_local_vertex_id(v_global)].vertices.size();
    }
    EdgeID global_num_labels;
    MPI_Allreduce(&local_num_labels,
            &global_num_labels,
            1,
            MPI_Instance::get_mpi_datatype<EdgeID>(),
            MPI_SUM,
            MPI_COMM_WORLD);
    printf("host_id: %u local_num_labels: %lu %.2f%%\n", host_id, local_num_labels, 100.0 * local_num_labels / global_num_labels);
    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == host_id) {
        printf("Global_num_labels: %lu average: %f\n", global_num_labels, 1.0 * global_num_labels / num_v);
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
//        std::vector<uint8_t> &used_bp_roots)
//{
////    VertexID num_v = G.num_v;
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
//        VertexID d_i_bound = G.local_out_degrees[r];
//        EdgeID i_start = G.vertices_idx[r] + d_i_bound - 1;
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
////                bit_parallel_push_labels(G,
////                        v,
////                        que,
////                        que_h,
////                        sibling_es,
////                        num_sibling_es,
////                        child_es,
////                        num_child_es,
////                        tmp_d,
////                        d);
//                EdgeID i_start = G.vertices_idx[v];
//                EdgeID i_bound = i_start + G.local_out_degrees[v];
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
//            {// test
//                printf("iter %u @%u host_id: %u num_sibling_es: %u num_child_es: %u\n", d, __LINE__, host_id, num_sibling_es, num_child_es);
////                if (4 == d) {
////                    exit(EXIT_SUCCESS);
////                }
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

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::bit_parallel_push_labels(
        const DistGraph &G,
        const VertexID v_global,
        std::vector<VertexID> &tmp_que,
        VertexID &end_tmp_que,
        std::vector< std::pair<VertexID, VertexID> > &sibling_es,
        VertexID &num_sibling_es,
        std::vector< std::pair<VertexID, VertexID> > &child_es,
        VertexID &num_child_es,
        std::vector<UnweightedDist> &dists,
        const UnweightedDist iter)
{
    EdgeID i_start = G.vertices_idx[v_global];
    EdgeID i_bound = i_start + G.local_out_degrees[v_global];
    for (EdgeID i = i_start; i < i_bound; ++i) {
        VertexID tv_global = G.out_edges[i];
        VertexID tv_local = G.get_local_vertex_id(tv_global);
        UnweightedDist td = iter + 1;

        if (iter > dists[tv_local]) {
            ;
        } else if (iter == dists[tv_local]) {
            if (v_global < tv_global) { // ??? Why need v < tv !!! Because it's a undirected graph.
                sibling_es[num_sibling_es].first = v_global;
                sibling_es[num_sibling_es].second = tv_global;
                ++num_sibling_es;
            }
        } else { // iter < dists[tv]
            if (dists[tv_local] == MAX_UNWEIGHTED_DIST) {
                tmp_que[end_tmp_que++] = tv_global;
                dists[tv_local] = td;
            }
            child_es[num_child_es].first = v_global;
            child_es[num_child_es].second = tv_global;
            ++num_child_es;
//            {
//                printf("num_child_es: %u v_global: %u tv_global: %u\n", num_child_es, v_global, tv_global);//test
//            }
        }
    }
}

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::bit_parallel_labeling(
        const DistGraph &G,
//        std::vector<IndexType> &L,
        std::vector<uint8_t> &used_bp_roots)
{
    // Class type of Bit-Parallel label message unit.
    struct MsgUnitBP {
        VertexID v_global;
        uint64_t S_n1;
        uint64_t S_0;

        MsgUnitBP() = default;
        MsgUnitBP(VertexID v, uint64_t sn1, uint64_t s0)
            : v_global(v), S_n1(sn1), S_0(s0) { }
    };
//    VertexID num_v = G.num_v;
//    EdgeID num_e = G.num_e;
    EdgeID local_num_edges = G.num_edges_local;

    std::vector<UnweightedDist> tmp_d(num_masters); // distances from the root to every v
    std::vector<std::pair<uint64_t, uint64_t> > tmp_s(num_v); // first is S_r^{-1}, second is S_r^{0}
    std::vector<VertexID> que(num_masters); // active queue
    VertexID end_que = 0;
    std::vector<VertexID> tmp_que(num_masters); // temporary queue, to be swapped with que
    VertexID end_tmp_que = 0;
    std::vector<std::pair<VertexID, VertexID> > sibling_es(local_num_edges); // siblings, their distances to the root are equal (have difference of 0)
    std::vector<std::pair<VertexID, VertexID> > child_es(local_num_edges); // child and father, their distances to the root have difference of 1.

//    std::vector<UnweightedDist> tmp_d(num_v); // distances from the root to every v
//    std::vector<std::pair<uint64_t, uint64_t> > tmp_s(num_v); // first is S_r^{-1}, second is S_r^{0}
//    std::vector<VertexID> que(num_v); // active queue
//    std::vector<std::pair<VertexID, VertexID> > sibling_es(num_e); // siblings, their distances to the root are equal (have difference of 0)
//    std::vector<std::pair<VertexID, VertexID> > child_es(num_e); // child and father, their distances to the root have difference of 1.

    VertexID r_global = 0; // root r
    for (VertexID i_bpspt = 0; i_bpspt < BITPARALLEL_SIZE; ++i_bpspt) {
        // Select the root r_global
        if (0 == host_id) {
            while (r_global < num_v && used_bp_roots[r_global]) {
                ++r_global;
            }
            if (r_global == num_v) {
                for (VertexID v = 0; v < num_v; ++v) {
                    L[v].bp_dist[i_bpspt] = MAX_UNWEIGHTED_DIST;
                }
                continue;
            }
        }
        // Broadcast the r here.
        MPI_Bcast(&r_global,
                1,
                V_ID_Type,
                0,
                MPI_COMM_WORLD);
        used_bp_roots[r_global] = 1;
//        {//test
//            if (0 == host_id) {
//                printf("i: %u r: %u\n", i_bpspt, r_global);
//            }
//        }

//        VertexID que_t0 = 0, que_t1 = 0, que_h = 0;
        fill(tmp_d.begin(), tmp_d.end(), MAX_UNWEIGHTED_DIST);
        fill(tmp_s.begin(), tmp_s.end(), std::make_pair(0, 0));

        // Mark the r_global
        if (G.get_master_host_id(r_global) == host_id) {
            tmp_d[G.get_local_vertex_id(r_global)] = 0;
            que[end_que++] = r_global;
        }
//        {// test
//            printf("r: %u @%u host_id: %u G.local_out_degrees[%u]: %u\n", r_global, __LINE__, host_id, r_global, G.local_out_degrees[r_global]);
//        }
        // Select the r_global's 64 neighbors
        {
            // Get r_global's neighbors into buffer_send, rank from low to high.
            VertexID local_degree = G.local_out_degrees[r_global];
            std::vector<VertexID> buffer_send(local_degree);
            if (local_degree) {
                EdgeID e_i_start = G.vertices_idx[r_global] + local_degree - 1;
                VertexID v_i = 0;
                for (VertexID d_i = 0; d_i < local_degree; ++d_i) {
                    EdgeID e_i = e_i_start - d_i;
                    buffer_send[v_i++] = G.out_edges[e_i];
                }
            }
//            {//test
//                printf("@%u host_id: %u buffer_send.size(): %lu\n", __LINE__, host_id, buffer_send.size());
//            }

//            std::vector<VertexID> all_nbrs(G.get_global_out_degree(r_global));
            // Get selected neighbors (up to 64)
            std::vector<VertexID> selected_nbrs;
            if (host_id) {
                // Every host other than 0 sends neighbors to host 0
                MPI_Send(buffer_send.data(),
                        buffer_send.size(),
                        V_ID_Type,
                        0,
                        SENDING_ROOT_NEIGHBORS,
                        MPI_COMM_WORLD);
                // Receive selected neighbors from host 0
                MPI_Instance::receive_dynamic_buffer_from_source(selected_nbrs,
                        num_hosts,
                        0,
                        SENDING_SELECTED_NEIGHBORS);
            } else {
                // Host 0 receives neighbors from others
                std::vector<VertexID> all_nbrs(buffer_send);
                std::vector<VertexID > buffer_recv;
                for (int loc = 0; loc < num_hosts - 1; ++loc) {
                    MPI_Instance::receive_dynamic_buffer_from_any(buffer_recv,
                            num_hosts,
                            SENDING_ROOT_NEIGHBORS);
                    if (buffer_recv.empty()) {
                        continue;
                    }
//                    {// test
//                        printf("@%u host_id: %u buffer_recv.size(): %lu\n", __LINE__, host_id, buffer_recv.size());
//                    }
                    buffer_send.resize(buffer_send.size() + buffer_recv.size());
                    std::merge(buffer_recv.begin(), buffer_recv.end(), all_nbrs.begin(), all_nbrs.end(), buffer_send.begin());
//                    all_nbrs.swap(buffer_send);
                    all_nbrs.resize(buffer_send.size());
                    std::copy(buffer_send.begin(), buffer_send.begin(), all_nbrs.begin());
//                    {
//                        printf("@%u host_id: %u loc: %u all_nbrs.size(): %lu buffer_send.size(): %lu\n", __LINE__, host_id, loc, all_nbrs.size(), buffer_send.size());
//                    }
                }
//                {//test
//                    printf("@%u host_id: %u all_nbrs: %lu global_out_degree: %u\n", __LINE__, host_id, all_nbrs.size(), G.get_global_out_degree(r_global));
//                }
                assert(all_nbrs.size() == G.get_global_out_degree(r_global));
                // Select 64 (or less) neighbors
                VertexID ns = 0; // number of selected neighbor, default 64
                for (VertexID v_global : all_nbrs) {
                    if (used_bp_roots[v_global]) {
                        continue;
                    }
                    selected_nbrs.push_back(v_global);
                    if (++ns == 64) {
                        break;
                    }
                }
                // Send selected neighbors to other hosts
                for (int dest = 1; dest < num_hosts; ++dest) {
                    MPI_Send(selected_nbrs.data(),
                            selected_nbrs.size(),
                            V_ID_Type,
                            dest,
                            SENDING_SELECTED_NEIGHBORS,
                            MPI_COMM_WORLD);
                }
            }

            // Synchronize the used_bp_roots.
            for (VertexID v_global : selected_nbrs) {
                used_bp_roots[v_global] = 1;
            }

            // Mark selected neighbors
            for (VertexID v_i = 0; v_i < selected_nbrs.size(); ++v_i) {
                VertexID v_global = selected_nbrs[v_i];
                if (host_id != G.get_master_host_id(v_global)) {
                    continue;
                }
                tmp_que[end_tmp_que++] = v_global;
                tmp_d[G.get_local_vertex_id(v_global)] = 1;
                tmp_s[v_global].first = 1ULL << v_i;
            }
        }

        // Reduce the global number of active vertices
        VertexID global_num_actives = 1;
//        MPI_Allreduce(&end_que,
//                &global_num_actives,
//                1,
//                V_ID_Type,
//                MPI_SUM,
//                MPI_COMM_WORLD);
        UnweightedDist d = 0;
        while (global_num_actives) {
//            for (UnweightedDist d = 0; que_t0 < que_h; ++d) {
            VertexID num_sibling_es = 0, num_child_es = 0;

            // Local scatter
//            {
//                printf("iter: %u end_que: %u\n", d, end_que);//test
//            }
            for (VertexID que_i = 0; que_i < end_que; ++que_i) {
                VertexID v_global = que[que_i];
                if (!G.local_out_degrees[v_global]) {
                    continue;
                }
                bit_parallel_push_labels(G,
                        v_global,
                        tmp_que,
                        end_tmp_que,
                        sibling_es,
                        num_sibling_es,
                        child_es,
                        num_child_es,
                        tmp_d,
                        d);
            }

            // Send active masters to mirrors
            {
                std::vector<MPI_Request> requests_send(num_hosts - 1);
                std::vector<MsgUnitBP> buffer_send;
                for (VertexID que_i = 0; que_i < end_que; ++que_i) {
                    VertexID v_global = que[que_i];
                    buffer_send.emplace_back(v_global, // v_global
                                             tmp_s[v_global].first, // S_n1
                                             tmp_s[v_global].second); // S_0
                }
                for (int loc = 0; loc < num_hosts - 1; ++loc) {
                    int dest_host_id = G.buffer_send_list_loc_2_master_host_id(loc);
                    MPI_Isend(buffer_send.data(),
                              MPI_Instance::get_sending_size(buffer_send),
                              MPI_CHAR,
                              dest_host_id,
                              SENDING_BP_ACTIVES,
                              MPI_COMM_WORLD,
                              &requests_send[loc]);
                }
                // Receive active masters from other hosts
                std::vector<MsgUnitBP> buffer_recv;
                for (int loc = 0; loc < num_hosts - 1; ++loc) {
                    MPI_Instance::receive_dynamic_buffer_from_any(buffer_recv,
                                                                  num_hosts,
                                                                  SENDING_BP_ACTIVES);
                    if (buffer_recv.empty()) {
                        continue;
                    }
                    for (const auto &m : buffer_recv) {
                        VertexID v_global = m.v_global;
                        if (!G.local_out_degrees[v_global]) {
                            continue;
                        }
                        tmp_s[v_global].first = m.S_n1;
                        tmp_s[v_global].second = m.S_0;
                        // Push labels
                        bit_parallel_push_labels(G,
                                                 v_global,
                                                 tmp_que,
                                                 end_tmp_que,
                                                 sibling_es,
                                                 num_sibling_es,
                                                 child_es,
                                                 num_child_es,
                                                 tmp_d,
                                                 d);
                    }
                }
                MPI_Waitall(num_hosts - 1,
                            requests_send.data(),
                            MPI_STATUSES_IGNORE);
            }

            // Update the sets in tmp_s
            {

                for (VertexID i = 0; i < num_sibling_es; ++i) {
                    VertexID v = sibling_es[i].first, w = sibling_es[i].second;
                    tmp_s[v].second |= tmp_s[w].first; // !!! Need to send back!!!
                    tmp_s[w].second |= tmp_s[v].first;

                }
                // Put into the buffer sending to others
                std::vector<MPI_Request> requests_send(num_hosts - 1);
                std::vector< std::pair<VertexID, uint64_t> > buffer_send;
                for (VertexID i = 0; i < num_sibling_es; ++i) {
                    VertexID v = sibling_es[i].first;
                    VertexID w = sibling_es[i].second;
                    buffer_send.emplace_back(v, tmp_s[v].second);
                    buffer_send.emplace_back(w, tmp_s[w].second);
                }
                // Send the messages
                for (int loc = 0; loc < num_hosts - 1; ++loc) {
                    int dest_host_id = G.buffer_send_list_loc_2_master_host_id(loc);
                    MPI_Isend(buffer_send.data(),
                              MPI_Instance::get_sending_size(buffer_send),
                              MPI_CHAR,
                              dest_host_id,
                              SENDING_SETS_UPDATES_BP,
                              MPI_COMM_WORLD,
                              &requests_send[loc]);
                }
                // Receive the messages
                std::vector<std::pair<VertexID, uint64_t> > buffer_recv;
                for (int loc = 0; loc < num_hosts - 1; ++loc) {
                    MPI_Instance::receive_dynamic_buffer_from_any(buffer_recv,
                            num_hosts,
                            SENDING_SETS_UPDATES_BP);
                    if (buffer_recv.empty()) {
                        continue;
                    }
                    for (const auto &m : buffer_recv) {
                        VertexID v_global = m.first;
//                        if (!G.local_out_degrees[v_global] && (G.get_master_host_id(v_global) != host_id)) {
//                        //  This if-condition is correct, but not necessary for performance improvement
//                            continue;
//                        }
                        tmp_s[v_global].second |= m.second;
                    }
                }
                MPI_Waitall(num_hosts - 1,
                        requests_send.data(),
                        MPI_STATUSES_IGNORE);
                for (VertexID i = 0; i < num_child_es; ++i) {
                    VertexID v = child_es[i].first, c = child_es[i].second;
                    tmp_s[c].first |= tmp_s[v].first;
                    tmp_s[c].second |= tmp_s[v].second;
                }
            }
//            {// test
//                VertexID global_num_sibling_es;
//                VertexID global_num_child_es;
//                MPI_Allreduce(&num_sibling_es,
//                        &global_num_sibling_es,
//                        1,
//                        V_ID_Type,
//                        MPI_SUM,
//                        MPI_COMM_WORLD);
//                MPI_Allreduce(&num_child_es,
//                              &global_num_child_es,
//                              1,
//                              V_ID_Type,
//                              MPI_SUM,
//                              MPI_COMM_WORLD);
//                if (0 == host_id) {
//                    printf("iter %u num_sibling_es: %u num_child_es: %u\n", d, global_num_sibling_es, global_num_child_es);
//                }
//
////                printf("iter %u @%u host_id: %u num_sibling_es: %u num_child_es: %u\n", d, __LINE__, host_id, num_sibling_es, num_child_es);
////                if (4 == d) {
////                    exit(EXIT_SUCCESS);
////                }
//            }

            // Swap que and tmp_que
            tmp_que.swap(que);
            end_que = end_tmp_que;
            end_tmp_que = 0;
            MPI_Allreduce(&end_que,
                      &global_num_actives,
                      1,
                      V_ID_Type,
                      MPI_SUM,
                      MPI_COMM_WORLD);

//            }
            ++d;
        }

        for (VertexID v_local = 0; v_local < num_masters; ++v_local) {
            VertexID v_global = G.get_global_vertex_id(v_local);
            L[v_local].bp_dist[i_bpspt] = tmp_d[v_local];
            L[v_local].bp_sets[i_bpspt][0] = tmp_s[v_global].first; // S_r^{-1}
            L[v_local].bp_sets[i_bpspt][1] = tmp_s[v_global].second & ~tmp_s[v_global].first; // Only need those r's neighbors who are not already in S_r^{-1}
        }
    }
//    {//test
//        struct TmpMsgBP {
//            UnweightedDist dist;
//            uint64_t sn1;
//            uint64_t s0;
//            TmpMsgBP() = default;
//            TmpMsgBP(UnweightedDist d, uint64_t sn1_, uint64_t s0_)
//                : dist(d), sn1(sn1_), s0(s0_) { }
//        };
//        for (VertexID v = 0; v < num_v; ++v) {
//            int v_host_id = G.get_master_host_id(v);
//            if (v_host_id == host_id) {
//                VertexID v_local = G.get_local_vertex_id(v);
//                if (host_id == 0) {
//                    for (VertexID bp_i = 0; bp_i < BITPARALLEL_SIZE; ++bp_i) {
//                        printf("v: %u d[%u]: %u s-1[%u]: %lu s0[%u]: %lu\n",
//                               v, bp_i, L[v_local].bp_dist[bp_i], bp_i, L[v_local].bp_sets[bp_i][0], bp_i, L[v_local].bp_sets[bp_i][1]);
//                    }
//                } else {
//                    std::vector<TmpMsgBP> buffer_send;
//                    for (VertexID bp_i = 0; bp_i < BITPARALLEL_SIZE; ++bp_i) {
//                        buffer_send.emplace_back(L[v_local].bp_dist[bp_i], L[v_local].bp_sets[bp_i][0], L[v_local].bp_sets[bp_i][1]);
//                    }
//                    MPI_Send(buffer_send.data(),
//                            MPI_Instance::get_sending_size(buffer_send),
//                            MPI_CHAR,
//                            0,
//                            GRAPH_SHUFFLE,
//                            MPI_COMM_WORLD);
//                }
//            } else if (0 == host_id) {
//                std::vector<TmpMsgBP> buffer_recv;
//                MPI_Instance::receive_dynamic_buffer_from_source(buffer_recv,
//                        num_hosts,
//                        v_host_id,
//                        GRAPH_SHUFFLE);
//                for (VertexID bp_i = 0; bp_i < BITPARALLEL_SIZE; ++bp_i) {
//                    printf("v: %u d[%u]: %u s-1[%u]: %lu s0[%u]: %lu\n",
//                           v, bp_i, buffer_recv[bp_i].dist, bp_i, buffer_recv[bp_i].sn1, bp_i, buffer_recv[bp_i].s0);
//                }
//            }
//            MPI_Barrier(MPI_COMM_WORLD);
//        }
//    }
}

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
        std::vector<BPLabelType> &bp_labels_table,
        std::vector<VertexID> &active_queue,
        VertexID &end_active_queue,
        std::vector<VertexID> &once_candidated_queue,
        VertexID &end_once_candidated_queue,
        std::vector<bool> &once_candidated,
        VertexID b_id,
        VertexID roots_start,
        VertexID roots_size,
        std::vector<VertexID> &roots_master_local,
        const std::vector<uint8_t> &used_bp_roots)
{
    // Get the roots_master_local, containing all local roots.
    VertexID roots_bound = roots_start + roots_size;
    for (VertexID r_global = roots_start; r_global < roots_bound; ++r_global) {
        if (G.get_master_host_id(r_global) == host_id && !used_bp_roots[r_global]) {
            roots_master_local.push_back(G.get_local_vertex_id(r_global));
        }
    }
    // Short_index
    {
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
    }
//
    // Real Index
    {
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
    }

    // Dist Table
    {
        struct LabelTableUnit {
            VertexID root_id;
            VertexID label_global_id;
            UnweightedDist dist;

            LabelTableUnit() = default;

            LabelTableUnit(VertexID r, VertexID l, UnweightedDist d) :
                    root_id(r), label_global_id(l), dist(d) {}
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
                            buffer_send.emplace_back(r_root_id, Lr.vertices[v_i] + id_offset,
                                                     dist); // buffer for sending
                        }
                    }
                }
            }
        }
        // Broadcast local roots labels
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

        // Receive labels from every other host
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
    }

	// Build the Bit-Parallel Labels Table
    {
        struct MsgBPLabel {
            VertexID r_root_id;
            UnweightedDist bp_dist[BITPARALLEL_SIZE];
            uint64_t bp_sets[BITPARALLEL_SIZE][2];

            MsgBPLabel() = default;
            MsgBPLabel(VertexID r, const UnweightedDist dist[], const uint64_t sets[][2])
                    : r_root_id(r)
            {
                memcpy(bp_dist, dist, sizeof(bp_dist));
                memcpy(bp_sets, sets, sizeof(bp_sets));
            }
        };
        std::vector<MPI_Request> requests_send(num_hosts - 1);
        std::vector<MsgBPLabel> buffer_send;
        for (VertexID r_global = roots_start; r_global < roots_bound; ++r_global) {
            if (G.get_master_host_id(r_global) != host_id) {
                continue;
            }
            VertexID r_local = G.get_local_vertex_id(r_global);
            VertexID r_root = r_global - roots_start;
            // Local roots
            memcpy(bp_labels_table[r_root].bp_dist, L[r_local].bp_dist, sizeof(bp_labels_table[r_root].bp_dist));
            memcpy(bp_labels_table[r_root].bp_sets, L[r_local].bp_sets, sizeof(bp_labels_table[r_root].bp_sets));
            // Prepare for sending
            buffer_send.emplace_back(r_root, L[r_local].bp_dist, L[r_local].bp_sets);
        }
        // Send
        for (int loc = 0; loc < num_hosts - 1; ++loc) {
            int dest_host_id = G.buffer_send_list_loc_2_master_host_id(loc);
            MPI_Isend(buffer_send.data(),
                    MPI_Instance::get_sending_size(buffer_send),
                    MPI_CHAR,
                    dest_host_id,
                    SENDING_ROOT_BP_LABELS,
                    MPI_COMM_WORLD,
                    &requests_send[loc]);
        }
        // Receive
        std::vector<MsgBPLabel> buffer_recv;
        for (int loc = 0; loc < num_hosts - 1; ++loc) {
            MPI_Instance::receive_dynamic_buffer_from_any(buffer_recv,
                    num_hosts,
                    SENDING_ROOT_BP_LABELS);
            if (buffer_recv.empty()) {
                continue;
            }
            for (const auto &m : buffer_recv) {
                VertexID r_root = m.r_root_id;
                memcpy(bp_labels_table[r_root].bp_dist, m.bp_dist, sizeof(bp_labels_table[r_root].bp_dist));
                memcpy(bp_labels_table[r_root].bp_sets, m.bp_sets, sizeof(bp_labels_table[r_root].bp_sets));
            }
        }
        MPI_Waitall(num_hosts - 1,
                    requests_send.data(),
                    MPI_STATUSES_IGNORE);
//        {// test
//            if (2 == host_id) {
//                for (VertexID r_i = 0; r_i < BATCH_SIZE; ++r_i) {
//                    for (VertexID b_i = 0; b_i < BITPARALLEL_SIZE; ++b_i) {
//                        printf("v: %u d[%u]: %u s-1[%u]: %lu s0[%u]: %lu\n",
//                                r_i + roots_start, b_i, bp_labels_table[r_i].bp_dist[b_i],
//                                b_i, bp_labels_table[r_i].bp_sets[b_i][0],
//                                b_i, bp_labels_table[r_i].bp_sets[b_i][1]);
//                    }
//                }
//            }
//            MPI_Barrier(MPI_COMM_WORLD);
//            exit(0);
//        }
    }

    // TODO: parallel enqueue
    // Active_queue
    VertexID global_num_actives = 0; // global number of active vertices.
    {
        for (VertexID r_local : roots_master_local) {
            active_queue[end_active_queue++] = r_local;
        }
        // Get the global number of active vertices;
        MPI_Allreduce(&end_active_queue,
                      &global_num_actives,
                      1,
                      V_ID_Type,
                      MPI_SUM,
                      MPI_COMM_WORLD);
    }

    return global_num_actives;
}

// Function: push v_head_global's newly added labels to its all neighbors.
template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::push_single_label(
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
        const std::vector<BPLabelType> &bp_labels_table,
        const std::vector<uint8_t> &used_bp_roots,
        UnweightedDist iter)
{
    const BPLabelType &L_label = bp_labels_table[label_root_id];
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
        const IndexType &L_tail = L[v_tail_local];
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
        // Bit Parallel Checking: if label_global_id to v_tail_global has shorter distance already
        //			++total_check_count;
//        const IndexType &L_label = L[label_global_id];
//        _mm_prefetch(&L_label.bp_dist[0], _MM_HINT_T0);
//        _mm_prefetch(&L_label.bp_sets[0][0], _MM_HINT_T0);
//			bp_checking_ins_count.measure_start();
        bool no_need_add = false;
        for (VertexID i = 0; i < BITPARALLEL_SIZE; ++i) {
            VertexID td = L_label.bp_dist[i] + L_tail.bp_dist[i];
            if (td - 2 <= iter) {
                td +=
                        (L_label.bp_sets[i][0] & L_tail.bp_sets[i][0]) ? -2 :
                        ((L_label.bp_sets[i][0] & L_tail.bp_sets[i][1]) |
                         (L_label.bp_sets[i][1] & L_tail.bp_sets[i][0]))
                        ? -1 : 0;
                if (td <= iter) {
                    no_need_add = true;
//						++bp_hit_count;
                    break;
                }
            }
        }
        if (no_need_add) {
//				bp_checking_ins_count.measure_stop();
            continue;
        }
//			bp_checking_ins_count.measure_stop();
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
        const std::vector<BPLabelType> &bp_labels_table,
        const std::vector<uint8_t> &used_bp_roots,
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
            const IndexType &L_tail = L[v_tail_local];
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

            // Bit Parallel Checking: if label_global_id to v_tail_global has shorter distance already
            //			++total_check_count;
//            const IndexType &L_label = L[label_global_id];
//            _mm_prefetch(&L_label.bp_dist[0], _MM_HINT_T0);
//            _mm_prefetch(&L_label.bp_sets[0][0], _MM_HINT_T0);
//			bp_checking_ins_count.measure_start();
            const BPLabelType &L_label = bp_labels_table[label_root_id];
            bool no_need_add = false;
            for (VertexID i = 0; i < BITPARALLEL_SIZE; ++i) {
                VertexID td = L_label.bp_dist[i] + L_tail.bp_dist[i];
                if (td - 2 <= iter) {
                    td +=
                            (L_label.bp_sets[i][0] & L_tail.bp_sets[i][0]) ? -2 :
                            ((L_label.bp_sets[i][0] & L_tail.bp_sets[i][1]) |
                             (L_label.bp_sets[i][1] & L_tail.bp_sets[i][0]))
                            ? -1 : 0;
                    if (td <= iter) {
                        no_need_add = true;
//						++bp_hit_count;
                        break;
                    }
                }
            }
            if (no_need_add) {
//				bp_checking_ins_count.measure_stop();
                continue;
            }
//			bp_checking_ins_count.measure_stop();
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


// DEPRECATED Function: in the scatter phase, synchronize local masters to mirrors on other hosts
// Has some mysterious problem: when I call this function, some hosts will receive wrong messages; when I copy all
// code of this function into the caller, all messages become right.
template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::sync_masters_2_mirrors(
        const DistGraph &G,
        const std::vector<VertexID> &active_queue,
        VertexID end_active_queue,
		std::vector< std::pair<VertexID, VertexID> > &buffer_send,
        std::vector<MPI_Request> &requests_send
)
{
//    std::vector< std::pair<VertexID, VertexID> > buffer_send;
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
//			{//test
//				if (1 == host_id) {
//					printf("@%u host_id: %u v_head_global: %u\n", __LINE__, host_id, v_head_global);//
//				}
//			}
        }
    }
	{
		if (!buffer_send.empty()) {
			printf("@%u host_id: %u sync_masters_2_mirrors: buffer_send.size: %lu buffer_send[0]:(%u %u)\n", __LINE__, host_id, buffer_send.size(), buffer_send[0].first, buffer_send[0].second);
		}
		assert(!requests_send.empty());
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
		{
			if (!buffer_send.empty()) {
				printf("@%u host_id: %u dest_host_id: %u buffer_send.size: %lu buffer_send[0]:(%u %u)\n", __LINE__, host_id, dest_host_id, buffer_send.size(), buffer_send[0].first, buffer_send[0].second);
			}
		}
    }
}

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
                if (d_tmp <= iter) {
                    return false;
                }
            }
        }
    }
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
        std::vector< std::vector<VertexID> > &recved_dist_table,
        std::vector<BPLabelType> &bp_labels_table)
{
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
    }
    // Reset dist_table according to received masters' labels from other hosts
    for (VertexID r_root_id = 0; r_root_id < BATCH_SIZE; ++r_root_id) {
        for (VertexID cand_real_id : recved_dist_table[r_root_id]) {
            dist_table[r_root_id][cand_real_id] = MAX_UNWEIGHTED_DIST;
        }
        recved_dist_table[r_root_id].clear();
    }
    // Reset bit-parallel labels table
    for (VertexID r_root_id = 0; r_root_id < BATCH_SIZE; ++r_root_id) {
        memset(bp_labels_table[r_root_id].bp_dist, 0, sizeof(bp_labels_table[r_root_id].bp_dist));
        memset(bp_labels_table[r_root_id].bp_sets, 0, sizeof(bp_labels_table[r_root_id].bp_sets));
    }
}

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::batch_process(
        const DistGraph &G,
        VertexID b_id,
        VertexID roots_start, // start id of roots
        VertexID roots_size, // how many roots in the batch
        const std::vector<uint8_t> &used_bp_roots,
        std::vector<VertexID> &active_queue,
        VertexID &end_active_queue,
        std::vector<VertexID> &got_candidates_queue,
        VertexID &end_got_candidates_queue,
        std::vector<ShortIndex> &short_index,
        std::vector< std::vector<UnweightedDist> > &dist_table,
        std::vector< std::vector<VertexID> > &recved_dist_table,
        std::vector<BPLabelType> &bp_labels_table,
        std::vector<bool> &got_candidates,
        std::vector<bool> &is_active,
        std::vector<VertexID> &once_candidated_queue,
        VertexID &end_once_candidated_queue,
        std::vector<bool> &once_candidated)
{
    // At the beginning of a batch, initialize the labels L and distance buffer dist_table;
    std::vector<VertexID> roots_master_local; // Roots which belongs to this host.
    VertexID global_num_actives = initialization(G,
                                    short_index,
                                    dist_table,
                                    recved_dist_table,
                                    bp_labels_table,
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

    UnweightedDist iter = 0; // The iterator, also the distance for current iteration


    while (global_num_actives) {
        ++iter;
        // Traverse active vertices to push their labels as candidates
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
                    bp_labels_table,
                    used_bp_roots,
                    iter);
        }

		// Send masters' newly added labels to other hosts
		{
			std::vector<MPI_Request> requests_send(num_hosts - 1);
//			sync_masters_2_mirrors(G,
//					active_queue,
//					end_active_queue,
//					requests_send);
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
					push_single_label(
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
                            bp_labels_table,
							used_bp_roots,
							iter);
				}
			}
			end_active_queue = 0;
			MPI_Waitall(num_hosts - 1,
					requests_send.data(),
					MPI_STATUSES_IGNORE);
//			{// test
//				VertexID global_end_got_candidates_queue;
//				MPI_Allreduce(&end_got_candidates_queue,
//						&global_end_got_candidates_queue,
//						1,
//						V_ID_Type,
//						MPI_SUM,
//						MPI_COMM_WORLD);
//				if (0 == host_id) {
//					printf("iter %u @%u host_id: %u global_end_got_candidates_queue: %u\n", iter, __LINE__, host_id, global_end_got_candidates_queue);
//				}
//			}
		}

        // Traverse vertices in the got_candidates_queue to insert labels
		{
            std::vector< std::pair<VertexID, VertexID> > buffer_send; // For sync elements in the dist_table
                // pair.first: root id
                // pair.second: label (global) id of the root
            for (VertexID i_queue = 0; i_queue < end_got_candidates_queue; ++i_queue) {
                VertexID v_id_local = got_candidates_queue[i_queue];
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

            // Sync the dist_table
            std::vector<MPI_Request> requests_send(num_hosts - 1);
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

            std::vector< std::pair<VertexID, VertexID> > buffer_recv;
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
//            {// test
//                if (0 == host_id) {
//                    printf("iter: %u @%u host_id: %u global_num_actives: %u\n", iter, __LINE__, host_id, global_num_actives);//test
//                }
//            }
		}
    }

    // Reset the dist_table
    reset_at_end(
            G,
            roots_start,
            roots_master_local,
            dist_table,
            recved_dist_table,
            bp_labels_table);
}

// Function: Distance query of a pair of vertices, used for distrubuted version.
template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
UnweightedDist DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::dist_distance_query_pair(
        VertexID a_input,
        VertexID b_input,
        const DistGraph &G)
{
    struct TmpMsgBPLabel {
        UnweightedDist bp_dist[BITPARALLEL_SIZE];
        uint64_t bp_sets[BITPARALLEL_SIZE][2];

        TmpMsgBPLabel() = default;
        TmpMsgBPLabel(const UnweightedDist dist[], const uint64_t sets[][2])
        {
            memcpy(bp_dist, dist, sizeof(bp_dist));
            memcpy(bp_sets, sets, sizeof(bp_sets));
        }
    };

    VertexID a_global = G.rank[a_input];
    VertexID b_global = G.rank[b_input];
    VertexID a_host_id = G.get_master_host_id(a_global);
    VertexID b_host_id = G.get_master_host_id(b_global);
    UnweightedDist min_d = MAX_UNWEIGHTED_DIST;

    // Both local
    if (a_host_id == host_id && b_host_id == host_id) {
        VertexID a_local = G.get_local_vertex_id(a_global);
        VertexID b_local = G.get_local_vertex_id(b_global);
        // Check Bit-Parallel Labels first
        {
            const IndexType &La = L[a_local];
            const IndexType &Lb = L[b_local];
            for (VertexID i = 0; i < BITPARALLEL_SIZE; ++i) {
                VertexID td = La.bp_dist[i] + Lb.bp_dist[i];
                if (td - 2 <= min_d) {
                    td +=
                            (La.bp_sets[i][0] & Lb.bp_sets[i][0]) ? -2 :
                            ((La.bp_sets[i][0] & Lb.bp_sets[i][1]) |
                             (La.bp_sets[i][1] & Lb.bp_sets[i][0]))
                            ? -1 : 0;
                    if (td < min_d) {
                        min_d = td;
                    }
                }
            }
        }

        std::map<VertexID, UnweightedDist> markers;
        // Traverse a's labels
        {
            const IndexType &Lr = L[a_local];
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
            const IndexType &Lr = L[b_local];
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
            VertexID b_local = G.get_local_vertex_id(b_global);
            const IndexType &Lr = L[b_local];
            // Bit-Parallel Labels
            {
                TmpMsgBPLabel msg_send(Lr.bp_dist, Lr.bp_sets);
                MPI_Send(&msg_send,
                        sizeof(msg_send),
                        MPI_CHAR,
                        a_host_id,
                        SENDING_QUERY_BP_LABELS,
                        MPI_COMM_WORLD);
            }
            // Normal Labels
            {
                std::vector<std::pair<VertexID, UnweightedDist> > buffer_send;
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
            }
        } else if (host_id == a_host_id) {
            VertexID a_local = G.get_local_vertex_id(a_global);
            const IndexType &Lr = L[a_local];
            // Receive BP labels
            {
                TmpMsgBPLabel msg_recv;
                MPI_Recv(&msg_recv,
                        sizeof(msg_recv),
                        MPI_CHAR,
                        b_host_id,
                        SENDING_QUERY_BP_LABELS,
                        MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE);
                for (VertexID i = 0; i < BITPARALLEL_SIZE; ++i) {
                    VertexID td = Lr.bp_dist[i] + msg_recv.bp_dist[i];
                    if (td - 2 <= min_d) {
                        td +=
                                (Lr.bp_sets[i][0] & msg_recv.bp_sets[i][0]) ? -2 :
                                ((Lr.bp_sets[i][0] & msg_recv.bp_sets[i][1]) |
                                 (Lr.bp_sets[i][1] & msg_recv.bp_sets[i][0]))
                                ? -1 : 0;
                        if (td < min_d) {
                            min_d = td;
                        }
                    }
                }
            }
            std::map<VertexID, UnweightedDist> markers;
            // Traverse a's labels
            {
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
