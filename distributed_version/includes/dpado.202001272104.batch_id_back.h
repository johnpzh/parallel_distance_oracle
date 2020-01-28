//
// Created by Zhen Peng on 1/6/20.
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
#include <omp.h>
#include "globals.h"
#include "dglobals.h"
#include "dgraph.h"


namespace PADO {

template <VertexID BATCH_SIZE = 1024>
class DistBVCPLL {
private:
    static const VertexID BITPARALLEL_SIZE = 50;
    const inti THRESHOLD_PARALLEL = 80;
    // Structure for the type of label
    struct IndexType {
        struct Batch {
//            VertexID batch_id; // Batch ID
            VertexID start_index; // Index to the array distances where the batch starts
            VertexID size; // Number of distances element in this batch

            Batch() = default;
            Batch(VertexID start_index_, VertexID size_):
                    start_index(start_index_), size(size_)
            { }
//            Batch(VertexID batch_id_, VertexID start_index_, VertexID size_):
//                    batch_id(batch_id_), start_index(start_index_), size(size_)
//            { }
        };

        struct DistanceIndexType {
            VertexID start_index; // Index to the array vertices where the same-distance vertices start
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

        size_t get_size_in_bytes() const
        {
            return sizeof(bp_dist) +
                    sizeof(bp_sets) +
//                    batches.size() * sizeof(Batch) +
                    distances.size() * sizeof(DistanceIndexType) +
                    vertices.size() * sizeof(VertexID);
        }

        void clean_all_indices()
        {
            std::vector<Batch>().swap(batches);
            std::vector<DistanceIndexType>().swap(distances);
            std::vector<VertexID>().swap(vertices);
        }

    }; //__attribute__((aligned(64)));

    struct ShortIndex {
        // I use BATCH_SIZE + 1 bit for indicator bit array.
        // The v.indicator[BATCH_SIZE] is set if in current batch v has got any new labels already.
        // In this way, it helps update_label_indices() and can be reset along with other indicator elements.
//        std::bitset<BATCH_SIZE + 1> indicator; // Global indicator, indicator[r] (0 <= r < BATCH_SIZE) is set means root r once selected as candidate already
        // If the Batch structure is not used, the indicator could just be BATCH_SIZE long.
//        std::vector<uint8_t> indicator = std::vector<uint8_t>(BATCH_SIZE, 0);
        std::vector<uint8_t> indicator = std::vector<uint8_t>(BATCH_SIZE + 1, 0);

        // Use a queue to store candidates
        std::vector<VertexID> candidates_que = std::vector<VertexID>(BATCH_SIZE);
        VertexID end_candidates_que = 0;
        std::vector<uint8_t> is_candidate = std::vector<uint8_t>(BATCH_SIZE, 0);

        void indicator_reset()
        {
            std::fill(indicator.begin(), indicator.end(), 0);
        }
    }; //__attribute__((aligned(64)));

    // Type of Bit-Parallel Label
    struct BPLabelType {
        UnweightedDist bp_dist[BITPARALLEL_SIZE] = { 0 };
        uint64_t bp_sets[BITPARALLEL_SIZE][2] = { {0} }; // [0]: S^{-1}, [1]: S^{0}
    };

    // Type of Label Message Unit, for initializing distance table
    struct LabelTableUnit {
        VertexID root_id;
        VertexID label_global_id;
        UnweightedDist dist;

        LabelTableUnit() = default;

        LabelTableUnit(VertexID r, VertexID l, UnweightedDist d) :
                root_id(r), label_global_id(l), dist(d) {}
    };

    // Type of BitParallel Label Message Unit for initializing bit-parallel labels
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

    VertexID num_v = 0;
    VertexID num_masters = 0;
//    VertexID BATCH_SIZE = 0;
    int host_id = 0;
    int num_hosts = 0;
    MPI_Datatype V_ID_Type;
    std::vector<IndexType> L;


    inline void bit_parallel_push_labels(
            const DistGraph &G,
            VertexID v_global,
//        std::vector<VertexID> &tmp_que,
//        VertexID &end_tmp_que,
//        std::vector< std::pair<VertexID, VertexID> > &sibling_es,
//        VertexID &num_sibling_es,
//        std::vector< std::pair<VertexID, VertexID> > &child_es,
//        VertexID &num_child_es,
            std::vector<VertexID> &tmp_q,
            VertexID &size_tmp_q,
            std::vector< std::pair<VertexID, VertexID> > &tmp_sibling_es,
            VertexID &size_tmp_sibling_es,
            std::vector< std::pair<VertexID, VertexID> > &tmp_child_es,
            VertexID &size_tmp_child_es,
            const VertexID &offset_tmp_q,
            std::vector<UnweightedDist> &dists,
            UnweightedDist iter);
    inline void bit_parallel_labeling(
            const DistGraph &G,
            std::vector<uint8_t> &used_bp_roots);

//    inline void bit_parallel_push_labels(
//            const DistGraph &G,
//            VertexID v_global,
//            std::vector<VertexID> &tmp_que,
//            VertexID &end_tmp_que,
//            std::vector< std::pair<VertexID, VertexID> > &sibling_es,
//            VertexID &num_sibling_es,
//            std::vector< std::pair<VertexID, VertexID> > &child_es,
//            VertexID &num_child_es,
//            std::vector<UnweightedDist> &dists,
//            UnweightedDist iter);
//    inline void bit_parallel_labeling(
//            const DistGraph &G,
////            std::vector<IndexType> &L,
//            std::vector<uint8_t> &used_bp_roots);

    inline void batch_process(
            const DistGraph &G,
//            const VertexID b_id,
            const VertexID roots_start,
            const VertexID roots_size,
            const std::vector<uint8_t> &used_bp_roots,
            std::vector<VertexID> &active_queue,
            VertexID &end_active_queue,
            std::vector<VertexID> &got_candidates_queue,
            VertexID &end_got_candidates_queue,
            std::vector<ShortIndex> &short_index,
            std::vector< std::vector<UnweightedDist> > &dist_table,
            std::vector< std::vector<VertexID> > &recved_dist_table,
            std::vector<BPLabelType> &bp_labels_table,
            std::vector<uint8_t> &got_candidates,
//        std::vector<bool> &got_candidates,
            std::vector<uint8_t> &is_active,
//        std::vector<bool> &is_active,
            std::vector<VertexID> &once_candidated_queue,
            VertexID &end_once_candidated_queue,
            std::vector<uint8_t> &once_candidated);
//            std::vector<bool> &once_candidated);
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
            std::vector<uint8_t> &once_candidated,
//            std::vector<bool> &once_candidated,
//            VertexID b_id,
            VertexID roots_start,
            VertexID roots_size,
//            std::vector<VertexID> &roots_master_local,
            const std::vector<uint8_t> &used_bp_roots);
//    inline void push_single_label(
//            VertexID v_head_global,
//            VertexID label_root_id,
//            VertexID roots_start,
//            const DistGraph &G,
//            std::vector<ShortIndex> &short_index,
//            std::vector<VertexID> &got_candidates_queue,
//            VertexID &end_got_candidates_queue,
//            std::vector<bool> &got_candidates,
//            std::vector<VertexID> &once_candidated_queue,
//            VertexID &end_once_candidated_queue,
//            std::vector<bool> &once_candidated,
//            const std::vector<BPLabelType> &bp_labels_table,
//            const std::vector<uint8_t> &used_bp_roots,
//            UnweightedDist iter);
    inline void schedule_label_pushing_para(
            const DistGraph &G,
            const VertexID roots_start,
            const std::vector<uint8_t> &used_bp_roots,
            const std::vector<VertexID> &active_queue,
            const VertexID global_start,
            const VertexID global_size,
            const VertexID local_size,
//            const VertexID start_active_queue,
//            const VertexID size_active_queue,
            std::vector<VertexID> &got_candidates_queue,
            VertexID &end_got_candidates_queue,
            std::vector<ShortIndex> &short_index,
            const std::vector<BPLabelType> &bp_labels_table,
            std::vector<uint8_t> &got_candidates,
            std::vector<uint8_t> &is_active,
            std::vector<VertexID> &once_candidated_queue,
            VertexID &end_once_candidated_queue,
            std::vector<uint8_t> &once_candidated,
            const UnweightedDist iter);
    inline void local_push_labels_seq(
            VertexID v_head_global,
            EdgeID start_index,
            EdgeID bound_index,
            VertexID roots_start,
            const std::vector<VertexID> &labels_buffer,
            const DistGraph &G,
            std::vector<ShortIndex> &short_index,
            std::vector<VertexID> &got_candidates_queue,
            VertexID &end_got_candidates_queue,
            std::vector<uint8_t> &got_candidates,
//            std::vector<bool> &got_candidates,
            std::vector<VertexID> &once_candidated_queue,
            VertexID &end_once_candidated_queue,
            std::vector<uint8_t> &once_candidated,
//            std::vector<bool> &once_candidated,
            const std::vector<BPLabelType> &bp_labels_table,
            const std::vector<uint8_t> &used_bp_roots,
            const UnweightedDist iter);
    inline void local_push_labels_para(
            const VertexID v_head_global,
            const EdgeID start_index,
            const EdgeID bound_index,
            const VertexID roots_start,
            const std::vector<VertexID> &labels_buffer,
            const DistGraph &G,
            std::vector<ShortIndex> &short_index,
//        std::vector<VertexID> &got_candidates_queue,
//        VertexID &end_got_candidates_queue,
            std::vector<VertexID> &tmp_got_candidates_queue,
            VertexID &size_tmp_got_candidates_queue,
            const VertexID offset_tmp_queue,
            std::vector<uint8_t> &got_candidates,
//        std::vector<VertexID> &once_candidated_queue,
//        VertexID &end_once_candidated_queue,
            std::vector<VertexID> &tmp_once_candidated_queue,
            VertexID &size_tmp_once_candidated_queue,
            std::vector<uint8_t> &once_candidated,
            const std::vector<BPLabelType> &bp_labels_table,
            const std::vector<uint8_t> &used_bp_roots,
            const UnweightedDist iter);
//    inline void local_push_labels(
//            VertexID v_head_local,
//            VertexID roots_start,
//            const DistGraph &G,
//            std::vector<ShortIndex> &short_index,
//            std::vector<VertexID> &got_candidates_queue,
//            VertexID &end_got_candidates_queue,
//            std::vector<bool> &got_candidates,
//            std::vector<VertexID> &once_candidated_queue,
//            VertexID &end_once_candidated_queue,
//            std::vector<bool> &once_candidated,
//            const std::vector<BPLabelType> &bp_labels_table,
//            const std::vector<uint8_t> &used_bp_roots,
//            UnweightedDist iter);
    inline void schedule_label_inserting_para(
            const DistGraph &G,
            const VertexID roots_start,
            const VertexID roots_size,
            std::vector<ShortIndex> &short_index,
            const std::vector< std::vector<UnweightedDist> > &dist_table,
            const std::vector<VertexID> &got_candidates_queue,
            const VertexID start_got_candidates_queue,
            const VertexID size_got_candidates_queue,
            std::vector<uint8_t> &got_candidates,
            std::vector<VertexID> &active_queue,
            VertexID &end_active_queue,
            std::vector<uint8_t> &is_active,
            std::vector< std::pair<VertexID, VertexID> > &buffer_send,
            const VertexID iter);
    inline bool distance_query(
            VertexID cand_root_id,
            VertexID v_id,
            VertexID roots_start,
//            const std::vector<IndexType> &L,
            const std::vector< std::vector<UnweightedDist> > &dist_table,
            UnweightedDist iter);
    inline void insert_label_only_seq(
            VertexID cand_root_id,
//            VertexID cand_root_id,
            VertexID v_id_local,
            VertexID roots_start,
            VertexID roots_size,
            const DistGraph &G,
//            std::vector< std::vector<UnweightedDist> > &dist_table,
            std::vector< std::pair<VertexID, VertexID> > &buffer_send);
//            UnweightedDist iter);
    inline void insert_label_only_para(
            VertexID cand_root_id,
            VertexID v_id_local,
            VertexID roots_start,
            VertexID roots_size,
            const DistGraph &G,
//        std::vector< std::pair<VertexID, VertexID> > &buffer_send)
            std::vector< std::pair<VertexID, VertexID> > &tmp_buffer_send,
            EdgeID &size_tmp_buffer_send,
            const EdgeID offset_tmp_buffer_send);
    inline void update_label_indices(
            const VertexID v_id,
            const VertexID inserted_count,
//            std::vector<IndexType> &L,
            std::vector<ShortIndex> &short_index,
//            VertexID b_id,
            const UnweightedDist iter);
    inline void reset_at_end(
            const DistGraph &G,
//            VertexID roots_start,
//            const std::vector<VertexID> &roots_master_local,
            std::vector< std::vector<UnweightedDist> > &dist_table,
            std::vector< std::vector<VertexID> > &recved_dist_table,
            std::vector<BPLabelType> &bp_labels_table,
            const std::vector<VertexID> &once_candidated_queue,
            const VertexID end_once_candidated_queue);
//    template <typename E_T, typename F>
//    inline void every_host_bcasts_buffer_and_proc(
//            std::vector<E_T> &buffer_send,
//            F &fun);
    template <typename E_T>
    inline void one_host_bcasts_buffer_to_buffer(
            int root,
            std::vector<E_T> &buffer_send,
            std::vector<E_T> &buffer_recv);

//    // Function: get the destination host id which is i hop from this host.
//    // For example, 1 hop from host 2 is host 0 (assume total 3 hosts);
//    // -1 hop from host 0 is host 2.
//    int hop_2_me_host_id(int hop) const
//    {
//        assert(hop >= -(num_hosts - 1) && hop < num_hosts && hop != 0);
//        return (host_id + hop + num_hosts) % num_hosts;
//    }
//    // Function: get the destination host id which is i hop from the root.
//    // For example, 1 hop from host 2 is host 0 (assume total 3 hosts);
//    // -1 hop from host 0 is host 2.
//    int hop_2_root_host_id(int hop, int root) const
//    {
//        assert(hop >= -(num_hosts - 1) && hop < num_hosts && hop != 0);
//        assert(root >= 0 && root < num_hosts);
//        return (root + hop + num_hosts) % num_hosts;
//    }

    size_t get_index_size()
    {
        size_t bytes = 0;
        for (VertexID v_i = 0; v_i < num_masters; ++v_i) {
            bytes += L[v_i].get_size_in_bytes();
        }
        return bytes;
    }


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
//    double message_time = 0;
//    double bp_labeling_time = 0;
//    double initializing_time = 0;
//    double scatter_time = 0;
//    double gather_time = 0;
//    double clearup_time = 0;

//	TotalInstructsExe candidating_ins_count;
//	TotalInstructsExe adding_ins_count;
//	TotalInstructsExe bp_labeling_ins_count;
//	TotalInstructsExe bp_checking_ins_count;
//	TotalInstructsExe dist_query_ins_count;
//    uint64_t caller_line = 0;
    // End test



public:
//    std::pair<uint64_t, uint64_t> length_larger_than_16 = std::make_pair(0, 0);
    DistBVCPLL() = default;
    explicit DistBVCPLL(
            const DistGraph &G);

//    UnweightedDist dist_distance_query_pair(
//            VertexID a_global,
//            VertexID b_global,
//            const DistGraph &G);

}; // class DistBVCPLL

template <VertexID BATCH_SIZE>
DistBVCPLL<BATCH_SIZE>::
DistBVCPLL(
        const DistGraph &G)
{
    num_v = G.num_v;
    assert(num_v >= BATCH_SIZE);
    num_masters = G.num_masters;
    host_id = G.host_id;
//    {
//        if (1 == host_id) {
//            volatile int i = 0;
//            while (i == 0) {
//                sleep(5);
//            }
//        }
//    }
    num_hosts = G.num_hosts;
    V_ID_Type = G.V_ID_Type;
//    L.resize(num_v);
    L.resize(num_masters);
    VertexID remainer = num_v % BATCH_SIZE;
    VertexID b_i_bound = num_v / BATCH_SIZE;
    std::vector<uint8_t> used_bp_roots(num_v, 0);
    //cache_miss.measure_start();
    double time_labeling = -WallTimer::get_time_mark();

//    bp_labeling_time -= WallTimer::get_time_mark();
    bit_parallel_labeling(G,
            used_bp_roots);
//    bp_labeling_time += WallTimer::get_time_mark();
    {//test
//#ifdef DEBUG_MESSAGES_ON
        if (0 == host_id) {
            printf("host_id: %u bp_labeling_finished.\n", host_id);
        }
//#endif
    }

    std::vector<VertexID> active_queue(num_masters); // Any vertex v who is active should be put into this queue.
    VertexID end_active_queue = 0;
    std::vector<uint8_t> is_active(num_masters, false);// is_active[v] is true means vertex v is in the active queue.
//    std::vector<bool> is_active(num_masters, false);// is_active[v] is true means vertex v is in the active queue.
    std::vector<VertexID> got_candidates_queue(num_masters); // Any vertex v who got candidates should be put into this queue.
    VertexID end_got_candidates_queue = 0;
    std::vector<uint8_t> got_candidates(num_masters, false); // got_candidates[v] is true means vertex v is in the queue got_candidates_queue
//    std::vector<bool> got_candidates(num_masters, false); // got_candidates[v] is true means vertex v is in the queue got_candidates_queue
    std::vector<ShortIndex> short_index(num_masters);
    std::vector< std::vector<UnweightedDist> > dist_table(BATCH_SIZE, std::vector<UnweightedDist>(num_v, MAX_UNWEIGHTED_DIST));
    std::vector<VertexID> once_candidated_queue(num_masters); // if short_index[v].indicator.any() is true, v is in the queue.
        // Used mainly for resetting short_index[v].indicator.
    VertexID end_once_candidated_queue = 0;
    std::vector<uint8_t> once_candidated(num_masters, false);
//    std::vector<bool> once_candidated(num_masters, false);

    std::vector< std::vector<VertexID> > recved_dist_table(BATCH_SIZE); // Some distances are from other hosts. This is used to reset the dist_table.
    std::vector<BPLabelType> bp_labels_table(BATCH_SIZE); // All roots' bit-parallel labels

    //printf("b_i_bound: %u\n", b_i_bound);//test
    for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
//        {// Batch number limit
//            if (10 == b_i) {
//                remainer = 0;
//                break;
//            }
//        }
////        {
//////#ifdef DEBUG_MESSAGES_ON
//            if (0 == host_id) {
//                printf("b_i: %u\n", b_i);//test
//            }
//////#endif
////        }

        batch_process(
                G,
//                b_i,
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
//        {
////#ifdef DEBUG_MESSAGES_ON
//            if (0 == host_id) {
//                printf("b_i: %u\n", b_i_bound);//test
//            }
////#endif
//        }
        batch_process(
                G,
//                b_i_bound,
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
        printf("BATCH_SIZE: %u ", BATCH_SIZE);
        printf("BP_Size: %u\n", BITPARALLEL_SIZE);
    }

    {// Total Number of Labels
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
//    printf("host_id: %u local_num_labels: %lu %.2f%%\n", host_id, local_num_labels, 100.0 * local_num_labels / global_num_labels);
        MPI_Barrier(MPI_COMM_WORLD);
        if (0 == host_id) {
            printf("Global_num_labels: %lu average: %f\n", global_num_labels, 1.0 * global_num_labels / num_v);
        }

//        VertexID local_num_batches = 0;
//        VertexID local_num_distances = 0;
////        double local_avg_distances_per_batches = 0;
//        for (VertexID v_global = 0; v_global < num_v; ++v_global) {
//            if (G.get_master_host_id(v_global) != host_id) {
//                continue;
//            }
//            VertexID v_local = G.get_local_vertex_id(v_global);
//            local_num_batches += L[v_local].batches.size();
//            local_num_distances += L[v_local].distances.size();
////            double avg_d_p_b = 0;
////            for (VertexID i_b = 0; i_b < L[v_local].batches.size(); ++i_b) {
////                avg_d_p_b += L[v_local].batches[i_b].size;
////            }
////            avg_d_p_b /= L[v_local].batches.size();
////            local_avg_distances_per_batches += avg_d_p_b;
//        }
////        local_avg_distances_per_batches /= num_masters;
////        double local_avg_batches = local_num_batches * 1.0 / num_masters;
////        double local_avg_distances = local_num_distances * 1.0 / num_masters;
//        uint64_t global_num_batches = 0;
//        uint64_t global_num_distances = 0;
//        MPI_Allreduce(
//                &local_num_batches,
//                &global_num_batches,
//                1,
//                MPI_UINT64_T,
//                MPI_SUM,
//                MPI_COMM_WORLD);
////        global_avg_batches /= num_hosts;
//        MPI_Allreduce(
//                &local_num_distances,
//                &global_num_distances,
//                1,
//                MPI_UINT64_T,
//                MPI_SUM,
//                MPI_COMM_WORLD);
////        global_avg_distances /= num_hosts;
//        double global_avg_d_p_b = global_num_distances * 1.0 / global_num_batches;
//        double global_avg_l_p_d = global_num_labels * 1.0 / global_num_distances;
//        double global_avg_batches = global_num_batches / num_v;
//        double global_avg_distances = global_num_distances / num_v;
////        MPI_Allreduce(
////                &local_avg_distances_per_batches,
////                &global_avg_d_p_b,
////                1,
////                MPI_DOUBLE,
////                MPI_SUM,
////                MPI_COMM_WORLD);
////        global_avg_d_p_b /= num_hosts;
//        MPI_Barrier(MPI_COMM_WORLD);
//        if (0 == host_id) {
//            printf("global_avg_batches: %f "
//                   "global_avg_distances: %f "
//                   "global_avg_distances_per_batch: %f "
//                   "global_avg_labels_per_distance: %f\n",
//                    global_avg_batches,
//                    global_avg_distances,
//                    global_avg_d_p_b,
//                    global_avg_l_p_d);
//        }
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

//    printf("num_hosts: %u host_id: %u\n"
//           "Local_labeling_time: %.2f seconds\n"
//           "bp_labeling_time: %.2f %.2f%%\n"
//           "initializing_time: %.2f %.2f%%\n"
//           "scatter_time: %.2f %.2f%%\n"
//           "gather_time: %.2f %.2f%%\n"
//           "clearup_time: %.2f %.2f%%\n"
//           "message_time: %.2f %.2f%%\n",
//           num_hosts, host_id,
//           time_labeling,
//           bp_labeling_time, 100.0 * bp_labeling_time / time_labeling,
//           initializing_time, 100.0 * initializing_time / time_labeling,
//           scatter_time, 100.0 * scatter_time / time_labeling,
//           gather_time, 100.0 * gather_time / time_labeling,
//           clearup_time, 100.0 * clearup_time / time_labeling,
//           message_time, 100.0 * message_time / time_labeling);
    double global_time_labeling;
    MPI_Allreduce(&time_labeling,
            &global_time_labeling,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == host_id) {
        printf("num_hosts: %d "
               "num_threads: %d "
               "Global_labeling_time: %.2f seconds\n",
               num_hosts,
               NUM_THREADS,
               global_time_labeling);
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


template <VertexID BATCH_SIZE>
inline void DistBVCPLL<BATCH_SIZE>::
bit_parallel_push_labels(
        const DistGraph &G,
        const VertexID v_global,
//        std::vector<VertexID> &tmp_que,
//        VertexID &end_tmp_que,
//        std::vector< std::pair<VertexID, VertexID> > &sibling_es,
//        VertexID &num_sibling_es,
//        std::vector< std::pair<VertexID, VertexID> > &child_es,
//        VertexID &num_child_es,
        std::vector<VertexID> &tmp_q,
        VertexID &size_tmp_q,
        std::vector< std::pair<VertexID, VertexID> > &tmp_sibling_es,
        VertexID &size_tmp_sibling_es,
        std::vector< std::pair<VertexID, VertexID> > &tmp_child_es,
        VertexID &size_tmp_child_es,
        const VertexID &offset_tmp_q,
        std::vector<UnweightedDist> &dists,
        const UnweightedDist iter)
{
    EdgeID i_start = G.vertices_idx[v_global];
    EdgeID i_bound = i_start + G.local_out_degrees[v_global];
//    {//test
//        printf("host_id: %u local_out_degrees[%u]: %u\n", host_id, v_global, G.local_out_degrees[v_global]);
//    }
    for (EdgeID i = i_start; i < i_bound; ++i) {
        VertexID tv_global = G.out_edges[i];
        VertexID tv_local = G.get_local_vertex_id(tv_global);
        UnweightedDist td = iter + 1;

        if (iter > dists[tv_local]) {
            ;
        } else if (iter == dists[tv_local]) {
            if (v_global < tv_global) { // ??? Why need v < tv !!! Because it's a undirected graph.
                tmp_sibling_es[offset_tmp_q + size_tmp_sibling_es].first = v_global;
                tmp_sibling_es[offset_tmp_q + size_tmp_sibling_es].second = tv_global;
                ++size_tmp_sibling_es;
//                sibling_es[num_sibling_es].first = v_global;
//                sibling_es[num_sibling_es].second = tv_global;
//                ++num_sibling_es;
            }
        } else { // iter < dists[tv]
            if (dists[tv_local] == MAX_UNWEIGHTED_DIST) {
                if (CAS(dists.data() + tv_local, MAX_UNWEIGHTED_DIST, td)) {
                    tmp_q[offset_tmp_q + size_tmp_q++] = tv_global;
                }
            }
//            if (dists[tv_local] == MAX_UNWEIGHTED_DIST) {
//                tmp_que[end_tmp_que++] = tv_global;
//                dists[tv_local] = td;
//            }
            tmp_child_es[offset_tmp_q + size_tmp_child_es].first = v_global;
            tmp_child_es[offset_tmp_q + size_tmp_child_es].second = tv_global;
            ++size_tmp_child_es;
//            child_es[num_child_es].first = v_global;
//            child_es[num_child_es].second = tv_global;
//            ++num_child_es;
        }
    }
}

template <VertexID BATCH_SIZE>
inline void DistBVCPLL<BATCH_SIZE>::
bit_parallel_labeling(
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
//        MsgUnitBP(MsgUnitBP&& other) = default;
//        MsgUnitBP(MsgUnitBP& other) = default;
//        MsgUnitBP& operator=(const MsgUnitBP& other) = default;
//        MsgUnitBP& operator=(MsgUnitBP&& other) = default;
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

    VertexID r_global = 0; // root r
    for (VertexID i_bpspt = 0; i_bpspt < BITPARALLEL_SIZE; ++i_bpspt) {
//        {// test
//            if (0 == host_id) {
//                printf("i_bpsp: %u\n", i_bpspt);
//            }
//        }
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
//        message_time -= WallTimer::get_time_mark();
        MPI_Bcast(&r_global,
                1,
                V_ID_Type,
                0,
                MPI_COMM_WORLD);
//        message_time += WallTimer::get_time_mark();
        used_bp_roots[r_global] = 1;
//#ifdef DEBUG_MESSAGES_ON
//        {//test
//            if (0 == host_id) {
//                printf("r_global: %u i_bpspt: %u\n", r_global, i_bpspt);
//            }
//        }
//#endif

//        VertexID que_t0 = 0, que_t1 = 0, que_h = 0;
        fill(tmp_d.begin(), tmp_d.end(), MAX_UNWEIGHTED_DIST);
        fill(tmp_s.begin(), tmp_s.end(), std::make_pair(0, 0));

        // Mark the r_global
        if (G.get_master_host_id(r_global) == host_id) {
            tmp_d[G.get_local_vertex_id(r_global)] = 0;
            que[end_que++] = r_global;
        }
        // Select the r_global's 64 neighbors
        {
            // Get r_global's neighbors into buffer_send, rank from high to low.
            VertexID local_degree = G.local_out_degrees[r_global];
            std::vector<VertexID> buffer_send(local_degree);
            if (local_degree) {
                EdgeID e_i_start = G.vertices_idx[r_global] + local_degree - 1;
                for (VertexID d_i = 0; d_i < local_degree; ++d_i) {
                    EdgeID e_i = e_i_start - d_i;
                    buffer_send[d_i] = G.out_edges[e_i];
                }
            }

            // Get selected neighbors (up to 64)
            std::vector<VertexID> selected_nbrs;
            if (0 != host_id) {
                // Every host other than 0 sends neighbors to host 0
//                message_time -= WallTimer::get_time_mark();
                MPI_Instance::send_buffer_2_dst(buffer_send,
                        0,
                        SENDING_ROOT_NEIGHBORS,
                        SENDING_SIZE_ROOT_NEIGHBORS);
                // Receive selected neighbors from host 0
                MPI_Instance::recv_buffer_from_src(selected_nbrs,
                        0,
                        SENDING_SELECTED_NEIGHBORS,
                        SENDING_SIZE_SELETED_NEIGHBORS);
//                message_time += WallTimer::get_time_mark();
            } else {
                // Host 0
                // Host 0 receives neighbors from others
                std::vector<VertexID> all_nbrs(buffer_send);
                std::vector<VertexID > buffer_recv;
                for (int loc = 0; loc < num_hosts - 1; ++loc) {
//                    message_time -= WallTimer::get_time_mark();
                    MPI_Instance::recv_buffer_from_any(buffer_recv,
                                                       SENDING_ROOT_NEIGHBORS,
                                                       SENDING_SIZE_ROOT_NEIGHBORS);
//                    message_time += WallTimer::get_time_mark();
                    if (buffer_recv.empty()) {
                        continue;
                    }

                    buffer_send.resize(buffer_send.size() + buffer_recv.size());
                    std::merge(buffer_recv.begin(), buffer_recv.end(), all_nbrs.begin(), all_nbrs.end(), buffer_send.begin());
                    all_nbrs.resize(buffer_send.size());
                    all_nbrs.assign(buffer_send.begin(), buffer_send.end());
                }
                assert(all_nbrs.size() == G.get_global_out_degree(r_global));
                // Select 64 (or less) neighbors
                VertexID ns = 0; // number of selected neighbor, default 64
                for (VertexID v_global : all_nbrs) {
                    if (used_bp_roots[v_global]) {
                        continue;
                    }
                    used_bp_roots[v_global] = 1;
                    selected_nbrs.push_back(v_global);
                    if (++ns == 64) {
                        break;
                    }
                }
                // Send selected neighbors to other hosts
//                message_time -= WallTimer::get_time_mark();
                for (int dest = 1; dest < num_hosts; ++dest) {
                    MPI_Instance::send_buffer_2_dst(selected_nbrs,
                            dest,
                            SENDING_SELECTED_NEIGHBORS,
                            SENDING_SIZE_SELETED_NEIGHBORS);
                }
//                message_time += WallTimer::get_time_mark();
            }
//            {//test
//                printf("host_id: %u selected_nbrs.size(): %lu\n", host_id, selected_nbrs.size());
//            }

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
        UnweightedDist d = 0;
        while (global_num_actives) {
            {// Limit the distance
                if (d > 7) {
                    break;
                }
            }
//#ifdef DEBUG_MESSAGES_ON
//            {//test
//                if (0 == host_id) {
//                    printf("d: %u que_size: %u\n", d, global_num_actives);
//                }
//            }
//#endif
//            for (UnweightedDist d = 0; que_t0 < que_h; ++d) {
            VertexID num_sibling_es = 0, num_child_es = 0;


            // Send active masters to mirrors
            {
                std::vector<MsgUnitBP> buffer_send(end_que);
                for (VertexID que_i = 0; que_i < end_que; ++que_i) {
                    VertexID v_global = que[que_i];
                    buffer_send[que_i] = MsgUnitBP(v_global, tmp_s[v_global].first, tmp_s[v_global].second);
                }
//                {// test
//                    printf("host_id: %u buffer_send.size(): %lu\n", host_id, buffer_send.size());
//                }

                for (int root = 0; root < num_hosts; ++root) {
                    std::vector<MsgUnitBP> buffer_recv;
                    one_host_bcasts_buffer_to_buffer(root,
                            buffer_send,
                            buffer_recv);
                    if (buffer_recv.empty()) {
                        continue;
                    }

                    // For parallel adding to queue
                    VertexID size_buffer_recv = buffer_recv.size();
                    std::vector<VertexID> offsets_tmp_q(size_buffer_recv);
#pragma omp parallel for
                    for (VertexID i_q = 0; i_q < size_buffer_recv; ++i_q) {
                        offsets_tmp_q[i_q] = G.local_out_degrees[buffer_recv[i_q].v_global];
                    }
                    VertexID num_neighbors = PADO::prefix_sum_for_offsets(offsets_tmp_q);
                    std::vector<VertexID> tmp_q(num_neighbors);
                    std::vector<VertexID> sizes_tmp_q(size_buffer_recv, 0);
                    // For parallel adding to sibling_es
                    std::vector< std::pair<VertexID, VertexID> > tmp_sibling_es(num_neighbors);
                    std::vector<VertexID> sizes_tmp_sibling_es(size_buffer_recv, 0);
                    // For parallel adding to child_es
                    std::vector< std::pair<VertexID, VertexID> > tmp_child_es(num_neighbors);
                    std::vector<VertexID> sizes_tmp_child_es(size_buffer_recv, 0);

#pragma omp parallel for
//                    for (const MsgUnitBP &m : buffer_recv) {
                    for (VertexID i_m = 0; i_m < size_buffer_recv; ++i_m) {
                        const MsgUnitBP &m = buffer_recv[i_m];
                        VertexID v_global = m.v_global;
                        if (!G.local_out_degrees[v_global]) {
                            continue;
                        }
                        tmp_s[v_global].first = m.S_n1;
                        tmp_s[v_global].second = m.S_0;
                        // Push labels
                        bit_parallel_push_labels(
                                G,
                                v_global,
                                tmp_q,
                                sizes_tmp_q[i_m],
                                tmp_sibling_es,
                                sizes_tmp_sibling_es[i_m],
                                tmp_child_es,
                                sizes_tmp_child_es[i_m],
                                offsets_tmp_q[i_m],
//                                                 tmp_que,
//                                                 end_tmp_que,
//                                                 sibling_es,
//                                                 num_sibling_es,
//                                                 child_es,
//                                                 num_child_es,
                                tmp_d,
                                d);
                    }

                    {// From tmp_sibling_es to sibling_es
                        idi total_size_tmp = PADO::prefix_sum_for_offsets(sizes_tmp_sibling_es);
                        PADO::collect_into_queue(
                                tmp_sibling_es,
                                offsets_tmp_q,
                                sizes_tmp_sibling_es,
                                total_size_tmp,
                                sibling_es,
                                num_sibling_es);
                    }

                    {// From tmp_child_es to child_es
                        idi total_size_tmp = PADO::prefix_sum_for_offsets(sizes_tmp_child_es);
                        PADO::collect_into_queue(
                                tmp_child_es,
                                offsets_tmp_q,
                                sizes_tmp_child_es,
                                total_size_tmp,
                                child_es,
                                num_child_es);
                    }

                    {// From tmp_q to tmp_que
                        idi total_size_tmp = PADO::prefix_sum_for_offsets(sizes_tmp_q);
                        PADO::collect_into_queue(
                                tmp_q,
                                offsets_tmp_q,
                                sizes_tmp_q,
                                total_size_tmp,
                                tmp_que,
                                end_tmp_que);
                    }

//                    {// test
//                        printf("host_id: %u root: %u done push.\n", host_id, root);
//                    }
                }
            }

            // Update the sets in tmp_s
            {
#pragma omp parallel for
                for (VertexID i = 0; i < num_sibling_es; ++i) {
                    VertexID v = sibling_es[i].first, w = sibling_es[i].second;
                    __atomic_or_fetch(&tmp_s[v].second, tmp_s[w].first, __ATOMIC_SEQ_CST);
                    __atomic_or_fetch(&tmp_s[w].second, tmp_s[v].first, __ATOMIC_SEQ_CST);
//                    tmp_s[v].second |= tmp_s[w].first; // !!! Need to send back!!!
//                    tmp_s[w].second |= tmp_s[v].first;
                }
                // Put into the buffer sending to others
                std::vector< std::pair<VertexID, uint64_t> > buffer_send(2 * num_sibling_es);
#pragma omp parallel for
                for (VertexID i = 0; i < num_sibling_es; ++i) {
                    VertexID v = sibling_es[i].first;
                    VertexID w = sibling_es[i].second;
                    buffer_send[2 * i] = std::make_pair(v, tmp_s[v].second);
                    buffer_send[2 * i + 1] = std::make_pair(w, tmp_s[w].second);
                }
                // Send the messages
                for (int root = 0; root < num_hosts; ++root) {
                    std::vector< std::pair<VertexID, uint64_t> > buffer_recv;
                    one_host_bcasts_buffer_to_buffer(root,
                                                     buffer_send,
                                                     buffer_recv);
                    if (buffer_recv.empty()) {
                        continue;
                    }
                    size_t i_m_bound = buffer_recv.size();
#pragma omp parallel for
                    for (size_t i_m = 0; i_m < i_m_bound; ++i_m) {
                        const auto &m = buffer_recv[i_m];
                        __atomic_or_fetch(&tmp_s[m.first].second, m.second, __ATOMIC_SEQ_CST);
                    }
//                    for (const std::pair<VertexID, uint64_t> &m : buffer_recv) {
//                        tmp_s[m.first].second |= m.second;
//                    }
                }
#pragma omp parallel for
                for (VertexID i = 0; i < num_child_es; ++i) {
                    VertexID v = child_es[i].first, c = child_es[i].second;
                    __atomic_or_fetch(&tmp_s[c].first, tmp_s[v].first, __ATOMIC_SEQ_CST);
                    __atomic_or_fetch(&tmp_s[c].second, tmp_s[v].second, __ATOMIC_SEQ_CST);
//                    tmp_s[c].first |= tmp_s[v].first;
//                    tmp_s[c].second |= tmp_s[v].second;
                }
            }
//#ifdef DEBUG_MESSAGES_ON
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
//                    printf("iter: %u num_sibling_es: %u num_child_es: %u\n", d, global_num_sibling_es, global_num_child_es);
//                }
//
////                printf("iter %u @%u host_id: %u num_sibling_es: %u num_child_es: %u\n", d, __LINE__, host_id, num_sibling_es, num_child_es);
////                if (0 == d) {
////                    exit(EXIT_SUCCESS);
////                }
//            }
//#endif

            // Swap que and tmp_que
            tmp_que.swap(que);
            end_que = end_tmp_que;
            end_tmp_que = 0;
            MPI_Allreduce(&end_que,
                      &global_num_actives,
                      1,
                      V_ID_Type,
                      MPI_MAX,
                      MPI_COMM_WORLD);

//            }
            ++d;
        }

#pragma omp parallel for
        for (VertexID v_local = 0; v_local < num_masters; ++v_local) {
            VertexID v_global = G.get_global_vertex_id(v_local);
            L[v_local].bp_dist[i_bpspt] = tmp_d[v_local];
            L[v_local].bp_sets[i_bpspt][0] = tmp_s[v_global].first; // S_r^{-1}
            L[v_local].bp_sets[i_bpspt][1] = tmp_s[v_global].second & ~tmp_s[v_global].first; // Only need those r's neighbors who are not already in S_r^{-1}
        }
    }
}

//template <VertexID BATCH_SIZE>
//inline void DistBVCPLL<BATCH_SIZE>::
//bit_parallel_push_labels(
//        const DistGraph &G,
//        const VertexID v_global,
//        std::vector<VertexID> &tmp_que,
//        VertexID &end_tmp_que,
//        std::vector< std::pair<VertexID, VertexID> > &sibling_es,
//        VertexID &num_sibling_es,
//        std::vector< std::pair<VertexID, VertexID> > &child_es,
//        VertexID &num_child_es,
//        std::vector<UnweightedDist> &dists,
//        const UnweightedDist iter)
//{
//    EdgeID i_start = G.vertices_idx[v_global];
//    EdgeID i_bound = i_start + G.local_out_degrees[v_global];
////    {//test
////        printf("host_id: %u local_out_degrees[%u]: %u\n", host_id, v_global, G.local_out_degrees[v_global]);
////    }
//    for (EdgeID i = i_start; i < i_bound; ++i) {
//        VertexID tv_global = G.out_edges[i];
//        VertexID tv_local = G.get_local_vertex_id(tv_global);
//        UnweightedDist td = iter + 1;
//
//        if (iter > dists[tv_local]) {
//            ;
//        } else if (iter == dists[tv_local]) {
//            if (v_global < tv_global) { // ??? Why need v < tv !!! Because it's a undirected graph.
//                sibling_es[num_sibling_es].first = v_global;
//                sibling_es[num_sibling_es].second = tv_global;
//                ++num_sibling_es;
//            }
//        } else { // iter < dists[tv]
//            if (dists[tv_local] == MAX_UNWEIGHTED_DIST) {
//                tmp_que[end_tmp_que++] = tv_global;
//                dists[tv_local] = td;
//            }
//            child_es[num_child_es].first = v_global;
//            child_es[num_child_es].second = tv_global;
//            ++num_child_es;
////            {
////                printf("host_id: %u num_child_es: %u v_global: %u tv_global: %u\n", host_id, num_child_es, v_global, tv_global);//test
////            }
//        }
//    }
//
//}
//
//template <VertexID BATCH_SIZE>
//inline void DistBVCPLL<BATCH_SIZE>::
//bit_parallel_labeling(
//        const DistGraph &G,
////        std::vector<IndexType> &L,
//        std::vector<uint8_t> &used_bp_roots)
//{
//    // Class type of Bit-Parallel label message unit.
//    struct MsgUnitBP {
//        VertexID v_global;
//        uint64_t S_n1;
//        uint64_t S_0;
//
//        MsgUnitBP() = default;
////        MsgUnitBP(MsgUnitBP&& other) = default;
////        MsgUnitBP(MsgUnitBP& other) = default;
////        MsgUnitBP& operator=(const MsgUnitBP& other) = default;
////        MsgUnitBP& operator=(MsgUnitBP&& other) = default;
//        MsgUnitBP(VertexID v, uint64_t sn1, uint64_t s0)
//                : v_global(v), S_n1(sn1), S_0(s0) { }
//    };
////    VertexID num_v = G.num_v;
////    EdgeID num_e = G.num_e;
//    EdgeID local_num_edges = G.num_edges_local;
//
//    std::vector<UnweightedDist> tmp_d(num_masters); // distances from the root to every v
//    std::vector<std::pair<uint64_t, uint64_t> > tmp_s(num_v); // first is S_r^{-1}, second is S_r^{0}
//    std::vector<VertexID> que(num_masters); // active queue
//    VertexID end_que = 0;
//    std::vector<VertexID> tmp_que(num_masters); // temporary queue, to be swapped with que
//    VertexID end_tmp_que = 0;
//    std::vector<std::pair<VertexID, VertexID> > sibling_es(local_num_edges); // siblings, their distances to the root are equal (have difference of 0)
//    std::vector<std::pair<VertexID, VertexID> > child_es(local_num_edges); // child and father, their distances to the root have difference of 1.
//
////    std::vector<UnweightedDist> tmp_d(num_v); // distances from the root to every v
////    std::vector<std::pair<uint64_t, uint64_t> > tmp_s(num_v); // first is S_r^{-1}, second is S_r^{0}
////    std::vector<VertexID> que(num_v); // active queue
////    std::vector<std::pair<VertexID, VertexID> > sibling_es(num_e); // siblings, their distances to the root are equal (have difference of 0)
////    std::vector<std::pair<VertexID, VertexID> > child_es(num_e); // child and father, their distances to the root have difference of 1.
//
//    VertexID r_global = 0; // root r
//    for (VertexID i_bpspt = 0; i_bpspt < BITPARALLEL_SIZE; ++i_bpspt) {
//        // Select the root r_global
//        if (0 == host_id) {
//            while (r_global < num_v && used_bp_roots[r_global]) {
//                ++r_global;
//            }
//            if (r_global == num_v) {
//                for (VertexID v = 0; v < num_v; ++v) {
//                    L[v].bp_dist[i_bpspt] = MAX_UNWEIGHTED_DIST;
//                }
//                continue;
//            }
//        }
//        // Broadcast the r here.
//        message_time -= WallTimer::get_time_mark();
//        MPI_Bcast(&r_global,
//                  1,
//                  V_ID_Type,
//                  0,
//                  MPI_COMM_WORLD);
//        message_time += WallTimer::get_time_mark();
//        used_bp_roots[r_global] = 1;
//#ifdef DEBUG_MESSAGES_ON
//        {//test
//            if (0 == host_id) {
//                printf("r_global: %u i_bpspt: %u\n", r_global, i_bpspt);
//            }
//        }
//#endif
//
////        VertexID que_t0 = 0, que_t1 = 0, que_h = 0;
//        fill(tmp_d.begin(), tmp_d.end(), MAX_UNWEIGHTED_DIST);
//        fill(tmp_s.begin(), tmp_s.end(), std::make_pair(0, 0));
//
//        // Mark the r_global
//        if (G.get_master_host_id(r_global) == host_id) {
//            tmp_d[G.get_local_vertex_id(r_global)] = 0;
//            que[end_que++] = r_global;
//        }
//        // Select the r_global's 64 neighbors
//        {
//            // Get r_global's neighbors into buffer_send, rank from low to high.
//            VertexID local_degree = G.local_out_degrees[r_global];
//            std::vector<VertexID> buffer_send(local_degree);
//            if (local_degree) {
//                EdgeID e_i_start = G.vertices_idx[r_global] + local_degree - 1;
//                for (VertexID d_i = 0; d_i < local_degree; ++d_i) {
//                    EdgeID e_i = e_i_start - d_i;
//                    buffer_send[d_i] = G.out_edges[e_i];
//                }
//            }
//
//            // Get selected neighbors (up to 64)
//            std::vector<VertexID> selected_nbrs;
//            if (0 != host_id) {
//                // Every host other than 0 sends neighbors to host 0
//                message_time -= WallTimer::get_time_mark();
//                MPI_Instance::send_buffer_2_dst(buffer_send,
//                                                0,
//                                                SENDING_ROOT_NEIGHBORS,
//                                                SENDING_SIZE_ROOT_NEIGHBORS);
//                // Receive selected neighbors from host 0
//                MPI_Instance::recv_buffer_from_src(selected_nbrs,
//                                                   0,
//                                                   SENDING_SELECTED_NEIGHBORS,
//                                                   SENDING_SIZE_SELETED_NEIGHBORS);
//                message_time += WallTimer::get_time_mark();
//            } else {
//                // Host 0
//                // Host 0 receives neighbors from others
//                std::vector<VertexID> all_nbrs(buffer_send);
//                std::vector<VertexID > buffer_recv;
//                for (int loc = 0; loc < num_hosts - 1; ++loc) {
//                    message_time -= WallTimer::get_time_mark();
//                    MPI_Instance::recv_buffer_from_any(buffer_recv,
//                                                       SENDING_ROOT_NEIGHBORS,
//                                                       SENDING_SIZE_ROOT_NEIGHBORS);
////                    MPI_Instance::receive_dynamic_buffer_from_any(buffer_recv,
////                            num_hosts,
////                            SENDING_ROOT_NEIGHBORS);
//                    message_time += WallTimer::get_time_mark();
//                    if (buffer_recv.empty()) {
//                        continue;
//                    }
//
//                    buffer_send.resize(buffer_send.size() + buffer_recv.size());
//                    std::merge(buffer_recv.begin(), buffer_recv.end(), all_nbrs.begin(), all_nbrs.end(), buffer_send.begin());
//                    all_nbrs.resize(buffer_send.size());
//                    all_nbrs.assign(buffer_send.begin(), buffer_send.end());
//                }
//                assert(all_nbrs.size() == G.get_global_out_degree(r_global));
//                // Select 64 (or less) neighbors
//                VertexID ns = 0; // number of selected neighbor, default 64
//                for (VertexID v_global : all_nbrs) {
//                    if (used_bp_roots[v_global]) {
//                        continue;
//                    }
//                    used_bp_roots[v_global] = 1;
//                    selected_nbrs.push_back(v_global);
//                    if (++ns == 64) {
//                        break;
//                    }
//                }
//                // Send selected neighbors to other hosts
//                message_time -= WallTimer::get_time_mark();
//                for (int dest = 1; dest < num_hosts; ++dest) {
//                    MPI_Instance::send_buffer_2_dst(selected_nbrs,
//                                                    dest,
//                                                    SENDING_SELECTED_NEIGHBORS,
//                                                    SENDING_SIZE_SELETED_NEIGHBORS);
//                }
//                message_time += WallTimer::get_time_mark();
//            }
////            {//test
////                printf("host_id: %u selected_nbrs.size(): %lu\n", host_id, selected_nbrs.size());
////            }
//
//            // Synchronize the used_bp_roots.
//            for (VertexID v_global : selected_nbrs) {
//                used_bp_roots[v_global] = 1;
//            }
//
//            // Mark selected neighbors
//            for (VertexID v_i = 0; v_i < selected_nbrs.size(); ++v_i) {
//                VertexID v_global = selected_nbrs[v_i];
//                if (host_id != G.get_master_host_id(v_global)) {
//                    continue;
//                }
//                tmp_que[end_tmp_que++] = v_global;
//                tmp_d[G.get_local_vertex_id(v_global)] = 1;
//                tmp_s[v_global].first = 1ULL << v_i;
//            }
//        }
//
//        // Reduce the global number of active vertices
//        VertexID global_num_actives = 1;
//        UnweightedDist d = 0;
//        while (global_num_actives) {
////            for (UnweightedDist d = 0; que_t0 < que_h; ++d) {
//            VertexID num_sibling_es = 0, num_child_es = 0;
//
//
//            // Send active masters to mirrors
//            {
//                std::vector<MsgUnitBP> buffer_send(end_que);
//                for (VertexID que_i = 0; que_i < end_que; ++que_i) {
//                    VertexID v_global = que[que_i];
//                    buffer_send[que_i] = MsgUnitBP(v_global, tmp_s[v_global].first, tmp_s[v_global].second);
//                }
////                {// test
////                    printf("host_id: %u buffer_send.size(): %lu\n", host_id, buffer_send.size());
////                }
//
//                for (int root = 0; root < num_hosts; ++root) {
//                    std::vector<MsgUnitBP> buffer_recv;
//                    one_host_bcasts_buffer_to_buffer(root,
//                                                     buffer_send,
//                                                     buffer_recv);
//                    if (buffer_recv.empty()) {
//                        continue;
//                    }
//                    for (const MsgUnitBP &m : buffer_recv) {
//                        VertexID v_global = m.v_global;
//                        if (!G.local_out_degrees[v_global]) {
//                            continue;
//                        }
//                        tmp_s[v_global].first = m.S_n1;
//                        tmp_s[v_global].second = m.S_0;
//                        // Push labels
//                        bit_parallel_push_labels(G,
//                                                 v_global,
//                                                 tmp_que,
//                                                 end_tmp_que,
//                                                 sibling_es,
//                                                 num_sibling_es,
//                                                 child_es,
//                                                 num_child_es,
//                                                 tmp_d,
//                                                 d);
//                    }
////                    {// test
////                        printf("host_id: %u root: %u done push.\n", host_id, root);
////                    }
//                }
//            }
//
//            // Update the sets in tmp_s
//            {
//
//                for (VertexID i = 0; i < num_sibling_es; ++i) {
//                    VertexID v = sibling_es[i].first, w = sibling_es[i].second;
//                    tmp_s[v].second |= tmp_s[w].first; // !!! Need to send back!!!
//                    tmp_s[w].second |= tmp_s[v].first;
//
//                }
//                // Put into the buffer sending to others
//                std::vector< std::pair<VertexID, uint64_t> > buffer_send(2 * num_sibling_es);
////                std::vector< std::vector<MPI_Request> > requests_list(num_hosts - 1);
//                for (VertexID i = 0; i < num_sibling_es; ++i) {
//                    VertexID v = sibling_es[i].first;
//                    VertexID w = sibling_es[i].second;
////                    buffer_send.emplace_back(v, tmp_s[v].second);
////                    buffer_send.emplace_back(w, tmp_s[w].second);
//                    buffer_send[2 * i] = std::make_pair(v, tmp_s[v].second);
//                    buffer_send[2 * i + 1] = std::make_pair(w, tmp_s[w].second);
//                }
//                // Send the messages
//                for (int root = 0; root < num_hosts; ++root) {
//                    std::vector< std::pair<VertexID, uint64_t> > buffer_recv;
//                    one_host_bcasts_buffer_to_buffer(root,
//                                                     buffer_send,
//                                                     buffer_recv);
//                    if (buffer_recv.empty()) {
//                        continue;
//                    }
//                    for (const std::pair<VertexID, uint64_t> &m : buffer_recv) {
//                        tmp_s[m.first].second |= m.second;
//                    }
//                }
//                for (VertexID i = 0; i < num_child_es; ++i) {
//                    VertexID v = child_es[i].first, c = child_es[i].second;
//                    tmp_s[c].first |= tmp_s[v].first;
//                    tmp_s[c].second |= tmp_s[v].second;
//                }
//            }
////#ifdef DEBUG_MESSAGES_ON
//            {// test
//                VertexID global_num_sibling_es;
//                VertexID global_num_child_es;
//                MPI_Allreduce(&num_sibling_es,
//                              &global_num_sibling_es,
//                              1,
//                              V_ID_Type,
//                              MPI_SUM,
//                              MPI_COMM_WORLD);
//                MPI_Allreduce(&num_child_es,
//                              &global_num_child_es,
//                              1,
//                              V_ID_Type,
//                              MPI_SUM,
//                              MPI_COMM_WORLD);
//                if (0 == host_id) {
//                    printf("iter: %u num_sibling_es: %u num_child_es: %u\n", d, global_num_sibling_es, global_num_child_es);
//                }
//            }
////#endif
//
//            // Swap que and tmp_que
//            tmp_que.swap(que);
//            end_que = end_tmp_que;
//            end_tmp_que = 0;
//            MPI_Allreduce(&end_que,
//                          &global_num_actives,
//                          1,
//                          V_ID_Type,
//                          MPI_SUM,
//                          MPI_COMM_WORLD);
//
////            }
//            ++d;
//        }
//
//        for (VertexID v_local = 0; v_local < num_masters; ++v_local) {
//            VertexID v_global = G.get_global_vertex_id(v_local);
//            L[v_local].bp_dist[i_bpspt] = tmp_d[v_local];
//            L[v_local].bp_sets[i_bpspt][0] = tmp_s[v_global].first; // S_r^{-1}
//            L[v_local].bp_sets[i_bpspt][1] = tmp_s[v_global].second & ~tmp_s[v_global].first; // Only need those r's neighbors who are not already in S_r^{-1}
//        }
//    }
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
template <VertexID BATCH_SIZE>
inline VertexID DistBVCPLL<BATCH_SIZE>::
initialization(
        const DistGraph &G,
        std::vector<ShortIndex> &short_index,
        std::vector< std::vector<UnweightedDist> > &dist_table,
        std::vector< std::vector<VertexID> > &recved_dist_table,
        std::vector<BPLabelType> &bp_labels_table,
        std::vector<VertexID> &active_queue,
        VertexID &end_active_queue,
        std::vector<VertexID> &once_candidated_queue,
        VertexID &end_once_candidated_queue,
        std::vector<uint8_t> &once_candidated,
//        VertexID b_id,
        VertexID roots_start,
        VertexID roots_size,
//        std::vector<VertexID> &roots_master_local,
        const std::vector<uint8_t> &used_bp_roots)
{
    // Get the roots_master_local, containing all local roots.
    std::vector<VertexID> roots_master_local;
    VertexID size_roots_master_local;
    VertexID roots_bound = roots_start + roots_size;
    try {
        for (VertexID r_global = roots_start; r_global < roots_bound; ++r_global) {
            if (G.get_master_host_id(r_global) == host_id && !used_bp_roots[r_global]) {
                roots_master_local.push_back(G.get_local_vertex_id(r_global));
//                {//test
//                    if (1024 == roots_start && 7 == host_id && 31600 == *roots_master_local.rbegin()) {
//                        printf("S0.0 host_id: %d "
//                               "31600 YES!\n",
//                               host_id);
//                    }
//                }
            }
        }
        size_roots_master_local = roots_master_local.size();
    }
    catch (const std::bad_alloc &) {
        double memtotal = 0;
        double memfree = 0;
        PADO::Utils::system_memory(memtotal, memfree);
        printf("initialization_roots_master_local: bad_alloc "
               "host_id: %d "
               "L.size(): %.2fGB "
               "memtotal: %.2fGB "
               "memfree: %.2fGB\n",
               host_id,
               get_index_size() * 1.0 / (1 << 30),
               memtotal / 1024,
               memfree / 1024);
        exit(1);
    }

    // Short_index
    {
        if (end_once_candidated_queue >= THRESHOLD_PARALLEL) {
#pragma omp parallel for
            for (VertexID v_i = 0; v_i < end_once_candidated_queue; ++v_i) {
                VertexID v_local = once_candidated_queue[v_i];
                short_index[v_local].indicator_reset();
                once_candidated[v_local] = 0;
            }
        } else {
            for (VertexID v_i = 0; v_i < end_once_candidated_queue; ++v_i) {
                VertexID v_local = once_candidated_queue[v_i];
                short_index[v_local].indicator_reset();
                once_candidated[v_local] = 0;
            }
        }
        end_once_candidated_queue = 0;
        if (size_roots_master_local >= THRESHOLD_PARALLEL) {
#pragma omp parallel for
            for (VertexID i_r = 0; i_r < size_roots_master_local; ++i_r) {
                VertexID r_local = roots_master_local[i_r];
                short_index[r_local].indicator[G.get_global_vertex_id(r_local) - roots_start] = 1; // v itself
//                short_index[r_local].indicator[BATCH_SIZE] = 1; // v got labels
            }
        } else {
            for (VertexID r_local : roots_master_local) {
                short_index[r_local].indicator[G.get_global_vertex_id(r_local) - roots_start] = 1; // v itself
//                short_index[r_local].indicator[BATCH_SIZE] = 1; // v got labels
            }
        }
    }
//
    // Real Index
    try
    {
        if (size_roots_master_local >= THRESHOLD_PARALLEL) {
#pragma omp parallel for
            for (VertexID i_r = 0; i_r < size_roots_master_local; ++i_r) {
                VertexID r_local = roots_master_local[i_r];
                IndexType &Lr = L[r_local];
                Lr.batches.emplace_back(
//                        b_id, // Batch ID
                        Lr.distances.size(), // start_index
                        1); // size
                Lr.distances.emplace_back(
                        Lr.vertices.size(), // start_index
                        1, // size
                        0); // dist
                Lr.vertices.push_back(G.get_global_vertex_id(r_local));
//                Lr.vertices.push_back(G.get_global_vertex_id(r_local) - roots_start);
            }
        } else {
            for (VertexID r_local : roots_master_local) {
                IndexType &Lr = L[r_local];
                Lr.batches.emplace_back(
//                        b_id, // Batch ID
                        Lr.distances.size(), // start_index
                        1); // size
                Lr.distances.emplace_back(
                        Lr.vertices.size(), // start_index
                        1, // size
                        0); // dist
                Lr.vertices.push_back(G.get_global_vertex_id(r_local));
//                Lr.vertices.push_back(G.get_global_vertex_id(r_local) - roots_start);
            }
        }
    }
    catch (const std::bad_alloc &) {
        double memtotal = 0;
        double memfree = 0;
        PADO::Utils::system_memory(memtotal, memfree);
        printf("initialization_real_index: bad_alloc "
               "host_id: %d "
               "L.size(): %.2fGB "
               "memtotal: %.2fGB "
               "memfree: %.2fGB\n",
               host_id,
               get_index_size() * 1.0 / (1 << 30),
               memtotal / 1024,
               memfree / 1024);
        exit(1);
    }

    // Dist Table
    try
    {
//        struct LabelTableUnit {
//            VertexID root_id;
//            VertexID label_global_id;
//            UnweightedDist dist;
//
//            LabelTableUnit() = default;
//
//            LabelTableUnit(VertexID r, VertexID l, UnweightedDist d) :
//                    root_id(r), label_global_id(l), dist(d) {}
//        };
        std::vector<LabelTableUnit> buffer_send; // buffer for sending
        // Dist_matrix
        {
            // Deprecated Old method: unpack the IndexType structure before sending.
            // Okay, it's back.
            if (size_roots_master_local >= THRESHOLD_PARALLEL) {
                // Offsets for adding labels to buffer_send in parallel
                std::vector<VertexID> offsets_beffer_send(size_roots_master_local);
#pragma omp parallel for
                for (VertexID i_r = 0; i_r < size_roots_master_local; ++i_r) {
                    VertexID r_local = roots_master_local[i_r];
                    offsets_beffer_send[i_r] = L[r_local].vertices.size();
                }
                EdgeID size_labels = PADO::prefix_sum_for_offsets(offsets_beffer_send);
                buffer_send.resize(size_labels);
#pragma omp parallel for
                for (VertexID i_r = 0; i_r < size_roots_master_local; ++i_r) {
                    VertexID r_local = roots_master_local[i_r];
                    VertexID top_location = 0;
                    IndexType &Lr = L[r_local];
                    VertexID r_root_id = G.get_global_vertex_id(r_local) - roots_start;
                    VertexID b_i_bound = Lr.batches.size();
                    _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
                    _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
                    _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
                    // Traverse batches array
                    for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
//                        VertexID id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
                        VertexID dist_start_index = Lr.batches[b_i].start_index;
                        VertexID dist_bound_index = dist_start_index + Lr.batches[b_i].size;
                        // Traverse distances array
                        for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
//                        VertexID dist_bound_index = Lr.distances.size();
//                        for (VertexID dist_i = 0; dist_i < dist_bound_index; ++dist_i) {
                            VertexID v_start_index = Lr.distances[dist_i].start_index;
                            VertexID v_bound_index = v_start_index + Lr.distances[dist_i].size;
                            UnweightedDist dist = Lr.distances[dist_i].dist;
                            // Traverse vertices array
                            for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                                // Write into the dist_table
//                                buffer_send[offsets_beffer_send[i_r] + top_location++] =
//                                        LabelTableUnit(r_root_id, Lr.vertices[v_i] + id_offset, dist);
                                buffer_send[offsets_beffer_send[i_r] + top_location++] =
                                        LabelTableUnit(r_root_id, Lr.vertices[v_i], dist);
                            }
                        }
                    }
                }
            } else {
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
//                        VertexID id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
                        VertexID dist_start_index = Lr.batches[b_i].start_index;
                        VertexID dist_bound_index = dist_start_index + Lr.batches[b_i].size;
                    // Traverse distances array
                        for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
//                    VertexID dist_bound_index = Lr.distances.size();
//                    for (VertexID dist_i = 0; dist_i < dist_bound_index; ++dist_i) {
                            VertexID v_start_index = Lr.distances[dist_i].start_index;
                            VertexID v_bound_index = v_start_index + Lr.distances[dist_i].size;
                            UnweightedDist dist = Lr.distances[dist_i].dist;
                            // Traverse vertices array
                            for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                                // Write into the dist_table
                                buffer_send.emplace_back(r_root_id, Lr.vertices[v_i],
                                                         dist); // buffer for sending
//                            buffer_send.emplace_back(r_root_id, Lr.vertices[v_i] + id_offset,
//                                                     dist); // buffer for sending
                            }
                        }
                    }
                }
            }
        }
        // Broadcast local roots labels
        for (int root = 0; root < num_hosts; ++root) {
            std::vector<LabelTableUnit> buffer_recv;
            one_host_bcasts_buffer_to_buffer(root,
                                             buffer_send,
                                             buffer_recv);
            if (buffer_recv.empty()) {
                continue;
            }
            EdgeID size_buffer_recv = buffer_recv.size();
            if (size_buffer_recv >= THRESHOLD_PARALLEL) {
                std::vector<VertexID> sizes_recved_root_labels(roots_size, 0);
#pragma omp parallel for
                for (EdgeID i_l = 0; i_l < size_buffer_recv; ++i_l) {
                    const LabelTableUnit &l = buffer_recv[i_l];
                    VertexID root_id = l.root_id;
                    VertexID label_global_id = l.label_global_id;
                    UnweightedDist dist = l.dist;
                    dist_table[root_id][label_global_id] = dist;
                    // Record root_id's number of its received label, for later adding to recved_dist_table
                    __atomic_add_fetch(sizes_recved_root_labels.data() + root_id, 1, __ATOMIC_SEQ_CST);
//                    recved_dist_table[root_id].push_back(label_global_id);
                }
                // Record the received label in recved_dist_table, for later reset
#pragma omp parallel for
                for (VertexID root_id = 0; root_id < roots_size; ++root_id) {
                    VertexID &size = sizes_recved_root_labels[root_id];
                    if (size) {
                        recved_dist_table[root_id].resize(size);
                        size = 0;
                    }
                }
#pragma omp parallel for
                for (EdgeID i_l = 0; i_l < size_buffer_recv; ++i_l) {
                    const LabelTableUnit &l = buffer_recv[i_l];
                    VertexID root_id = l.root_id;
                    VertexID label_global_id = l.label_global_id;
                    PADO::TS_enqueue(recved_dist_table[root_id], sizes_recved_root_labels[root_id], label_global_id);
                }
            } else {
                for (const LabelTableUnit &l : buffer_recv) {
                    VertexID root_id = l.root_id;
                    VertexID label_global_id = l.label_global_id;
                    UnweightedDist dist = l.dist;
                    dist_table[root_id][label_global_id] = dist;
                    // Record the received label in recved_dist_table, for later reset
                    recved_dist_table[root_id].push_back(label_global_id);
                }
            }
        }
    }
    catch (const std::bad_alloc &) {
        double memtotal = 0;
        double memfree = 0;
        PADO::Utils::system_memory(memtotal, memfree);
        printf("initialization_dist_table: bad_alloc "
               "host_id: %d "
               "L.size(): %.2fGB "
               "memtotal: %.2fGB "
               "memfree: %.2fGB\n",
               host_id,
               get_index_size() * 1.0 / (1 << 30),
               memtotal / 1024,
               memfree / 1024);
        exit(1);
    }

	// Build the Bit-Parallel Labels Table
	try
    {
//        struct MsgBPLabel {
//            VertexID r_root_id;
//            UnweightedDist bp_dist[BITPARALLEL_SIZE];
//            uint64_t bp_sets[BITPARALLEL_SIZE][2];
//
//            MsgBPLabel() = default;
//            MsgBPLabel(VertexID r, const UnweightedDist dist[], const uint64_t sets[][2])
//                    : r_root_id(r)
//            {
//                memcpy(bp_dist, dist, sizeof(bp_dist));
//                memcpy(bp_sets, sets, sizeof(bp_sets));
//            }
//        };
//        std::vector<MPI_Request> requests_send(num_hosts - 1);
        std::vector<MsgBPLabel> buffer_send;

        std::vector<VertexID> roots_queue;
        for (VertexID r_global = roots_start; r_global < roots_bound; ++r_global) {
            if (G.get_master_host_id(r_global) != host_id) {
                continue;
            }
            roots_queue.push_back(r_global);
        }
        VertexID  size_roots_queue = roots_queue.size();
        if (size_roots_queue >= THRESHOLD_PARALLEL) {
            buffer_send.resize(size_roots_queue);
#pragma omp parallel for
            for (VertexID i_r = 0; i_r < size_roots_queue; ++i_r) {
                VertexID r_global = roots_queue[i_r];
                VertexID r_local = G.get_local_vertex_id(r_global);
                VertexID r_root = r_global - roots_start;
                // Prepare for sending
//                buffer_send.emplace_back(r_root, L[r_local].bp_dist, L[r_local].bp_sets);
                buffer_send[i_r] = MsgBPLabel(r_root, L[r_local].bp_dist, L[r_local].bp_sets);
            }
        } else {
//            for (VertexID r_global = roots_start; r_global < roots_bound; ++r_global) {
//                if (G.get_master_host_id(r_global) != host_id) {
//                    continue;
//                }
            for (VertexID r_global : roots_queue) {
                VertexID r_local = G.get_local_vertex_id(r_global);
                VertexID r_root = r_global - roots_start;
                // Local roots
//            memcpy(bp_labels_table[r_root].bp_dist, L[r_local].bp_dist, sizeof(bp_labels_table[r_root].bp_dist));
//            memcpy(bp_labels_table[r_root].bp_sets, L[r_local].bp_sets, sizeof(bp_labels_table[r_root].bp_sets));
                // Prepare for sending
                buffer_send.emplace_back(r_root, L[r_local].bp_dist, L[r_local].bp_sets);
            }
        }

        for (int root = 0; root < num_hosts; ++root) {
            std::vector<MsgBPLabel> buffer_recv;
            one_host_bcasts_buffer_to_buffer(root,
                                             buffer_send,
                                             buffer_recv);
            if (buffer_recv.empty()) {
                continue;
            }
            VertexID size_buffer_recv = buffer_recv.size();
            if (size_buffer_recv >= THRESHOLD_PARALLEL) {
#pragma omp parallel for
                for (VertexID i_m = 0; i_m < size_buffer_recv; ++i_m) {
                    const MsgBPLabel &m = buffer_recv[i_m];
                    VertexID r_root = m.r_root_id;
                    memcpy(bp_labels_table[r_root].bp_dist, m.bp_dist, sizeof(bp_labels_table[r_root].bp_dist));
                    memcpy(bp_labels_table[r_root].bp_sets, m.bp_sets, sizeof(bp_labels_table[r_root].bp_sets));
                }
            } else {
                for (const MsgBPLabel &m : buffer_recv) {
                    VertexID r_root = m.r_root_id;
                    memcpy(bp_labels_table[r_root].bp_dist, m.bp_dist, sizeof(bp_labels_table[r_root].bp_dist));
                    memcpy(bp_labels_table[r_root].bp_sets, m.bp_sets, sizeof(bp_labels_table[r_root].bp_sets));
                }
            }
        }
    }
    catch (const std::bad_alloc &) {
        double memtotal = 0;
        double memfree = 0;
        PADO::Utils::system_memory(memtotal, memfree);
        printf("initialization_bp_labels_table: bad_alloc "
               "host_id: %d "
               "L.size(): %.2fGB "
               "memtotal: %.2fGB "
               "memfree: %.2fGB\n",
               host_id,
               get_index_size() * 1.0 / (1 << 30),
               memtotal / 1024,
               memfree / 1024);
        exit(1);
    }

    // Active_queue
    VertexID global_num_actives = 0; // global number of active vertices.
    {
        if (size_roots_master_local >= THRESHOLD_PARALLEL) {
#pragma omp parallel for
            for (VertexID i_r = 0; i_r < size_roots_master_local; ++i_r) {
                VertexID r_local = roots_master_local[i_r];
                active_queue[i_r] = r_local;
            }
            end_active_queue = size_roots_master_local;
        } else {
            for (VertexID r_local : roots_master_local) {
                active_queue[end_active_queue++] = r_local;
            }
            {

            }
        }
        // Get the global number of active vertices;
//        message_time -= WallTimer::get_time_mark();
        MPI_Allreduce(&end_active_queue,
                      &global_num_actives,
                      1,
                      V_ID_Type,
//                      MPI_SUM,
                      MPI_MAX,
                      MPI_COMM_WORLD);
//        message_time += WallTimer::get_time_mark();
    }

    return global_num_actives;
}

// Sequential Version
//// Function for initializing at the begin of a batch
//// For a batch, initialize the temporary labels and real labels of roots;
//// traverse roots' labels to initialize distance buffer;
//// unset flag arrays is_active and got_labels
//template <VertexID BATCH_SIZE>
//inline VertexID DistBVCPLL<BATCH_SIZE>::
//initialization(
//        const DistGraph &G,
//        std::vector<ShortIndex> &short_index,
//        std::vector< std::vector<UnweightedDist> > &dist_table,
//        std::vector< std::vector<VertexID> > &recved_dist_table,
//        std::vector<BPLabelType> &bp_labels_table,
//        std::vector<VertexID> &active_queue,
//        VertexID &end_active_queue,
//        std::vector<VertexID> &once_candidated_queue,
//        VertexID &end_once_candidated_queue,
//        std::vector<uint8_t> &once_candidated,
//        VertexID b_id,
//        VertexID roots_start,
//        VertexID roots_size,
////        std::vector<VertexID> &roots_master_local,
//        const std::vector<uint8_t> &used_bp_roots)
//{
//    // Get the roots_master_local, containing all local roots.
//    std::vector<VertexID> roots_master_local;
//    VertexID roots_bound = roots_start + roots_size;
//    for (VertexID r_global = roots_start; r_global < roots_bound; ++r_global) {
//        if (G.get_master_host_id(r_global) == host_id && !used_bp_roots[r_global]) {
//            roots_master_local.push_back(G.get_local_vertex_id(r_global));
//        }
//    }
//    // Short_index
//    {
//        for (VertexID v_i = 0; v_i < end_once_candidated_queue; ++v_i) {
//            VertexID v_local = once_candidated_queue[v_i];
//            short_index[v_local].indicator_reset();
//            once_candidated[v_local] = 0;
//        }
//        end_once_candidated_queue = 0;
//        for (VertexID r_local : roots_master_local) {
//            short_index[r_local].indicator[G.get_global_vertex_id(r_local) - roots_start] = 1; // v itself
//            short_index[r_local].indicator[BATCH_SIZE] = 1; // v got labels
////            short_index[r_local].indicator.set(G.get_global_vertex_id(r_local) - roots_start); // v itself
////            short_index[r_local].indicator.set(BATCH_SIZE); // v got labels
//        }
//    }
////
//    // Real Index
//    {
//        for (VertexID r_local : roots_master_local) {
//            IndexType &Lr = L[r_local];
//            Lr.batches.emplace_back(
//                    b_id, // Batch ID
//                    Lr.distances.size(), // start_index
//                    1); // size
//            Lr.distances.emplace_back(
//                    Lr.vertices.size(), // start_index
//                    1, // size
//                    0); // dist
//            Lr.vertices.push_back(G.get_global_vertex_id(r_local) - roots_start);
//        }
//    }
//
//    // Dist Table
//    {
////        struct LabelTableUnit {
////            VertexID root_id;
////            VertexID label_global_id;
////            UnweightedDist dist;
////
////            LabelTableUnit() = default;
////
////            LabelTableUnit(VertexID r, VertexID l, UnweightedDist d) :
////                    root_id(r), label_global_id(l), dist(d) {}
////        };
//        std::vector<LabelTableUnit> buffer_send; // buffer for sending
//        // Dist_matrix
//        {
//            // Deprecated Old method: unpack the IndexType structure before sending.
//            for (VertexID r_local : roots_master_local) {
//                // The distance table.
//                IndexType &Lr = L[r_local];
//                VertexID r_root_id = G.get_global_vertex_id(r_local) - roots_start;
//                VertexID b_i_bound = Lr.batches.size();
//                _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
//                _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
//                _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
//                // Traverse batches array
//                for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
//                    VertexID id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
//                    VertexID dist_start_index = Lr.batches[b_i].start_index;
//                    VertexID dist_bound_index = dist_start_index + Lr.batches[b_i].size;
//                    // Traverse distances array
//                    for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
//                        VertexID v_start_index = Lr.distances[dist_i].start_index;
//                        VertexID v_bound_index = v_start_index + Lr.distances[dist_i].size;
//                        UnweightedDist dist = Lr.distances[dist_i].dist;
//                        // Traverse vertices array
//                        for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
//                            // Write into the dist_table
////                            dist_table[r_root_id][Lr.vertices[v_i] + id_offset] = dist; // distance table
//                            buffer_send.emplace_back(r_root_id, Lr.vertices[v_i] + id_offset,
//                                                     dist); // buffer for sending
//                        }
//                    }
//                }
//            }
//        }
//        // Broadcast local roots labels
//        for (int root = 0; root < num_hosts; ++root) {
//            std::vector<LabelTableUnit> buffer_recv;
//            one_host_bcasts_buffer_to_buffer(root,
//                                             buffer_send,
//                                             buffer_recv);
//            if (buffer_recv.empty()) {
//                continue;
//            }
//            for (const LabelTableUnit &l : buffer_recv) {
//                VertexID root_id = l.root_id;
//                VertexID label_global_id = l.label_global_id;
//                UnweightedDist dist = l.dist;
//                dist_table[root_id][label_global_id] = dist;
//                // Record the received label in recved_dist_table, for later reset
//                recved_dist_table[root_id].push_back(label_global_id);
//            }
//        }
//    }
//
//	// Build the Bit-Parallel Labels Table
//    {
////        struct MsgBPLabel {
////            VertexID r_root_id;
////            UnweightedDist bp_dist[BITPARALLEL_SIZE];
////            uint64_t bp_sets[BITPARALLEL_SIZE][2];
////
////            MsgBPLabel() = default;
////            MsgBPLabel(VertexID r, const UnweightedDist dist[], const uint64_t sets[][2])
////                    : r_root_id(r)
////            {
////                memcpy(bp_dist, dist, sizeof(bp_dist));
////                memcpy(bp_sets, sets, sizeof(bp_sets));
////            }
////        };
////        std::vector<MPI_Request> requests_send(num_hosts - 1);
//        std::vector<MsgBPLabel> buffer_send;
//        for (VertexID r_global = roots_start; r_global < roots_bound; ++r_global) {
//            if (G.get_master_host_id(r_global) != host_id) {
//                continue;
//            }
//            VertexID r_local = G.get_local_vertex_id(r_global);
//            VertexID r_root = r_global - roots_start;
//            // Local roots
////            memcpy(bp_labels_table[r_root].bp_dist, L[r_local].bp_dist, sizeof(bp_labels_table[r_root].bp_dist));
////            memcpy(bp_labels_table[r_root].bp_sets, L[r_local].bp_sets, sizeof(bp_labels_table[r_root].bp_sets));
//            // Prepare for sending
//            buffer_send.emplace_back(r_root, L[r_local].bp_dist, L[r_local].bp_sets);
//        }
//
//        for (int root = 0; root < num_hosts; ++root) {
//            std::vector<MsgBPLabel> buffer_recv;
//            one_host_bcasts_buffer_to_buffer(root,
//                                             buffer_send,
//                                             buffer_recv);
//            if (buffer_recv.empty()) {
//                continue;
//            }
//            for (const MsgBPLabel &m : buffer_recv) {
//                VertexID r_root = m.r_root_id;
//                memcpy(bp_labels_table[r_root].bp_dist, m.bp_dist, sizeof(bp_labels_table[r_root].bp_dist));
//                memcpy(bp_labels_table[r_root].bp_sets, m.bp_sets, sizeof(bp_labels_table[r_root].bp_sets));
//            }
//        }
//    }
//
//    // TODO: parallel enqueue
//    // Active_queue
//    VertexID global_num_actives = 0; // global number of active vertices.
//    {
//        for (VertexID r_local : roots_master_local) {
//            active_queue[end_active_queue++] = r_local;
//        }
//        // Get the global number of active vertices;
//        message_time -= WallTimer::get_time_mark();
//        MPI_Allreduce(&end_active_queue,
//                      &global_num_actives,
//                      1,
//                      V_ID_Type,
//                      MPI_SUM,
//                      MPI_COMM_WORLD);
//        message_time += WallTimer::get_time_mark();
//    }
//
//    return global_num_actives;
//}

//// Function: push v_head_global's newly added labels to its all neighbors.
//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::
//push_single_label(
//        VertexID v_head_global,
//        VertexID label_root_id,
//        VertexID roots_start,
//        const DistGraph &G,
//        std::vector<ShortIndex> &short_index,
//        std::vector<VertexID> &got_candidates_queue,
//        VertexID &end_got_candidates_queue,
//        std::vector<bool> &got_candidates,
//        std::vector<VertexID> &once_candidated_queue,
//        VertexID &end_once_candidated_queue,
//        std::vector<bool> &once_candidated,
//        const std::vector<BPLabelType> &bp_labels_table,
//        const std::vector<uint8_t> &used_bp_roots,
//        UnweightedDist iter)
//{
//    const BPLabelType &L_label = bp_labels_table[label_root_id];
//    VertexID label_global_id = label_root_id + roots_start;
//    EdgeID e_i_start = G.vertices_idx[v_head_global];
//    EdgeID e_i_bound = e_i_start + G.local_out_degrees[v_head_global];
//    for (EdgeID e_i = e_i_start; e_i < e_i_bound; ++e_i) {
//        VertexID v_tail_global = G.out_edges[e_i];
//        if (used_bp_roots[v_tail_global]) {
//            continue;
//        }
//        if (v_tail_global < roots_start) { // all remaining v_tail_global has higher rank than any roots, then no roots can push new labels to it.
//            return;
//        }
//
//        VertexID v_tail_local = G.get_local_vertex_id(v_tail_global);
//        const IndexType &L_tail = L[v_tail_local];
//        if (v_tail_global <= label_global_id) {
//            // remaining v_tail_global has higher rank than the label
//            return;
//        }
//        ShortIndex &SI_v_tail = short_index[v_tail_local];
//        if (SI_v_tail.indicator[label_root_id]) {
//            // The label is already selected before
//            continue;
//        }
//        // Record label_root_id as once selected by v_tail_global
//        SI_v_tail.indicator.set(label_root_id);
//        // Add into once_candidated_queue
//
//        if (!once_candidated[v_tail_local]) {
//            // If v_tail_global is not in the once_candidated_queue yet, add it in
//            once_candidated[v_tail_local] = true;
//            once_candidated_queue[end_once_candidated_queue++] = v_tail_local;
//        }
//        // Bit Parallel Checking: if label_global_id to v_tail_global has shorter distance already
//        //			++total_check_count;
////        const IndexType &L_label = L[label_global_id];
////        _mm_prefetch(&L_label.bp_dist[0], _MM_HINT_T0);
////        _mm_prefetch(&L_label.bp_sets[0][0], _MM_HINT_T0);
////			bp_checking_ins_count.measure_start();
//        bool no_need_add = false;
//        for (VertexID i = 0; i < BITPARALLEL_SIZE; ++i) {
//            VertexID td = L_label.bp_dist[i] + L_tail.bp_dist[i];
//            if (td - 2 <= iter) {
//                td +=
//                        (L_label.bp_sets[i][0] & L_tail.bp_sets[i][0]) ? -2 :
//                        ((L_label.bp_sets[i][0] & L_tail.bp_sets[i][1]) |
//                         (L_label.bp_sets[i][1] & L_tail.bp_sets[i][0]))
//                        ? -1 : 0;
//                if (td <= iter) {
//                    no_need_add = true;
////						++bp_hit_count;
//                    break;
//                }
//            }
//        }
//        if (no_need_add) {
////				bp_checking_ins_count.measure_stop();
//            continue;
//        }
////			bp_checking_ins_count.measure_stop();
//        if (SI_v_tail.is_candidate[label_root_id]) {
//            continue;
//        }
//        SI_v_tail.is_candidate[label_root_id] = true;
//        SI_v_tail.candidates_que[SI_v_tail.end_candidates_que++] = label_root_id;
//
//        if (!got_candidates[v_tail_local]) {
//            // If v_tail_global is not in got_candidates_queue, add it in (prevent duplicate)
//            got_candidates[v_tail_local] = true;
//            got_candidates_queue[end_got_candidates_queue++] = v_tail_local;
//        }
//    }
////    {// Just for the complain from the compiler
////        assert(iter >= iter);
////    }
//}

template<VertexID BATCH_SIZE>
inline void DistBVCPLL<BATCH_SIZE>::
schedule_label_pushing_para(
        const DistGraph &G,
        const VertexID roots_start,
        const std::vector<uint8_t> &used_bp_roots,
        const std::vector<VertexID> &active_queue,
        const VertexID global_start,
        const VertexID global_size,
        const VertexID local_size,
//        const VertexID start_active_queue,
//        const VertexID size_active_queue,
        std::vector<VertexID> &got_candidates_queue,
        VertexID &end_got_candidates_queue,
        std::vector<ShortIndex> &short_index,
        const std::vector<BPLabelType> &bp_labels_table,
        std::vector<uint8_t> &got_candidates,
        std::vector<uint8_t> &is_active,
        std::vector<VertexID> &once_candidated_queue,
        VertexID &end_once_candidated_queue,
        std::vector<uint8_t> &once_candidated,
        const UnweightedDist iter)
{
    std::vector<std::pair<VertexID, VertexID> > buffer_send_indices;
                                                //.first: Vertex ID
                                                //.second: size of labels
    std::vector<VertexID> buffer_send_labels;
    if (local_size) {
        const VertexID start_active_queue = global_start;
        const VertexID size_active_queue = global_size <= local_size ?
                                            global_size :
                                            local_size;
        const VertexID bound_active_queue = start_active_queue + size_active_queue;

        buffer_send_indices.resize(size_active_queue);
        // Prepare offset for inserting
        std::vector<VertexID> offsets_buffer_locs(size_active_queue);
#pragma omp parallel for
        for (VertexID i_q = start_active_queue; i_q < bound_active_queue; ++i_q) {
            VertexID v_head_local = active_queue[i_q];
            is_active[v_head_local] = 0; // reset is_active
            const IndexType &Lv = L[v_head_local];
            offsets_buffer_locs[i_q - start_active_queue] = Lv.distances.rbegin()->size;
        }
        EdgeID size_buffer_send_labels = PADO::prefix_sum_for_offsets(offsets_buffer_locs);
        try {
            buffer_send_labels.resize(size_buffer_send_labels);
        }
        catch (const std::bad_alloc &) {
            double memtotal = 0;
            double memfree = 0;
            PADO::Utils::system_memory(memtotal, memfree);
            printf("schedule_label_pushing_para.buffer_send_labels: bad_alloc "
                   "host_id: %d "
                   "size_buffer_send_labels: %lu "
                   "L.size(): %.2fGB "
                   "memtotal: %.2fGB "
                   "memfree: %.2fGB\n",
                   host_id,
                   size_buffer_send_labels,
                   get_index_size() * 1.0 / (1 << 30),
                   memtotal / 1024,
                   memfree / 1024);
            exit(1);
        }


        // Build buffer_send_labels by parallel inserting
#pragma omp parallel for
        for (VertexID i_q = start_active_queue; i_q < bound_active_queue; ++i_q) {
            VertexID v_head_local = active_queue[i_q];
            is_active[v_head_local] = 0; // reset is_active
            VertexID v_head_global = G.get_global_vertex_id(v_head_local);
            const IndexType &Lv = L[v_head_local];
            // Prepare the buffer_send_indices
            VertexID tmp_i_q = i_q - start_active_queue;
            buffer_send_indices[tmp_i_q] = std::make_pair(v_head_global, Lv.distances.rbegin()->size);
            // These 2 index are used for traversing v_head's last inserted labels
            VertexID l_i_start = Lv.distances.rbegin()->start_index;
            VertexID l_i_bound = l_i_start + Lv.distances.rbegin()->size;
            VertexID top_labels = offsets_buffer_locs[tmp_i_q];
            for (VertexID l_i = l_i_start; l_i < l_i_bound; ++l_i) {
                VertexID label_root_id = Lv.vertices[l_i] - roots_start;
                buffer_send_labels[top_labels++] = label_root_id;
//                        buffer_send_labels.push_back(label_root_id);
            }
        }
    }


////////////////////////////////////////////////
////
//    const VertexID bound_active_queue = start_active_queue + size_active_queue;
//    std::vector<std::pair<VertexID, VertexID> > buffer_send_indices(size_active_queue);
//        //.first: Vertex ID
//        //.second: size of labels
//    std::vector<VertexID> buffer_send_labels;
//    // Prepare masters' newly added labels for sending
//    // Parallel Version
//    // Prepare offset for inserting
//    std::vector<VertexID> offsets_buffer_locs(size_active_queue);
//#pragma omp parallel for
//    for (VertexID i_q = start_active_queue; i_q < bound_active_queue; ++i_q) {
//        VertexID v_head_local = active_queue[i_q];
//        is_active[v_head_local] = 0; // reset is_active
//        const IndexType &Lv = L[v_head_local];
//        offsets_buffer_locs[i_q - start_active_queue] = Lv.distances.rbegin()->size;
//    }
//    EdgeID size_buffer_send_labels = PADO::prefix_sum_for_offsets(offsets_buffer_locs);
////    {// test
////        if (0 == host_id) {
////            double memtotal = 0;
////            double memfree = 0;
////            double bytes_buffer_send_labels = size_buffer_send_labels * sizeof(VertexID);
////            PADO::Utils::system_memory(memtotal, memfree);
////            printf("bytes_buffer_send_labels: %fGB memtotal: %fGB memfree: %fGB\n",
////                   bytes_buffer_send_labels / (1 << 30), memtotal / 1024, memfree / 1024);
////        }
////    }
//    buffer_send_labels.resize(size_buffer_send_labels);
////    {// test
////        if (0 == host_id) {
////            printf("buffer_send_labels created.\n");
////        }
////    }
//
//    // Build buffer_send_labels by parallel inserting
//#pragma omp parallel for
//    for (VertexID i_q = start_active_queue; i_q < bound_active_queue; ++i_q) {
//        VertexID tmp_i_q = i_q - start_active_queue;
//        VertexID v_head_local = active_queue[i_q];
//        is_active[v_head_local] = 0; // reset is_active
//        VertexID v_head_global = G.get_global_vertex_id(v_head_local);
//        const IndexType &Lv = L[v_head_local];
//        // Prepare the buffer_send_indices
//        buffer_send_indices[tmp_i_q] = std::make_pair(v_head_global, Lv.distances.rbegin()->size);
//        // These 2 index are used for traversing v_head's last inserted labels
//        VertexID l_i_start = Lv.distances.rbegin()->start_index;
//        VertexID l_i_bound = l_i_start + Lv.distances.rbegin()->size;
//        VertexID top_labels = offsets_buffer_locs[tmp_i_q];
//        for (VertexID l_i = l_i_start; l_i < l_i_bound; ++l_i) {
//            VertexID label_root_id = Lv.vertices[l_i];
//            buffer_send_labels[top_labels++] = label_root_id;
////                        buffer_send_labels.push_back(label_root_id);
//        }
//    }
////    end_active_queue = 0;
////
////////////////////////////////////////////////

    for (int root = 0; root < num_hosts; ++root) {
        // Get the indices
        std::vector<std::pair<VertexID, VertexID> > indices_buffer;

        one_host_bcasts_buffer_to_buffer(root,
                                         buffer_send_indices,
                                         indices_buffer);
        if (indices_buffer.empty()) {
            continue;
        }
        // Get the labels
        std::vector<VertexID> labels_buffer;
        one_host_bcasts_buffer_to_buffer(root,
                                         buffer_send_labels,
                                         labels_buffer);
        VertexID size_indices_buffer = indices_buffer.size();
        // Prepare the offsets for reading indices_buffer
        std::vector<EdgeID> starts_locs_index(size_indices_buffer);
#pragma omp parallel for
        for (VertexID i_i = 0; i_i < size_indices_buffer; ++i_i) {
            const std::pair<VertexID, VertexID> &e = indices_buffer[i_i];
            starts_locs_index[i_i] = e.second;
        }
        EdgeID total_recved_labels = PADO::prefix_sum_for_offsets(starts_locs_index);

        // Prepare the offsets for inserting v_tails into queue
        std::vector<VertexID> offsets_tmp_queue(size_indices_buffer);
#pragma omp parallel for
        for (VertexID i_i = 0; i_i < size_indices_buffer; ++i_i) {
            const std::pair<VertexID, VertexID> &e = indices_buffer[i_i];
            offsets_tmp_queue[i_i] = G.local_out_degrees[e.first];
        }
        EdgeID num_ngbrs = PADO::prefix_sum_for_offsets(offsets_tmp_queue);

        std::vector<VertexID> tmp_got_candidates_queue;
        std::vector<VertexID> sizes_tmp_got_candidates_queue;
        std::vector<VertexID> tmp_once_candidated_queue;
        std::vector<VertexID> sizes_tmp_once_candidated_queue;
        try {
            tmp_got_candidates_queue.resize(num_ngbrs);
            sizes_tmp_got_candidates_queue.resize(size_indices_buffer, 0);
            tmp_once_candidated_queue.resize(num_ngbrs);
            sizes_tmp_once_candidated_queue.resize(size_indices_buffer, 0);
        }
        catch (const std::bad_alloc &) {
            double memtotal = 0;
            double memfree = 0;
            PADO::Utils::system_memory(memtotal, memfree);
            printf("schedule_label_pushing_para.tmp_queues: bad_alloc "
                   "host_id: %d "
                   "num_ngbrs: %lu "
                   "size_indices_buffer: %u "
                   "L.size(): %.2fGB "
                   "memtotal: %.2fGB "
                   "memfree: %.2fGB\n",
                   host_id,
                   num_ngbrs,
                   size_indices_buffer,
                   get_index_size() * 1.0 / (1 << 30),
                   memtotal / 1024,
                   memfree / 1024);
            exit(1);
        }

#pragma omp parallel for
        for (VertexID i_i = 0; i_i < size_indices_buffer; ++i_i) {
            VertexID v_head_global = indices_buffer[i_i].first;
            EdgeID start_index = starts_locs_index[i_i];
            EdgeID bound_index = i_i != size_indices_buffer - 1 ?
                                 starts_locs_index[i_i + 1] : total_recved_labels;
            if (G.local_out_degrees[v_head_global]) {
                local_push_labels_para(
                        v_head_global,
                        start_index,
                        bound_index,
                        roots_start,
                        labels_buffer,
                        G,
                        short_index,
                        //        std::vector<VertexID> &got_candidates_queue,
                        //        VertexID &end_got_candidates_queue,
                        tmp_got_candidates_queue,
                        sizes_tmp_got_candidates_queue[i_i],
                        offsets_tmp_queue[i_i],
                        got_candidates,
                        //        std::vector<VertexID> &once_candidated_queue,
                        //        VertexID &end_once_candidated_queue,
                        tmp_once_candidated_queue,
                        sizes_tmp_once_candidated_queue[i_i],
                        once_candidated,
                        bp_labels_table,
                        used_bp_roots,
                        iter);
            }
        }

        {// Collect elements from tmp_got_candidates_queue to got_candidates_queue
            VertexID total_new = PADO::prefix_sum_for_offsets(sizes_tmp_got_candidates_queue);
            PADO::collect_into_queue(
                    tmp_got_candidates_queue,
                    offsets_tmp_queue, // the locations for reading tmp_got_candidate_queue
                    sizes_tmp_got_candidates_queue, // the locations for writing got_candidate_queue
                    total_new,
                    got_candidates_queue,
                    end_got_candidates_queue);
        }
        {// Collect elements from tmp_once_candidated_queue to once_candidated_queue
            VertexID total_new = PADO::prefix_sum_for_offsets(sizes_tmp_once_candidated_queue);
            PADO::collect_into_queue(
                    tmp_once_candidated_queue,
                    offsets_tmp_queue, // the locations for reading tmp_once_candidats_queue
                    sizes_tmp_once_candidated_queue, // the locations for writing once_candidated_queue
                    total_new,
                    once_candidated_queue,
                    end_once_candidated_queue);
        }
    }
}

// Function: pushes v_head's labels to v_head's every (master) neighbor
template <VertexID BATCH_SIZE>
inline void DistBVCPLL<BATCH_SIZE>::
local_push_labels_para(
        const VertexID v_head_global,
        const EdgeID start_index,
        const EdgeID bound_index,
        const VertexID roots_start,
        const std::vector<VertexID> &labels_buffer,
        const DistGraph &G,
        std::vector<ShortIndex> &short_index,
//        std::vector<VertexID> &got_candidates_queue,
//        VertexID &end_got_candidates_queue,
        std::vector<VertexID> &tmp_got_candidates_queue,
        VertexID &size_tmp_got_candidates_queue,
        const VertexID offset_tmp_queue,
        std::vector<uint8_t> &got_candidates,
//        std::vector<VertexID> &once_candidated_queue,
//        VertexID &end_once_candidated_queue,
        std::vector<VertexID> &tmp_once_candidated_queue,
        VertexID &size_tmp_once_candidated_queue,
        std::vector<uint8_t> &once_candidated,
        const std::vector<BPLabelType> &bp_labels_table,
        const std::vector<uint8_t> &used_bp_roots,
        const UnweightedDist iter)
{
    // Traverse v_head's every neighbor v_tail
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
        VertexID v_tail_local = G.get_local_vertex_id(v_tail_global);
        const IndexType &L_tail = L[v_tail_local];
        ShortIndex &SI_v_tail = short_index[v_tail_local];
        // Traverse v_head's last inserted labels
        for (VertexID l_i = start_index; l_i < bound_index; ++l_i) {
            VertexID label_root_id = labels_buffer[l_i];
            VertexID label_global_id = label_root_id + roots_start;
            if (v_tail_global <= label_global_id) {
                // v_tail_global has higher rank than the label
                continue;
            }
//            if (SI_v_tail.indicator[label_root_id]) {
//                // The label is already selected before
//                continue;
//            }
//            // Record label_root_id as once selected by v_tail_global
//            SI_v_tail.indicator[label_root_id] = 1;
            {// Deal with race condition
                if (!PADO::CAS(SI_v_tail.indicator.data() + label_root_id, static_cast<uint8_t>(0),
                        static_cast<uint8_t>(1))) {
                    // The label is already selected before
                    continue;
                }
            }
            // Add into once_candidated_queue
            if (!once_candidated[v_tail_local]) {
                // If v_tail_global is not in the once_candidated_queue yet, add it in
                if (PADO::CAS(once_candidated.data() + v_tail_local, static_cast<uint8_t>(0), static_cast<uint8_t>(1))) {
                    tmp_once_candidated_queue[offset_tmp_queue + size_tmp_once_candidated_queue++] = v_tail_local;
                }
//                once_candidated[v_tail_local] = 1;
//                once_candidated_queue[end_once_candidated_queue++] = v_tail_local;
            }

            // Bit Parallel Checking: if label_global_id to v_tail_global has shorter distance already
//            const IndexType &L_label = L[label_global_id];
//            _mm_prefetch(&L_label.bp_dist[0], _MM_HINT_T0);
//            _mm_prefetch(&L_label.bp_sets[0][0], _MM_HINT_T0);
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
                        break;
                    }
                }
            }
            if (no_need_add) {
                continue;
            }
//            if (SI_v_tail.is_candidate[label_root_id]) {
//                continue;
//            }
//            SI_v_tail.is_candidate[label_root_id] = 1;
//            SI_v_tail.candidates_que[SI_v_tail.end_candidates_que++] = label_root_id;
            if (!SI_v_tail.is_candidate[label_root_id]) {
                if (CAS(SI_v_tail.is_candidate.data() + label_root_id, static_cast<uint8_t>(0), static_cast<uint8_t>(1))) {
                    PADO::TS_enqueue(SI_v_tail.candidates_que, SI_v_tail.end_candidates_que, label_root_id);
                }
            }

            // Add into got_candidates queue
//            if (!got_candidates[v_tail_local]) {
//                // If v_tail_global is not in got_candidates_queue, add it in (prevent duplicate)
//                got_candidates[v_tail_local] = 1;
//                got_candidates_queue[end_got_candidates_queue++] = v_tail_local;
//            }
            if (!got_candidates[v_tail_local]) {
                if (CAS(got_candidates.data() + v_tail_local, static_cast<uint8_t>(0), static_cast<uint8_t>(1))) {
                    tmp_got_candidates_queue[offset_tmp_queue + size_tmp_got_candidates_queue++] = v_tail_local;
                }
            }
        }
    }

//    {
//        assert(iter >= iter);
//    }
}
// Function: pushes v_head's labels to v_head's every (master) neighbor
template <VertexID BATCH_SIZE>
inline void DistBVCPLL<BATCH_SIZE>::
local_push_labels_seq(
        VertexID v_head_global,
        EdgeID start_index,
        EdgeID bound_index,
        VertexID roots_start,
        const std::vector<VertexID> &labels_buffer,
        const DistGraph &G,
        std::vector<ShortIndex> &short_index,
        std::vector<VertexID> &got_candidates_queue,
        VertexID &end_got_candidates_queue,
        std::vector<uint8_t> &got_candidates,
        std::vector<VertexID> &once_candidated_queue,
        VertexID &end_once_candidated_queue,
        std::vector<uint8_t> &once_candidated,
        const std::vector<BPLabelType> &bp_labels_table,
        const std::vector<uint8_t> &used_bp_roots,
        const UnweightedDist iter)
{
    // Traverse v_head's every neighbor v_tail
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
        for (VertexID l_i = start_index; l_i < bound_index; ++l_i) {
            VertexID label_root_id = labels_buffer[l_i];
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
            SI_v_tail.indicator[label_root_id] = 1;
//            SI_v_tail.indicator.set(label_root_id);
            // Add into once_candidated_queue

            if (!once_candidated[v_tail_local]) {
                // If v_tail_global is not in the once_candidated_queue yet, add it in
                once_candidated[v_tail_local] = 1;
                once_candidated_queue[end_once_candidated_queue++] = v_tail_local;
            }

            // Bit Parallel Checking: if label_global_id to v_tail_global has shorter distance already
//            const IndexType &L_label = L[label_global_id];
//            _mm_prefetch(&L_label.bp_dist[0], _MM_HINT_T0);
//            _mm_prefetch(&L_label.bp_sets[0][0], _MM_HINT_T0);
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
                        break;
                    }
                }
            }
            if (no_need_add) {
                continue;
            }
            if (SI_v_tail.is_candidate[label_root_id]) {
                continue;
            }
            SI_v_tail.is_candidate[label_root_id] = 1;
            SI_v_tail.candidates_que[SI_v_tail.end_candidates_que++] = label_root_id;

            if (!got_candidates[v_tail_local]) {
                // If v_tail_global is not in got_candidates_queue, add it in (prevent duplicate)
                got_candidates[v_tail_local] = 1;
                got_candidates_queue[end_got_candidates_queue++] = v_tail_local;
            }
        }
    }

//    {
//        assert(iter >= iter);
//    }
}


//// Function: pushes v_head's labels to v_head's every (master) neighbor
//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::
//local_push_labels(
//        VertexID v_head_local,
//        VertexID roots_start,
//        const DistGraph &G,
//        std::vector<ShortIndex> &short_index,
//        std::vector<VertexID> &got_candidates_queue,
//        VertexID &end_got_candidates_queue,
//        std::vector<bool> &got_candidates,
//        std::vector<VertexID> &once_candidated_queue,
//        VertexID &end_once_candidated_queue,
//        std::vector<bool> &once_candidated,
//        const std::vector<BPLabelType> &bp_labels_table,
//        const std::vector<uint8_t> &used_bp_roots,
//        UnweightedDist iter)
//{
//    // The data structure of a message
////    std::vector< LabelUnitType > buffer_recv;
//    const IndexType &Lv = L[v_head_local];
//    // These 2 index are used for traversing v_head's last inserted labels
//    VertexID l_i_start = Lv.distances.rbegin() -> start_index;
//    VertexID l_i_bound = l_i_start + Lv.distances.rbegin() -> size;
//    // Traverse v_head's every neighbor v_tail
//    VertexID v_head_global = G.get_global_vertex_id(v_head_local);
//    EdgeID e_i_start = G.vertices_idx[v_head_global];
//    EdgeID e_i_bound = e_i_start + G.local_out_degrees[v_head_global];
//    for (EdgeID e_i = e_i_start; e_i < e_i_bound; ++e_i) {
//        VertexID v_tail_global = G.out_edges[e_i];
//        if (used_bp_roots[v_tail_global]) {
//            continue;
//        }
//        if (v_tail_global < roots_start) { // v_tail_global has higher rank than any roots, then no roots can push new labels to it.
//            return;
//        }
//
//        // Traverse v_head's last inserted labels
//        for (VertexID l_i = l_i_start; l_i < l_i_bound; ++l_i) {
//            VertexID label_root_id = Lv.vertices[l_i];
//            VertexID label_global_id = label_root_id + roots_start;
//            if (v_tail_global <= label_global_id) {
//                // v_tail_global has higher rank than the label
//                continue;
//            }
//            VertexID v_tail_local = G.get_local_vertex_id(v_tail_global);
//            const IndexType &L_tail = L[v_tail_local];
//            ShortIndex &SI_v_tail = short_index[v_tail_local];
//            if (SI_v_tail.indicator[label_root_id]) {
//                // The label is already selected before
//                continue;
//            }
//            // Record label_root_id as once selected by v_tail_global
//            SI_v_tail.indicator.set(label_root_id);
//            // Add into once_candidated_queue
//
//            if (!once_candidated[v_tail_local]) {
//                // If v_tail_global is not in the once_candidated_queue yet, add it in
//                once_candidated[v_tail_local] = true;
//                once_candidated_queue[end_once_candidated_queue++] = v_tail_local;
//            }
//
//            // Bit Parallel Checking: if label_global_id to v_tail_global has shorter distance already
//            //			++total_check_count;
////            const IndexType &L_label = L[label_global_id];
////            _mm_prefetch(&L_label.bp_dist[0], _MM_HINT_T0);
////            _mm_prefetch(&L_label.bp_sets[0][0], _MM_HINT_T0);
////			bp_checking_ins_count.measure_start();
//            const BPLabelType &L_label = bp_labels_table[label_root_id];
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
//            if (SI_v_tail.is_candidate[label_root_id]) {
//                continue;
//            }
//            SI_v_tail.is_candidate[label_root_id] = true;
//            SI_v_tail.candidates_que[SI_v_tail.end_candidates_que++] = label_root_id;
//
//            if (!got_candidates[v_tail_local]) {
//                // If v_tail_global is not in got_candidates_queue, add it in (prevent duplicate)
//                got_candidates[v_tail_local] = true;
//                got_candidates_queue[end_got_candidates_queue++] = v_tail_local;
//            }
//        }
//    }
//
//    {
//        assert(iter >= iter);
//    }
//}


//// DEPRECATED Function: in the scatter phase, synchronize local masters to mirrors on other hosts
//// Has some mysterious problem: when I call this function, some hosts will receive wrong messages; when I copy all
//// code of this function into the caller, all messages become right.
//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::
//sync_masters_2_mirrors(
//        const DistGraph &G,
//        const std::vector<VertexID> &active_queue,
//        VertexID end_active_queue,
//		std::vector< std::pair<VertexID, VertexID> > &buffer_send,
//        std::vector<MPI_Request> &requests_send
//)
//{
////    std::vector< std::pair<VertexID, VertexID> > buffer_send;
//        // pair.first: Owener vertex ID of the label
//        // pair.first: label vertex ID of the label
//    // Prepare masters' newly added labels for sending
//    for (VertexID i_q = 0; i_q < end_active_queue; ++i_q) {
//        VertexID v_head_local = active_queue[i_q];
//        VertexID v_head_global = G.get_global_vertex_id(v_head_local);
//        const IndexType &Lv = L[v_head_local];
//        // These 2 index are used for traversing v_head's last inserted labels
//        VertexID l_i_start = Lv.distances.rbegin()->start_index;
//        VertexID l_i_bound = l_i_start + Lv.distances.rbegin()->size;
//        for (VertexID l_i = l_i_start; l_i < l_i_bound; ++l_i) {
//            VertexID label_root_id = Lv.vertices[l_i];
//            buffer_send.emplace_back(v_head_global, label_root_id);
////			{//test
////				if (1 == host_id) {
////					printf("@%u host_id: %u v_head_global: %u\n", __LINE__, host_id, v_head_global);//
////				}
////			}
//        }
//    }
//	{
//		if (!buffer_send.empty()) {
//			printf("@%u host_id: %u sync_masters_2_mirrors: buffer_send.size: %lu buffer_send[0]:(%u %u)\n", __LINE__, host_id, buffer_send.size(), buffer_send[0].first, buffer_send[0].second);
//		}
//		assert(!requests_send.empty());
//	}
//
//    // Send messages
//    for (int loc = 0; loc < num_hosts - 1; ++loc) {
//        int dest_host_id = G.buffer_send_list_loc_2_master_host_id(loc);
//        MPI_Isend(buffer_send.data(),
//                MPI_Instance::get_sending_size(buffer_send),
//                MPI_CHAR,
//                dest_host_id,
//                SENDING_MASTERS_TO_MIRRORS,
//                MPI_COMM_WORLD,
//                &requests_send[loc]);
//		{
//			if (!buffer_send.empty()) {
//				printf("@%u host_id: %u dest_host_id: %u buffer_send.size: %lu buffer_send[0]:(%u %u)\n", __LINE__, host_id, dest_host_id, buffer_send.size(), buffer_send[0].first, buffer_send[0].second);
//			}
//		}
//    }
//}

template <VertexID BATCH_SIZE>
inline void DistBVCPLL<BATCH_SIZE>::
schedule_label_inserting_para(
        const DistGraph &G,
        const VertexID roots_start,
        const VertexID roots_size,
        std::vector<ShortIndex> &short_index,
        const std::vector< std::vector<UnweightedDist> > &dist_table,
        const std::vector<VertexID> &got_candidates_queue,
        const VertexID start_got_candidates_queue,
        const VertexID size_got_candidates_queue,
        std::vector<uint8_t> &got_candidates,
        std::vector<VertexID> &active_queue,
        VertexID &end_active_queue,
        std::vector<uint8_t> &is_active,
        std::vector< std::pair<VertexID, VertexID> > &buffer_send,
        const VertexID iter)
{
    const VertexID bound_got_candidates_queue = start_got_candidates_queue + size_got_candidates_queue;
    std::vector<VertexID> offsets_tmp_active_queue;
    std::vector<VertexID> tmp_active_queue;
    std::vector<VertexID> sizes_tmp_active_queue;
    std::vector<EdgeID> offsets_tmp_buffer_send;
    std::vector< std::pair<VertexID, VertexID> > tmp_buffer_send;
    std::vector<EdgeID> sizes_tmp_buffer_send;
    EdgeID total_send_labels;

    try {
        offsets_tmp_active_queue.resize(size_got_candidates_queue);
#pragma omp parallel for
        for (VertexID i_q = 0; i_q < size_got_candidates_queue; ++i_q) {
            offsets_tmp_active_queue[i_q] = i_q;
        }
        tmp_active_queue.resize(size_got_candidates_queue);
        sizes_tmp_active_queue.resize(size_got_candidates_queue,
                                      0); // Size will only be 0 or 1, but it will become offsets eventually.

        // Prepare for parallel buffer_send
//                std::vector<EdgeID> offsets_tmp_buffer_send(size_got_candidates_queue);
        offsets_tmp_buffer_send.resize(size_got_candidates_queue);
#pragma omp parallel for
        for (VertexID i_q = start_got_candidates_queue; i_q < bound_got_candidates_queue; ++i_q) {
            VertexID v_id_local = got_candidates_queue[i_q];
            VertexID v_global_id = G.get_global_vertex_id(v_id_local);
            VertexID tmp_i_q = i_q - start_got_candidates_queue;
            if (v_global_id >= roots_start && v_global_id < roots_start + roots_size) {
                // If v_global_id is root, its new labels should be put into buffer_send
                offsets_tmp_buffer_send[tmp_i_q] = short_index[v_id_local].end_candidates_que;
            } else {
                offsets_tmp_buffer_send[tmp_i_q] = 0;
            }
        }
        total_send_labels = PADO::prefix_sum_for_offsets(offsets_tmp_buffer_send);
        tmp_buffer_send.resize(total_send_labels);
        sizes_tmp_buffer_send.resize(size_got_candidates_queue, 0);
    }
    catch (const std::bad_alloc &) {
        double memtotal = 0;
        double memfree = 0;
        PADO::Utils::system_memory(memtotal, memfree);
        printf("L%u_tmp_buffer_send: bad_alloc "
               "host_id: %d "
               "iter: %u "
               "size_got_candidates_queue: %u "
               "total_send_labels: %u "
               "L.size(): %.2fGB "
               "memtotal: %.2fGB "
               "memfree: %.2fGB\n",
               __LINE__,
               host_id,
               iter,
               size_got_candidates_queue,
               total_send_labels,
               get_index_size() * 1.0 / (1 << 30),
               memtotal / 1024,
               memfree / 1024);
        exit(1);
    }

#pragma omp parallel for
    for (VertexID i_queue = start_got_candidates_queue; i_queue < bound_got_candidates_queue; ++i_queue) {
        VertexID v_id_local = got_candidates_queue[i_queue];
        VertexID inserted_count = 0; //recording number of v_id's truly inserted candidates
        got_candidates[v_id_local] = 0; // reset got_candidates
        // Traverse v_id's all candidates
        VertexID tmp_i_queue = i_queue - start_got_candidates_queue;
        VertexID bound_cand_i = short_index[v_id_local].end_candidates_que;
        for (VertexID cand_i = 0; cand_i < bound_cand_i; ++cand_i) {
            VertexID cand_root_id = short_index[v_id_local].candidates_que[cand_i];
            short_index[v_id_local].is_candidate[cand_root_id] = 0;
            // Only insert cand_root_id into v_id's label if its distance to v_id is shorter than existing distance
            if (distance_query(
                    cand_root_id,
                    v_id_local,
                    roots_start,
                    //                        L,
                    dist_table,
                    iter)) {
                if (!is_active[v_id_local]) {
                    is_active[v_id_local] = 1;
//                                active_queue[end_active_queue++] = v_id_local;
                    tmp_active_queue[tmp_i_queue + sizes_tmp_active_queue[tmp_i_queue]++] = v_id_local;
                }
                ++inserted_count;
                // The candidate cand_root_id needs to be added into v_id's label
                insert_label_only_para(
                        cand_root_id,
                        v_id_local,
                        roots_start,
                        roots_size,
                        G,
                        tmp_buffer_send,
                        sizes_tmp_buffer_send[tmp_i_queue],
                        offsets_tmp_buffer_send[tmp_i_queue]);
//                                    buffer_send);
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
//                                b_id,
                    iter);
        }
    }

    {// Collect elements from tmp_active_queue to active_queue
        VertexID total_new = PADO::prefix_sum_for_offsets(sizes_tmp_active_queue);
        PADO::collect_into_queue(
                tmp_active_queue,
                offsets_tmp_active_queue,
                sizes_tmp_active_queue,
                total_new,
                active_queue,
                end_active_queue);
    }
    {// Collect elements from tmp_buffer_send to buffer_send
        EdgeID old_size_buffer_send = buffer_send.size();
        EdgeID total_new = PADO::prefix_sum_for_offsets(sizes_tmp_buffer_send);
        try {
            buffer_send.resize(total_new + old_size_buffer_send);
        }
        catch (const std::bad_alloc &) {
            double memtotal = 0;
            double memfree = 0;
            PADO::Utils::system_memory(memtotal, memfree);
            printf("L%u_buffer_send: bad_alloc "
                   "iter: %u "
                   "host_id: %d "
                   "L.size(): %.2fGB "
                   "memtotal: %.2fGB "
                   "memfree: %.2fGB\n",
                   __LINE__,
                   iter,
                   host_id,
                   get_index_size() * 1.0 / (1 << 30),
                   memtotal / 1024,
                   memfree / 1024);
            exit(1);
        }
//        EdgeID zero_size = 0;
        PADO::collect_into_queue(
                tmp_buffer_send,
                offsets_tmp_buffer_send,
                sizes_tmp_buffer_send,
                total_new,
                buffer_send,
                old_size_buffer_send);
//                zero_size);
    }
}

// Function for distance query;
// traverse vertex v_id's labels;
// return false if shorter distance exists already, return true if the cand_root_id can be added into v_id's label.
template <VertexID BATCH_SIZE>
inline bool DistBVCPLL<BATCH_SIZE>::
distance_query(
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
//        VertexID id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
        VertexID dist_start_index = Lv.batches[b_i].start_index;
        VertexID dist_bound_index = dist_start_index + Lv.batches[b_i].size;
        // Traverse dist_table
        for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
//        VertexID dist_bound_index = Lv.distances.size();
//        for (VertexID dist_i = 0; dist_i < dist_bound_index; ++dist_i) {
            UnweightedDist dist = Lv.distances[dist_i].dist;
            // Cannot use this, because no batch_id any more, so distances are not all in order among batches.
            if (dist >= iter) { // In a batch, the labels' distances are increasingly ordered.
                // If the half path distance is already greater than their targeted distance, jump to next batch
                break;
            }
            VertexID v_start_index = Lv.distances[dist_i].start_index;
            VertexID v_bound_index = v_start_index + Lv.distances[dist_i].size;
//            _mm_prefetch(&dist_table[cand_root_id][0], _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(dist_table[cand_root_id].data()), _MM_HINT_T0);
            for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
//                VertexID v = Lv.vertices[v_i] + id_offset; // v is a label hub of v_id
                VertexID v = Lv.vertices[v_i]; // v is a label hub of v_id
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

//// Sequential version
// Function inserts candidate cand_root_id into vertex v_id's labels;
// update the distance buffer dist_table;
// but it only update the v_id's labels' vertices array;
template <VertexID BATCH_SIZE>
inline void DistBVCPLL<BATCH_SIZE>::
insert_label_only_seq(
        VertexID cand_root_id,
        VertexID v_id_local,
        VertexID roots_start,
        VertexID roots_size,
        const DistGraph &G,
//        std::vector< std::vector<UnweightedDist> > &dist_table,
        std::vector< std::pair<VertexID, VertexID> > &buffer_send)
//        UnweightedDist iter)
{
    try {
        VertexID cand_real_id = cand_root_id + roots_start;
        L[v_id_local].vertices.push_back(cand_real_id);
//    L[v_id_local].vertices.push_back(cand_root_id);
        // Update the distance buffer if v_id is a root
        VertexID v_id_global = G.get_global_vertex_id(v_id_local);
        VertexID v_root_id = v_id_global - roots_start;
        if (v_id_global >= roots_start && v_root_id < roots_size) {
//        VertexID cand_real_id = cand_root_id + roots_start;
//        dist_table[v_root_id][cand_real_id] = iter;
            // Put the update into the buffer_send for later sending
            buffer_send.emplace_back(v_root_id, cand_real_id);
        }
    }
    catch (const std::bad_alloc &) {
        double memtotal = 0;
        double memfree = 0;
        PADO::Utils::system_memory(memtotal, memfree);
        printf("insert_label_only_seq: bad_alloc "
               "host_id: %d "
               "L.size(): %.2fGB "
               "memtotal: %.2fGB "
               "memfree: %.2fGB\n",
               host_id,
               get_index_size() * 1.0 / (1 << 30),
               memtotal / 1024,
               memfree / 1024);
        exit(1);
    }
}

//// Parallel Version
// Function inserts candidate cand_root_id into vertex v_id's labels;
// update the distance buffer dist_table;
// but it only update the v_id's labels' vertices array;
template <VertexID BATCH_SIZE>
inline void DistBVCPLL<BATCH_SIZE>::
insert_label_only_para(
        VertexID cand_root_id,
        VertexID v_id_local,
        VertexID roots_start,
        VertexID roots_size,
        const DistGraph &G,
//        std::vector< std::pair<VertexID, VertexID> > &buffer_send)
        std::vector< std::pair<VertexID, VertexID> > &tmp_buffer_send,
        EdgeID &size_tmp_buffer_send,
        const EdgeID offset_tmp_buffer_send)
{
    try {
        VertexID cand_real_id = cand_root_id + roots_start;
        L[v_id_local].vertices.push_back(cand_real_id);
//    L[v_id_local].vertices.push_back(cand_root_id);
        // Update the distance buffer if v_id is a root
        VertexID v_id_global = G.get_global_vertex_id(v_id_local);
        VertexID v_root_id = v_id_global - roots_start;
        if (v_id_global >= roots_start && v_root_id < roots_size) {
//        VertexID cand_real_id = cand_root_id + roots_start;
            // Put the update into the buffer_send for later sending
            tmp_buffer_send[offset_tmp_buffer_send + size_tmp_buffer_send++] = std::make_pair(v_root_id, cand_real_id);
        }
    }
    catch (const std::bad_alloc &) {
        double memtotal = 0;
        double memfree = 0;
        PADO::Utils::system_memory(memtotal, memfree);
        printf("insert_label_only_para: bad_alloc "
               "host_id: %d "
               "L.size(): %.2fGB "
               "memtotal: %.2fGB "
               "memfree: %.2fGB\n",
               host_id,
               get_index_size() * 1.0 / (1 << 30),
               memtotal / 1024,
               memfree / 1024);
        exit(1);
    }
}

// Function updates those index arrays in v_id's label only if v_id has been inserted new labels
template <VertexID BATCH_SIZE>
inline void DistBVCPLL<BATCH_SIZE>::
update_label_indices(
        const VertexID v_id_local,
        const VertexID inserted_count,
//        std::vector<IndexType> &L,
        std::vector<ShortIndex> &short_index,
//        VertexID b_id,
        const UnweightedDist iter)
{
    try {
        IndexType &Lv = L[v_id_local];
    // indicator[BATCH_SIZE + 1] is true, means v got some labels already in this batch
    if (short_index[v_id_local].indicator[BATCH_SIZE]) {
        // Increase the batches' last element's size because a new distance element need to be added
        ++(Lv.batches.rbegin() -> size);
    } else {
        short_index[v_id_local].indicator[BATCH_SIZE] = 1;
//        short_index[v_id_local].indicator.set(BATCH_SIZE);
        // Insert a new Batch with batch_id, start_index, and size because a new distance element need to be added
        Lv.batches.emplace_back(
//                b_id, // batch id
                Lv.distances.size(), // start index
                1); // size
    }
        // Insert a new distance element with start_index, size, and dist
        Lv.distances.emplace_back(
                Lv.vertices.size() - inserted_count, // start index
                inserted_count, // size
                iter); // distance
    }
    catch (const std::bad_alloc &) {
        double memtotal = 0;
        double memfree = 0;
        PADO::Utils::system_memory(memtotal, memfree);
        printf("update_label_indices: bad_alloc "
               "host_id: %d "
               "L.size(): %.2fGB "
               "memtotal: %.2fGB "
               "memfree: %.2fGB\n",
               host_id,
               get_index_size() * 1.0 / (1 << 30),
               memtotal / 1024,
               memfree / 1024);
        exit(1);
    }
}

// Function to reset dist_table the distance buffer to INF
// Traverse every root's labels to reset its distance buffer elements to INF.
// In this way to reduce the cost of initialization of the next batch.
template <VertexID BATCH_SIZE>
inline void DistBVCPLL<BATCH_SIZE>::
reset_at_end(
        const DistGraph &G,
//        VertexID roots_start,
//        const std::vector<VertexID> &roots_master_local,
        std::vector< std::vector<UnweightedDist> > &dist_table,
        std::vector< std::vector<VertexID> > &recved_dist_table,
        std::vector<BPLabelType> &bp_labels_table,
        const std::vector<VertexID> &once_candidated_queue,
        const VertexID end_once_candidated_queue)
{
//    // Reset dist_table according to local masters' labels
//    for (VertexID r_local_id : roots_master_local) {
//        IndexType &Lr = L[r_local_id];
//        VertexID r_root_id = G.get_global_vertex_id(r_local_id) - roots_start;
//        VertexID b_i_bound = Lr.batches.size();
//        _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
//        _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
//        _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
//        for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
//            VertexID id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
//            VertexID dist_start_index = Lr.batches[b_i].start_index;
//            VertexID dist_bound_index = dist_start_index + Lr.batches[b_i].size;
//            // Traverse dist_table
//            for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
//                VertexID v_start_index = Lr.distances[dist_i].start_index;
//                VertexID v_bound_index = v_start_index + Lr.distances[dist_i].size;
//                for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
//                    dist_table[r_root_id][Lr.vertices[v_i] + id_offset] = MAX_UNWEIGHTED_DIST;
//                }
//            }
//        }
//    }
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

    // Remove labels of local minimum set
    for (VertexID v_i = 0; v_i < end_once_candidated_queue; ++v_i) {
        VertexID v_local_id = once_candidated_queue[v_i];
        if (!G.is_local_minimum[v_local_id]) {
            continue;
        }
        L[v_local_id].clean_all_indices();
    }
}

template <VertexID BATCH_SIZE>
inline void DistBVCPLL<BATCH_SIZE>::
batch_process(
        const DistGraph &G,
//        const VertexID b_id,
        const VertexID roots_start, // start id of roots
        const VertexID roots_size, // how many roots in the batch
        const std::vector<uint8_t> &used_bp_roots,
        std::vector<VertexID> &active_queue,
        VertexID &end_active_queue,
        std::vector<VertexID> &got_candidates_queue,
        VertexID &end_got_candidates_queue,
        std::vector<ShortIndex> &short_index,
        std::vector< std::vector<UnweightedDist> > &dist_table,
        std::vector< std::vector<VertexID> > &recved_dist_table,
        std::vector<BPLabelType> &bp_labels_table,
        std::vector<uint8_t> &got_candidates,
//        std::vector<bool> &got_candidates,
        std::vector<uint8_t> &is_active,
//        std::vector<bool> &is_active,
        std::vector<VertexID> &once_candidated_queue,
        VertexID &end_once_candidated_queue,
        std::vector<uint8_t> &once_candidated)
//        std::vector<bool> &once_candidated)
{
    // At the beginning of a batch, initialize the labels L and distance buffer dist_table;
//    initializing_time -= WallTimer::get_time_mark();
    // The Maximum of active vertices among hosts.
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
//                                    b_id,
                                    roots_start,
                                    roots_size,
//                                    roots_master_local,
                                    used_bp_roots);
//    initializing_time += WallTimer::get_time_mark();
    UnweightedDist iter = 0; // The iterator, also the distance for current iteration
//    {//test
//        if (0 == host_id) {
//            printf("host_id: %u initialization finished.\n", host_id);
//        }
//    }


    while (global_num_actives) {
        ++iter;
        {// Limit the distance
            if (iter >7 ) {
                end_active_queue = 0;
                break;
            }
        }
        //#ifdef DEBUG_MESSAGES_ON
//        {//test
////            if (0 == host_id) {
//                double memtotal = 0;
//                double memfree = 0;
//                PADO::Utils::system_memory(memtotal, memfree);
//                printf("iter: %u "
//                       "host_id: %d "
//                       "global_num_actives: %u "
//                       "L.size(): %.2fGB "
//                       "memtotal: %.2fGB "
//                       "memfree: %.2fGB\n",
//                       iter,
//                       host_id,
//                       global_num_actives,
//                       get_index_size() * 1.0 / (1 << 30),
//                       memtotal / 1024,
//                       memfree / 1024);
////            }
//        }
//#endif
        // Traverse active vertices to push their labels as candidates
		// Send masters' newly added labels to other hosts
		try
        {
//            scatter_time -= WallTimer::get_time_mark();
            // Divide the pushing into many-time runs, to reduce the peak memory footprint.
            const VertexID chunk_size = 1 << 14;
            VertexID remainder = global_num_actives % chunk_size;
            VertexID bound_global_i = global_num_actives - remainder;
//            VertexID remainder = end_active_queue % chunk_size;
//            VertexID bound_active_queue = end_active_queue - remainder;
            VertexID local_size;
            for (VertexID global_i = 0; global_i < bound_global_i; global_i += chunk_size) {
                if (global_i < end_active_queue) {
                    local_size = end_active_queue - global_i;
                } else {
                    local_size = 0;
                }
//                {//test
//                    if (1024 == roots_start && 7 == host_id) {
//                        printf("S0 host_id: %d global_i: %u bound_global_i: %u local_size: %u\n",
//                                host_id, global_i, bound_global_i, local_size);
//                    }
//                }
                schedule_label_pushing_para(
                        G,
                        roots_start,
                        used_bp_roots,
                        active_queue,
                        global_i,
                        chunk_size,
                        local_size,
                        got_candidates_queue,
                        end_got_candidates_queue,
                        short_index,
                        bp_labels_table,
                        got_candidates,
                        is_active,
                        once_candidated_queue,
                        end_once_candidated_queue,
                        once_candidated,
                        iter);
            }
            if (remainder) {
                if (bound_global_i < end_active_queue) {
                    local_size = end_active_queue - bound_global_i;
                } else {
                    local_size = 0;
                }
                schedule_label_pushing_para(
                        G,
                        roots_start,
                        used_bp_roots,
                        active_queue,
                        bound_global_i,
                        remainder,
                        local_size,
                        got_candidates_queue,
                        end_got_candidates_queue,
                        short_index,
                        bp_labels_table,
                        got_candidates,
                        is_active,
                        once_candidated_queue,
                        end_once_candidated_queue,
                        once_candidated,
                        iter);
            }
//
//                schedule_label_pushing_para(
//                        G,
//                        roots_start,
//                        used_bp_roots,
//                        active_queue,
//                        0,
//                        end_active_queue,
//                        got_candidates_queue,
//                        end_got_candidates_queue,
//                        short_index,
//                        bp_labels_table,
//                        got_candidates,
//                        is_active,
//                        once_candidated_queue,
//                        end_once_candidated_queue,
//                        once_candidated,
//                        iter);
            end_active_queue = 0;
//            scatter_time += WallTimer::get_time_mark();
        }
		catch (const std::bad_alloc &) {
            double memtotal = 0;
            double memfree = 0;
            PADO::Utils::system_memory(memtotal, memfree);
            printf("pushing: bad_alloc "
                   "iter: %u "
                   "host_id: %d "
                   "global_num_actives: %u "
                   "L.size(): %.2fGB "
                   "memtotal: %.2fGB "
                   "memfree: %.2fGB\n",
                   iter,
                   host_id,
                   global_num_actives,
                   get_index_size() * 1.0 / (1 << 30),
                   memtotal / 1024,
                   memfree / 1024);
            exit(1);
		}

//        {//test
//            if (0 == host_id) {
//                printf("host_id: %u pushing finished...\n", host_id);
//            }
//        }

        // Traverse vertices in the got_candidates_queue to insert labels
		{
//		    gather_time -= WallTimer::get_time_mark();
            std::vector< std::pair<VertexID, VertexID> > buffer_send; // For sync elements in the dist_table
                // pair.first: root id
                // pair.second: label (global) id of the root
            if (end_got_candidates_queue >= THRESHOLD_PARALLEL) {
                const VertexID chunk_size = 1 << 16;
                VertexID remainder = end_got_candidates_queue % chunk_size;
                VertexID bound_i_q = end_got_candidates_queue - remainder;
                for (VertexID i_q = 0; i_q < bound_i_q; i_q += chunk_size) {
                    schedule_label_inserting_para(
                            G,
                            roots_start,
                            roots_size,
                            short_index,
                            dist_table,
                            got_candidates_queue,
                            i_q,
                            chunk_size,
                            got_candidates,
                            active_queue,
                            end_active_queue,
                            is_active,
                            buffer_send,
                            iter);
                }
                if (remainder) {
                    schedule_label_inserting_para(
                            G,
                            roots_start,
                            roots_size,
                            short_index,
                            dist_table,
                            got_candidates_queue,
                            bound_i_q,
                            remainder,
                            got_candidates,
                            active_queue,
                            end_active_queue,
                            is_active,
                            buffer_send,
                            iter);
                }

////// Backup
//                // Prepare for parallel active_queue
//                // Don't need offsets_tmp_active_queue here, because the index i_queue is the offset already.
//                // Actually we still need offsets_tmp_active_queue, because collect_into_queue() needs it.
//                std::vector<VertexID> offsets_tmp_active_queue;
//                std::vector<VertexID> tmp_active_queue;
//                std::vector<VertexID> sizes_tmp_active_queue;
//                std::vector<EdgeID> offsets_tmp_buffer_send;
//                std::vector< std::pair<VertexID, VertexID> > tmp_buffer_send;
//                std::vector<EdgeID> sizes_tmp_buffer_send;
//                EdgeID total_send_labels;
//
//                try {
//                    offsets_tmp_active_queue.resize(end_got_candidates_queue);
//#pragma omp parallel for
//                    for (VertexID i_q = 0; i_q < end_got_candidates_queue; ++i_q) {
//                        offsets_tmp_active_queue[i_q] = i_q;
//                    }
//                    tmp_active_queue.resize(end_got_candidates_queue);
//                    sizes_tmp_active_queue.resize(end_got_candidates_queue,
//                                                  0); // Size will only be 0 or 1, but it will become offsets eventually.
//
//                    // Prepare for parallel buffer_send
////                std::vector<EdgeID> offsets_tmp_buffer_send(end_got_candidates_queue);
//                    offsets_tmp_buffer_send.resize(end_got_candidates_queue);
//#pragma omp parallel for
//                    for (VertexID i_q = 0; i_q < end_got_candidates_queue; ++i_q) {
//                        VertexID v_id_local = got_candidates_queue[i_q];
//                        VertexID v_global_id = G.get_global_vertex_id(v_id_local);
//                        if (v_global_id >= roots_start && v_global_id < roots_start + roots_size) {
//                            // If v_global_id is root, its new labels should be put into buffer_send
//                            offsets_tmp_buffer_send[i_q] = short_index[v_id_local].end_candidates_que;
//                        } else {
//                            offsets_tmp_buffer_send[i_q] = 0;
//                        }
//                    }
//                    total_send_labels = PADO::prefix_sum_for_offsets(offsets_tmp_buffer_send);
//                    tmp_buffer_send.resize(total_send_labels);
//                    sizes_tmp_buffer_send.resize(end_got_candidates_queue, 0);
//                }
//                catch (const std::bad_alloc &) {
//                    double memtotal = 0;
//                    double memfree = 0;
//                    PADO::Utils::system_memory(memtotal, memfree);
//                    printf("L%u_tmp_buffer_send: bad_alloc "
//                           "host_id: %d "
//                           "iter: %u "
//                           "end_got_candidates_queue: %u "
//                           "total_send_labels: %u "
//                           "L.size(): %.2fGB "
//                           "memtotal: %.2fGB "
//                           "memfree: %.2fGB\n",
//                           __LINE__,
//                           host_id,
//                           iter,
//                           end_got_candidates_queue,
//                           total_send_labels,
//                           get_index_size() * 1.0 / (1 << 30),
//                           memtotal / 1024,
//                           memfree / 1024);
//                    exit(1);
//                }
//
//#pragma omp parallel for
//                for (VertexID i_queue = 0; i_queue < end_got_candidates_queue; ++i_queue) {
//                    VertexID v_id_local = got_candidates_queue[i_queue];
//                    VertexID inserted_count = 0; //recording number of v_id's truly inserted candidates
//                    got_candidates[v_id_local] = 0; // reset got_candidates
//                    // Traverse v_id's all candidates
//                    VertexID bound_cand_i = short_index[v_id_local].end_candidates_que;
//                    for (VertexID cand_i = 0; cand_i < bound_cand_i; ++cand_i) {
//                        VertexID cand_root_id = short_index[v_id_local].candidates_que[cand_i];
//                        short_index[v_id_local].is_candidate[cand_root_id] = 0;
//                        // Only insert cand_root_id into v_id's label if its distance to v_id is shorter than existing distance
//                        if (distance_query(
//                                cand_root_id,
//                                v_id_local,
//                                roots_start,
//                                //                        L,
//                                dist_table,
//                                iter)) {
//                            if (!is_active[v_id_local]) {
//                                is_active[v_id_local] = 1;
////                                active_queue[end_active_queue++] = v_id_local;
//                                tmp_active_queue[i_queue + sizes_tmp_active_queue[i_queue]++] = v_id_local;
//                            }
//                            ++inserted_count;
//                            // The candidate cand_root_id needs to be added into v_id's label
//                            insert_label_only_para(
//                                    cand_root_id,
//                                    v_id_local,
//                                    roots_start,
//                                    roots_size,
//                                    G,
//                                    tmp_buffer_send,
//                                    sizes_tmp_buffer_send[i_queue],
//                                    offsets_tmp_buffer_send[i_queue]);
////                                    buffer_send);
//                        }
//                    }
//                    short_index[v_id_local].end_candidates_que = 0;
//                    if (0 != inserted_count) {
//                        // Update other arrays in L[v_id] if new labels were inserted in this iteration
//                        update_label_indices(
//                                v_id_local,
//                                inserted_count,
//                                //                        L,
////                                short_index,
////                                b_id,
//                                iter);
//                    }
//                }
//
//                {// Collect elements from tmp_active_queue to active_queue
//                    VertexID total_new = PADO::prefix_sum_for_offsets(sizes_tmp_active_queue);
//                    PADO::collect_into_queue(
//                            tmp_active_queue,
//                            offsets_tmp_active_queue,
//                            sizes_tmp_active_queue,
//                            total_new,
//                            active_queue,
//                            end_active_queue);
//                }
//                {// Collect elements from tmp_buffer_send to buffer_send
//                    EdgeID total_new = PADO::prefix_sum_for_offsets(sizes_tmp_buffer_send);
//                    try {
//                        buffer_send.resize(total_new);
//                    }
//                    catch (const std::bad_alloc &) {
//                        double memtotal = 0;
//                        double memfree = 0;
//                        PADO::Utils::system_memory(memtotal, memfree);
//                        printf("L%u_buffer_send: bad_alloc "
//                               "iter: %u "
//                               "host_id: %d "
//                               "L.size(): %.2fGB "
//                               "memtotal: %.2fGB "
//                               "memfree: %.2fGB\n",
//                               __LINE__,
//                               iter,
//                               host_id,
//                               get_index_size() * 1.0 / (1 << 30),
//                               memtotal / 1024,
//                               memfree / 1024);
//                        exit(1);
//                    }
//                    EdgeID zero_size = 0;
//                    PADO::collect_into_queue(
//                            tmp_buffer_send,
//                            offsets_tmp_buffer_send,
//                            sizes_tmp_buffer_send,
//                            total_new,
//                            buffer_send,
//                            zero_size);
//                }

            } else {
                for (VertexID i_queue = 0; i_queue < end_got_candidates_queue; ++i_queue) {
                    VertexID v_id_local = got_candidates_queue[i_queue];
                    VertexID inserted_count = 0; //recording number of v_id's truly inserted candidates
                    got_candidates[v_id_local] = 0; // reset got_candidates
                    // Traverse v_id's all candidates
                    VertexID bound_cand_i = short_index[v_id_local].end_candidates_que;
                    for (VertexID cand_i = 0; cand_i < bound_cand_i; ++cand_i) {
                        VertexID cand_root_id = short_index[v_id_local].candidates_que[cand_i];
                        short_index[v_id_local].is_candidate[cand_root_id] = 0;
                        // Only insert cand_root_id into v_id's label if its distance to v_id is shorter than existing distance
                        if (distance_query(
                                cand_root_id,
                                v_id_local,
                                roots_start,
                                //                        L,
                                dist_table,
                                iter)) {
                            if (!is_active[v_id_local]) {
                                is_active[v_id_local] = 1;
                                active_queue[end_active_queue++] = v_id_local;
                            }
                            ++inserted_count;
                            // The candidate cand_root_id needs to be added into v_id's label
                            insert_label_only_seq(
                                    cand_root_id,
                                    v_id_local,
                                    roots_start,
                                    roots_size,
                                    G,
//                                dist_table,
                                    buffer_send);
//                                iter);
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
//                                b_id,
                                iter);
                    }
                }
            }
//            {//test
//                printf("host_id: %u gather: buffer_send.size(); %lu bytes: %lu\n", host_id, buffer_send.size(), MPI_Instance::get_sending_size(buffer_send));
//            }
            end_got_candidates_queue = 0; // Set the got_candidates_queue empty
            // Sync the dist_table
            for (int root = 0; root < num_hosts; ++root) {
                std::vector<std::pair<VertexID, VertexID>> buffer_recv;
                one_host_bcasts_buffer_to_buffer(root,
                                                 buffer_send,
                                                 buffer_recv);

                if (buffer_recv.empty()) {
                    continue;
                }

                EdgeID size_buffer_recv = buffer_recv.size();

                try {
                    if (size_buffer_recv >= THRESHOLD_PARALLEL) {
                        // Get label number for every root
                        std::vector<VertexID> sizes_recved_root_labels(roots_size, 0);
#pragma omp parallel for
                        for (EdgeID i_l = 0; i_l < size_buffer_recv; ++i_l) {
                            const std::pair<VertexID, VertexID> &e = buffer_recv[i_l];
                            VertexID root_id = e.first;
                            __atomic_add_fetch(sizes_recved_root_labels.data() + root_id, 1, __ATOMIC_SEQ_CST);
                        }
                        // Resize the recved_dist_table for every root
#pragma omp parallel for
                        for (VertexID root_id = 0; root_id < roots_size; ++root_id) {
                            VertexID old_size = recved_dist_table[root_id].size();
                            VertexID tmp_size = sizes_recved_root_labels[root_id];
                            if (tmp_size) {
                                recved_dist_table[root_id].resize(old_size + tmp_size);
                                sizes_recved_root_labels[root_id] = old_size; // sizes_recved_root_labels now records old_size
                            }
                            // If tmp_size ==  0, root_id has no received labels.
//                        sizes_recved_root_labels[root_id] = old_size; // sizes_recved_root_labels now records old_size
                        }
                        // Recorde received labels in recved_dist_table
#pragma omp parallel for
                        for (EdgeID i_l = 0; i_l < size_buffer_recv; ++i_l) {
                            const std::pair<VertexID, VertexID> &e = buffer_recv[i_l];
                            VertexID root_id = e.first;
                            VertexID cand_real_id = e.second;
                            dist_table[root_id][cand_real_id] = iter;
                            PADO::TS_enqueue(recved_dist_table[root_id], sizes_recved_root_labels[root_id],
                                             cand_real_id);
                        }
                    } else {
                        for (const std::pair<VertexID, VertexID> &e : buffer_recv) {
                            VertexID root_id = e.first;
                            VertexID cand_real_id = e.second;
                            dist_table[root_id][cand_real_id] = iter;
                            // Record the received element, for future reset
                            recved_dist_table[root_id].push_back(cand_real_id);
                        }
                    }
                }
                catch (const std::bad_alloc &) {
                    double memtotal = 0;
                    double memfree = 0;
                    PADO::Utils::system_memory(memtotal, memfree);
                    printf("recved_dist_table: bad_alloc "
                           "host_id: %d "
                           "iter: %u "
                           "L.size(): %.2fGB "
                           "memtotal: %.2fGB "
                           "memfree: %.2fGB\n",
                           host_id,
                           iter,
                           get_index_size() * 1.0 / (1 << 30),
                           memtotal / 1024,
                           memfree / 1024);
                    exit(1);
                }
            }

            // Sync the global_num_actives
            MPI_Allreduce(&end_active_queue,
                    &global_num_actives,
                    1,
                    V_ID_Type,
                    MPI_MAX,
//                    MPI_SUM,
                    MPI_COMM_WORLD);
//            gather_time += WallTimer::get_time_mark();
		}
//        {//test
//            if (0 == host_id) {
//                printf("iter: %u inserting labels finished.\n", iter);
//            }
//        }
    }

    // Reset the dist_table
//    clearup_time -= WallTimer::get_time_mark();
    reset_at_end(
            G,
//            roots_start,
//            roots_master_local,
            dist_table,
            recved_dist_table,
            bp_labels_table,
            once_candidated_queue,
            end_once_candidated_queue);
//    clearup_time += WallTimer::get_time_mark();
//    {//test
//        if (0 == host_id) {
//            printf("host_id: %u resetting finished.\n", host_id);
//        }
//    }
}

//// Sequential Version
//template <VertexID BATCH_SIZE>
//inline void DistBVCPLL<BATCH_SIZE>::
//batch_process(
//        const DistGraph &G,
//        VertexID b_id,
//        VertexID roots_start, // start id of roots
//        VertexID roots_size, // how many roots in the batch
//        const std::vector<uint8_t> &used_bp_roots,
//        std::vector<VertexID> &active_queue,
//        VertexID &end_active_queue,
//        std::vector<VertexID> &got_candidates_queue,
//        VertexID &end_got_candidates_queue,
//        std::vector<ShortIndex> &short_index,
//        std::vector< std::vector<UnweightedDist> > &dist_table,
//        std::vector< std::vector<VertexID> > &recved_dist_table,
//        std::vector<BPLabelType> &bp_labels_table,
//        std::vector<uint8_t> &got_candidates,
////        std::vector<bool> &got_candidates,
//        std::vector<uint8_t> &is_active,
////        std::vector<bool> &is_active,
//        std::vector<VertexID> &once_candidated_queue,
//        VertexID &end_once_candidated_queue,
//        std::vector<uint8_t> &once_candidated)
////        std::vector<bool> &once_candidated)
//{
//    // At the beginning of a batch, initialize the labels L and distance buffer dist_table;
//    initializing_time -= WallTimer::get_time_mark();
//    VertexID global_num_actives = initialization(G,
//                                    short_index,
//                                    dist_table,
//                                    recved_dist_table,
//                                    bp_labels_table,
//                                    active_queue,
//                                    end_active_queue,
//                                    once_candidated_queue,
//                                    end_once_candidated_queue,
//                                    once_candidated,
//                                    b_id,
//                                    roots_start,
//                                    roots_size,
////                                    roots_master_local,
//                                    used_bp_roots);
//    initializing_time += WallTimer::get_time_mark();
//    UnweightedDist iter = 0; // The iterator, also the distance for current iteration
////    {//test
////        printf("host_id: %u initialization finished.\n", host_id);
////    }
//
//
//    while (global_num_actives) {
////#ifdef DEBUG_MESSAGES_ON
////        {//
////           if (0 == host_id) {
////               printf("iter: %u global_num_actives: %u\n", iter, global_num_actives);
////           }
////        }
////#endif
//        ++iter;
//        // Traverse active vertices to push their labels as candidates
//		// Send masters' newly added labels to other hosts
//        {
//            scatter_time -= WallTimer::get_time_mark();
//            std::vector<std::pair<VertexID, VertexID> > buffer_send_indices(end_active_queue);
//                //.first: Vertex ID
//                //.second: size of labels
//            std::vector<VertexID> buffer_send_labels;
//            // Prepare masters' newly added labels for sending
//            for (VertexID i_q = 0; i_q < end_active_queue; ++i_q) {
//                VertexID v_head_local = active_queue[i_q];
//                is_active[v_head_local] = 0; // reset is_active
//                VertexID v_head_global = G.get_global_vertex_id(v_head_local);
//                const IndexType &Lv = L[v_head_local];
//                // Prepare the buffer_send_indices
//                buffer_send_indices[i_q] = std::make_pair(v_head_global, Lv.distances.rbegin()->size);
//                // These 2 index are used for traversing v_head's last inserted labels
//                VertexID l_i_start = Lv.distances.rbegin()->start_index;
//                VertexID l_i_bound = l_i_start + Lv.distances.rbegin()->size;
//                for (VertexID l_i = l_i_start; l_i < l_i_bound; ++l_i) {
//                    VertexID label_root_id = Lv.vertices[l_i];
//                    buffer_send_labels.push_back(label_root_id);
//                }
//            }
//            end_active_queue = 0;
//
//            for (int root = 0; root < num_hosts; ++root) {
//                // Get the indices
//                std::vector< std::pair<VertexID, VertexID> > indices_buffer;
//                one_host_bcasts_buffer_to_buffer(root,
//                                                 buffer_send_indices,
//                                                 indices_buffer);
//                if (indices_buffer.empty()) {
//                    continue;
//                }
//                // Get the labels
//                std::vector<VertexID> labels_buffer;
//                one_host_bcasts_buffer_to_buffer(root,
//                                                 buffer_send_labels,
//                                                 labels_buffer);
//                // Push those labels
//                EdgeID start_index = 0;
//                for (const std::pair<VertexID, VertexID> e : indices_buffer) {
//                    VertexID v_head_global = e.first;
//                    EdgeID bound_index = start_index + e.second;
//                    if (G.local_out_degrees[v_head_global]) {
//                        local_push_labels(
//                                v_head_global,
//                                start_index,
//                                bound_index,
//                                roots_start,
//                                labels_buffer,
//                                G,
//                                short_index,
//                                got_candidates_queue,
//                                end_got_candidates_queue,
//                                got_candidates,
//                                once_candidated_queue,
//                                end_once_candidated_queue,
//                                once_candidated,
//                                bp_labels_table,
//                                used_bp_roots,
//                                iter);
//                    }
//                    start_index = bound_index;
//                }
//            }
//            scatter_time += WallTimer::get_time_mark();
//        }
//
//        // Traverse vertices in the got_candidates_queue to insert labels
//		{
//		    gather_time -= WallTimer::get_time_mark();
//            std::vector< std::pair<VertexID, VertexID> > buffer_send; // For sync elements in the dist_table
//                // pair.first: root id
//                // pair.second: label (global) id of the root
//            for (VertexID i_queue = 0; i_queue < end_got_candidates_queue; ++i_queue) {
//                VertexID v_id_local = got_candidates_queue[i_queue];
//                VertexID inserted_count = 0; //recording number of v_id's truly inserted candidates
//                got_candidates[v_id_local] = 0; // reset got_candidates
//                // Traverse v_id's all candidates
//                VertexID bound_cand_i = short_index[v_id_local].end_candidates_que;
//                for (VertexID cand_i = 0; cand_i < bound_cand_i; ++cand_i) {
//                    VertexID cand_root_id = short_index[v_id_local].candidates_que[cand_i];
//                    short_index[v_id_local].is_candidate[cand_root_id] = 0;
//                    // Only insert cand_root_id into v_id's label if its distance to v_id is shorter than existing distance
//                    if ( distance_query(
//                            cand_root_id,
//                            v_id_local,
//                            roots_start,
//    //                        L,
//                            dist_table,
//                            iter) ) {
//                        if (!is_active[v_id_local]) {
//                            is_active[v_id_local] = 1;
//                            active_queue[end_active_queue++] = v_id_local;
//                        }
//                        ++inserted_count;
//                        // The candidate cand_root_id needs to be added into v_id's label
//                        insert_label_only(
//                                cand_root_id,
//                                v_id_local,
//                                roots_start,
//                                roots_size,
//                                G,
////                                dist_table,
//                                buffer_send);
////                                iter);
//                    }
//                }
//                short_index[v_id_local].end_candidates_que = 0;
//                if (0 != inserted_count) {
//                    // Update other arrays in L[v_id] if new labels were inserted in this iteration
//                    update_label_indices(
//                            v_id_local,
//                            inserted_count,
//    //                        L,
//                            short_index,
//                            b_id,
//                            iter);
//                }
//            }
////            {//test
////                printf("host_id: %u gather: buffer_send.size(); %lu bytes: %lu\n", host_id, buffer_send.size(), MPI_Instance::get_sending_size(buffer_send));
////            }
//            end_got_candidates_queue = 0; // Set the got_candidates_queue empty
//            // Sync the dist_table
//            for (int root = 0; root < num_hosts; ++root) {
//                std::vector<std::pair<VertexID, VertexID>> buffer_recv;
//                one_host_bcasts_buffer_to_buffer(root,
//                                                 buffer_send,
//                                                 buffer_recv);
//                if (buffer_recv.empty()) {
//                    continue;
//                }
//                for (const std::pair<VertexID, VertexID> &e : buffer_recv) {
//                    VertexID root_id = e.first;
//                    VertexID cand_real_id = e.second;
//                    dist_table[root_id][cand_real_id] = iter;
//                    // Record the received element, for future reset
//                    recved_dist_table[root_id].push_back(cand_real_id);
//                }
//            }
//
//            // Sync the global_num_actives
//            MPI_Allreduce(&end_active_queue,
//                    &global_num_actives,
//                    1,
//                    V_ID_Type,
//                    MPI_SUM,
//                    MPI_COMM_WORLD);
//            gather_time += WallTimer::get_time_mark();
//		}
//    }
//
//    // Reset the dist_table
//    clearup_time -= WallTimer::get_time_mark();
//    reset_at_end(
////            G,
////            roots_start,
////            roots_master_local,
//            dist_table,
//            recved_dist_table,
//            bp_labels_table);
//    clearup_time += WallTimer::get_time_mark();
//}

//// Function: every host broadcasts its sending buffer, and does fun for every element it received in the unit buffer.
//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//template <typename E_T, typename F>
//inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::
//every_host_bcasts_buffer_and_proc(
//        std::vector<E_T> &buffer_send,
//        F &fun)
//{
//    // Every host h_i broadcast to others
//    for (int root = 0; root < num_hosts; ++root) {
//        std::vector<E_T> buffer_recv;
//        one_host_bcasts_buffer_to_buffer(root,
//                buffer_send,
//                buffer_recv);
//        if (buffer_recv.empty()) {
//            continue;
//        }
////        uint64_t size_buffer_send = buffer_send.size();
////        // Sync the size_buffer_send.
////        message_time -= WallTimer::get_time_mark();
////        MPI_Bcast(&size_buffer_send,
////                  1,
////                  MPI_UINT64_T,
////                  root,
////                  MPI_COMM_WORLD);
////        message_time += WallTimer::get_time_mark();
//////        {// test
//////            printf("host_id: %u h_i: %u bcast_buffer_send.size(): %lu\n", host_id, h_i, size_buffer_send);
//////        }
////        if (!size_buffer_send) {
////            continue;
////        }
////        message_time -= WallTimer::get_time_mark();
////        std::vector<E_T> buffer_recv(size_buffer_send);
////        if (host_id == root) {
////            buffer_recv.assign(buffer_send.begin(), buffer_send.end());
////        }
////        uint64_t bytes_buffer_send = size_buffer_send * ETypeSize;
////        if (bytes_buffer_send < static_cast<size_t>(INT_MAX)) {
////            // Only need 1 broadcast
////
////            MPI_Bcast(buffer_recv.data(),
////                     bytes_buffer_send,
////                     MPI_CHAR,
////                     root,
////                     MPI_COMM_WORLD);
////        } else {
////            const uint32_t num_unit_buffers = ((bytes_buffer_send - 1) / static_cast<size_t>(INT_MAX)) + 1;
////            const uint64_t unit_buffer_size = ((size_buffer_send - 1) / num_unit_buffers) + 1;
////            size_t offset = 0;
////            for (uint64_t b_i = 0; b_i < num_unit_buffers; ++b_i) {
//////                size_t offset = b_i * unit_buffer_size;
////                size_t size_unit_buffer = b_i == num_unit_buffers - 1
////                                          ? size_buffer_send - offset
////                                          : unit_buffer_size;
////                MPI_Bcast(buffer_recv.data() + offset,
////                         size_unit_buffer * ETypeSize,
////                         MPI_CHAR,
////                         root,
////                         MPI_COMM_WORLD);
////                offset += unit_buffer_size;
////            }
////        }
////        message_time += WallTimer::get_time_mark();
//        for (const E_T &e : buffer_recv) {
//            fun(e);
//        }
//    }
//}
//// Function: every host broadcasts its sending buffer, and does fun for every element it received in the unit buffer.
//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//template <typename E_T, typename F>
//inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::
//every_host_bcasts_buffer_and_proc(
//        std::vector<E_T> &buffer_send,
//        F &fun)
//{
//    // Host processes locally.
//    for (const E_T &e : buffer_send) {
//        fun(e);
//    }
//
//    // Every host sends to others
//    for (int src = 0; src < num_hosts; ++src) {
//        if (host_id == src) {
//            // Send from src
//            message_time -= WallTimer::get_time_mark();
//            for (int hop = 1; hop < num_hosts; ++hop) {
//                int dst = hop_2_root_host_id(hop, host_id);
//                MPI_Instance::send_buffer_2_dst(buffer_send,
//                        dst,
//                        SENDING_BUFFER_SEND,
//                        SENDING_SIZE_BUFFER_SEND);
//            }
//            message_time += WallTimer::get_time_mark();
//        } else {
//            // Receive from src
//            for (int hop = 1; hop < num_hosts; ++hop) {
//                int dst = hop_2_root_host_id(hop, src);
//                if (host_id == dst) {
//                    message_time -= WallTimer::get_time_mark();
//                    std::vector<E_T> buffer_recv;
//                    MPI_Instance::recv_buffer_from_src(buffer_recv,
//                            src,
//                            SENDING_BUFFER_SEND,
//                            SENDING_SIZE_BUFFER_SEND);
//                    message_time += WallTimer::get_time_mark();
//                    // Process
//                    for (const E_T &e : buffer_recv) {
//                        fun(e);
//                    }
//                }
//            }
//        }
//    }
//}
//// Function: every host broadcasts its sending buffer, and does fun for every element it received in the unit buffer.
//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//template <typename E_T, typename F>
//inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::
//every_host_bcasts_buffer_and_proc(
//        std::vector<E_T> &buffer_send,
//        F &fun)
//{
//    // Host processes locally.
//    for (const E_T &e : buffer_send) {
//        fun(e);
//    }
//    // Every host sends (num_hosts - 1) times
//    for (int hop = 1; hop < num_hosts; ++hop) {
//        int src = hop_2_me_host_id(-hop);
//        int dst = hop_2_me_host_id(hop);
//        if (src != dst) { // Normal case
//            // When host_id is odd, first receive, then send.
//            if (static_cast<uint32_t>(host_id) & 1U) {
//                message_time -= WallTimer::get_time_mark();
//                // Receive first.
//                std::vector<E_T> buffer_recv;
//                MPI_Instance::recv_buffer_from_src(buffer_recv,
//                                                   src,
//                                                   SENDING_BUFFER_SEND,
//                                                   SENDING_SIZE_BUFFER_SEND);
//                {//test
//                    printf("host_id: %u recved_from: %u\n", host_id, src);
//                }
//                // Send then.
//                MPI_Instance::send_buffer_2_dst(buffer_send,
//                                                dst,
//                                                SENDING_BUFFER_SEND,
//                                                SENDING_SIZE_BUFFER_SEND);
//                {//test
//                    printf("host_id: %u send_to: %u\n", host_id, dst);
//                }
//                message_time += WallTimer::get_time_mark();
//                // Process
//                if (buffer_recv.empty()) {
//                    continue;
//                }
//                for (const E_T &e : buffer_recv) {
//                    fun(e);
//                }
//            } else { // When host_id is even, first send, then receive.
//                // Send first.
//                message_time -= WallTimer::get_time_mark();
//                MPI_Instance::send_buffer_2_dst(buffer_send,
//                                                dst,
//                                                SENDING_BUFFER_SEND,
//                                                SENDING_SIZE_BUFFER_SEND);
//                {//test
//                    printf("host_id: %u send_to: %u\n", host_id, dst);
//                }
//                // Receive then.
//                std::vector<E_T> buffer_recv;
//                MPI_Instance::recv_buffer_from_src(buffer_recv,
//                                                   src,
//                                                   SENDING_BUFFER_SEND,
//                                                   SENDING_SIZE_BUFFER_SEND);
//                {//test
//                    printf("host_id: %u recved_from: %u\n", host_id, src);
//                }
//                message_time += WallTimer::get_time_mark();
//                // Process
//                if (buffer_recv.empty()) {
//                    continue;
//                }
//                for (const E_T &e : buffer_recv) {
//                    fun(e);
//                }
//            }
//        } else { // If host_id is higher than dst, first send, then receive
//            // This is a special case. It only happens when the num_hosts is even and hop equals to num_hosts/2.
//            if (host_id < dst) {
//                // Send
//                message_time -= WallTimer::get_time_mark();
//                MPI_Instance::send_buffer_2_dst(buffer_send,
//                                                dst,
//                                                SENDING_BUFFER_SEND,
//                                                SENDING_SIZE_BUFFER_SEND);
//                // Receive
//                std::vector<E_T> buffer_recv;
//                MPI_Instance::recv_buffer_from_src(buffer_recv,
//                                                   src,
//                                                   SENDING_BUFFER_SEND,
//                                                   SENDING_SIZE_BUFFER_SEND);
//                message_time += WallTimer::get_time_mark();
//                // Process
//                if (buffer_recv.empty()) {
//                    continue;
//                }
//                for (const E_T &e : buffer_recv) {
//                    fun(e);
//                }
//            } else { // Otherwise, if host_id is lower than dst, first receive, then send
//                // Receive
//                message_time -= WallTimer::get_time_mark();
//                std::vector<E_T> buffer_recv;
//                MPI_Instance::recv_buffer_from_src(buffer_recv,
//                                                   src,
//                                                   SENDING_BUFFER_SEND,
//                                                   SENDING_SIZE_BUFFER_SEND);
//                // Send
//                MPI_Instance::send_buffer_2_dst(buffer_send,
//                                                dst,
//                                                SENDING_BUFFER_SEND,
//                                                SENDING_SIZE_BUFFER_SEND);
//                message_time += WallTimer::get_time_mark();
//                // Process
//                if (buffer_recv.empty()) {
//                    continue;
//                }
//                for (const E_T &e : buffer_recv) {
//                    fun(e);
//                }
//            }
//        }
//    }
//}
//// DEPRECATED version Function: every host broadcasts its sending buffer, and does fun for every element it received in the unit buffer.
//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//template <typename E_T, typename F>
//inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::
//every_host_bcasts_buffer_and_proc(
//        std::vector<E_T> &buffer_send,
//        F &fun)
//{
//    const uint32_t UNIT_BUFFER_SIZE = 16U << 20U;
//    // Every host h_i broadcast to others
//    for (int h_i = 0; h_i < num_hosts; ++h_i) {
//        uint64_t size_buffer_send = buffer_send.size();
//        // Sync the size_buffer_send.
//        message_time -= WallTimer::get_time_mark();
//        MPI_Bcast(&size_buffer_send,
//                  1,
//                  MPI_UINT64_T,
//                  h_i,
//                  MPI_COMM_WORLD);
//        message_time += WallTimer::get_time_mark();
////        {// test
////            printf("host_id: %u h_i: %u bcast_buffer_send.size(): %lu\n", host_id, h_i, size_buffer_send);
////        }
//        if (!size_buffer_send) {
//            continue;
//        }
//        uint32_t num_unit_buffers = (size_buffer_send + UNIT_BUFFER_SIZE - 1) / UNIT_BUFFER_SIZE;
//
//        // Broadcast the buffer_send
//        for (uint32_t b_i = 0; b_i < num_unit_buffers; ++b_i) {
//            // Prepare the unit buffer
//            message_time -= WallTimer::get_time_mark();
//            size_t offset = b_i * UNIT_BUFFER_SIZE;
//            size_t size_unit_buffer = b_i == num_unit_buffers - 1
//                                        ? size_buffer_send - offset
//                                        : UNIT_BUFFER_SIZE;
//            std::vector<E_T> unit_buffer(size_unit_buffer);
//            // Copy the messages from buffer_send to unit buffer.
//            if (host_id == h_i) {
//                unit_buffer.assign(buffer_send.begin() + offset, buffer_send.begin() + offset + size_unit_buffer);
//            }
//            // Broadcast the unit buffer
//            MPI_Bcast(unit_buffer.data(),
//                    MPI_Instance::get_sending_size(unit_buffer),
//                    MPI_CHAR,
//                    h_i,
//                    MPI_COMM_WORLD);
//            message_time += WallTimer::get_time_mark();
//            // Process every element of unit_buffer
//            for (const E_T &e : unit_buffer) {
//                fun(e);
//            }
//        }
//    }
//}

// Function: Host root broadcasts its sending buffer to a receiving buffer.
template <VertexID BATCH_SIZE>
template <typename E_T>
inline void DistBVCPLL<BATCH_SIZE>::
one_host_bcasts_buffer_to_buffer(
        int root,
        std::vector<E_T> &buffer_send,
        std::vector<E_T> &buffer_recv)
{
    const size_t ETypeSize = sizeof(E_T);
    volatile uint64_t size_buffer_send = 0;
    if (host_id == root) {
        size_buffer_send = buffer_send.size();
    }
    // Sync the size_buffer_send.
//    message_time -= WallTimer::get_time_mark();
//    {//test
//        if (0 == root && size_buffer_send == 16 && 1024 == caller_line) {
////        if (0 == root && size_buffer_send == 16 && 0 == host_id) {
//            printf("before: host_id: %d size_buffer_send: %lu\n",
//                    host_id,
//                    size_buffer_send);
//        }
//    }
    MPI_Bcast((void *) &size_buffer_send,
              1,
              MPI_UINT64_T,
              root,
              MPI_COMM_WORLD);
//    message_time += WallTimer::get_time_mark();
//    {//test
////        if (0 == root && size_buffer_send == 16 && 0 == host_id) {
//        if (0 == root && size_buffer_send == 16 && 1024 == caller_line) {
//            printf("after: host_id: %d size_buffer_send: %lu\n",
//                    host_id,
//                    size_buffer_send);
//        }
//    }
    try {
        buffer_recv.resize(size_buffer_send);
    }
    catch (const std::bad_alloc &) {
        double memtotal = 0;
        double memfree = 0;
        PADO::Utils::system_memory(memtotal, memfree);
        printf("one_host_bcasts_buffer_to_buffer: bad_alloc "
               "host_id: %d "
               "L.size(): %.2fGB "
               "memtotal: %.2fGB "
               "memfree: %.2fGB\n",
               host_id,
               get_index_size() * 1.0 / (1 << 30),
               memtotal / 1024,
               memfree / 1024);
        exit(1);
    }

    if (!size_buffer_send) {
        return;
    }
    // Broadcast the buffer_send
//    message_time -= WallTimer::get_time_mark();
    if (host_id == root) {
//        buffer_recv.assign(buffer_send.begin(), buffer_send.end());
        buffer_recv.swap(buffer_send);
    }
    uint64_t bytes_buffer_send = size_buffer_send * ETypeSize;
    if (bytes_buffer_send <= static_cast<size_t>(INT_MAX)) {
        // Only need 1 broadcast
        MPI_Bcast(buffer_recv.data(),
                  bytes_buffer_send,
                  MPI_CHAR,
                  root,
                  MPI_COMM_WORLD);
    } else {
        const uint32_t num_unit_buffers = ((bytes_buffer_send - 1) / static_cast<size_t>(INT_MAX)) + 1;
        const uint64_t unit_buffer_size = ((size_buffer_send - 1) / num_unit_buffers) + 1;
        size_t offset = 0;
        for (uint64_t b_i = 0; b_i < num_unit_buffers; ++b_i) {
            size_t size_unit_buffer = b_i == num_unit_buffers - 1
                                      ? size_buffer_send - offset
                                      : unit_buffer_size;
            MPI_Bcast(buffer_recv.data() + offset,
                      size_unit_buffer * ETypeSize,
                      MPI_CHAR,
                      root,
                      MPI_COMM_WORLD);
            offset += unit_buffer_size;
        }
    }
//    message_time += WallTimer::get_time_mark();
}

}

#endif //PADO_DPADO_H
