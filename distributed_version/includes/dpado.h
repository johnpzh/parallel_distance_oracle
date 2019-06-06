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
//    VertexID num_v_ = 0;

    // Structure for the type of label
    struct IndexType {
        struct Batch {
            VertexID batch_id; // Batch ID
            VertexID start_index; // Index to the array distances where the batch starts
            VertexID size; // Number of distances element in this batch

            Batch(VertexID batch_id_, VertexID start_index_, VertexID size_):
                    batch_id(batch_id_), start_index(start_index_), size(size_)
            { }
        };

        struct DistanceIndexType {
            VertexID start_index; // Index to the array vertices where the same-ditance vertices start
            VertexID size; // Number of the same-distance vertices
            UnweightedDist dist; // The real distance

            DistanceIndexType(VertexID start_index_, VertexID size_, smalli dist_):
                    start_index(start_index_), size(size_), dist(dist_)
            { }
        };
//        // Bit-parallel Labels
//        UnweightedDist bp_dist[BITPARALLEL_SIZE];
//        uint64_t bp_sets[BITPARALLEL_SIZE][2];  // [0]: S^{-1}, [1]: S^{0}

        vector<Batch> batches; // Batch info
        vector<DistanceIndexType> distances; // Distance info
        vector<VertexID> vertices; // Vertices in the label, presented as temporary ID

//        // Clean up all labels
//        void cleanup()
//        {
//            vector<Batch>().swap(batches);
//            vector<DistanceIndexType>().swap(distances);
//            vector<VertexID>().swap(vertices);
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
        vector<VertexID> candidates_que = vector<VertexID>(BATCH_SIZE);
        VertexID end_candidates_que = 0;
        vector<bool> is_candidate = vector<bool>(BATCH_SIZE, false);

    }; //__attribute__((aligned(64)));

    // Structure of the public ordered index for distance queries.
    struct IndexOrdered {
        UnweightedDist bp_dist[BITPARALLEL_SIZE];
        uint64_t bp_sets[BITPARALLEL_SIZE][2]; // [0]: S^{-1}, [1]: S^{0}

        vector<VertexID> label_id;
        vector<UnweightedDist> label_dists;
    };

    vector<IndexType> L;
    vector<IndexOrdered> Index; // Ordered labels for original vertex ID

    void construct(const DistGraph &G);
//    inline void bit_parallel_labeling(
//            const DistGraph &G,
//            vector<IndexType> &L,
//            vector<bool> &used_bp_roots);
//    inline bool bit_parallel_checking(
//            VertexID v_id,
//            VertexID w_id,
//            const vector<IndexType> &L,
//            UnweightedDist iter);

    inline void batch_process(
            const DistGraph &G,
            VertexID b_id,
            VertexID roots_start,
            VertexID roots_size,
//            vector<IndexType> &L,
            const vector<bool> &used_bp_roots,
            vector<VertexID> &active_queue,
            VertexID &end_active_queue,
            vector<VertexID> &got_candidates_queue,
            VertexID &end_got_candidates_queue,
            vector<ShortIndex> &short_index,
            vector< vector<UnweightedDist> > &dist_matrix,
            vector<bool> &got_candidates,
            vector<bool> &is_active,
            vector<VertexID> &once_candidated_queue,
            VertexID &end_once_candidated_queue,
            vector<bool> &once_candidated);

    inline void initialization(
            const DistGraph &G,
            vector<ShortIndex> &short_index,
            vector< vector<UnweightedDist> > &dist_matrix,
            vector<VertexID> &active_queue,
            VertexID &end_active_queue,
            vector<VertexID> &once_candidated_queue,
            VertexID &end_once_candidated_queue,
            vector<bool> &once_candidated,
            VertexID b_id,
            VertexID roots_start,
            VertexID roots_size,
//            vector<IndexType> &L,
            const vector<bool> &used_bp_roots);
    inline void push_labels(
            VertexID v_head,
            VertexID roots_start,
            const DistGraph &G,
//            const vector<IndexType> &L,
            vector<ShortIndex> &short_index,
            vector<VertexID> &got_candidates_queue,
            VertexID &end_got_candidates_queue,
            vector<bool> &got_candidates,
            vector<VertexID> &once_candidated_queue,
            VertexID &end_once_candidated_queue,
            vector<bool> &once_candidated,
            const vector<bool> &used_bp_roots,
            UnweightedDist iter);
    inline bool distance_query(
            VertexID cand_root_id,
            VertexID v_id,
            VertexID roots_start,
//            const vector<IndexType> &L,
            const vector< vector<UnweightedDist> > &dist_matrix,
            UnweightedDist iter);
    inline void insert_label_only(
            VertexID cand_root_id,
            VertexID v_id,
            VertexID roots_start,
            VertexID roots_size,
//            vector<IndexType> &L,
            vector< vector<UnweightedDist> > &dist_matrix,
            UnweightedDist iter);
    inline void update_label_indices(
            VertexID v_id,
            VertexID inserted_count,
//            vector<IndexType> &L,
            vector<ShortIndex> &short_index,
            VertexID b_id,
            UnweightedDist iter);
    inline void reset_at_end(
            VertexID roots_start,
            VertexID roots_size,
//            vector<IndexType> &L,
            vector< vector<UnweightedDist> > &dist_matrix);

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
            const vector<VertexID> &rank2id);
//            const vector<VertexID> &rank);
    void store_index_to_file(
            const char *filename,
            const vector<VertexID> &rank);
    void load_index_from_file(
            const char *filename);
    void order_labels(
            const vector<VertexID> &rank2id);
//            const vector<VertexID> &rank);
    UnweightedDist query_distance(
            VertexID a,
            VertexID b);
}; // class DistBVCPLL

//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//const VertexID DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::BITPARALLEL_SIZE;

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::DistBVCPLL(const DistGraph &G)
{
    construct(G);
}

//template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
//inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::bit_parallel_labeling(
//        const DistGraph &G,
//        vector<IndexType> &L,
//        vector<bool> &used_bp_roots)
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
//        const vector<IndexType> &L,
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
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::initialization(
        const DistGraph &G,
        vector<ShortIndex> &short_index,
        vector< vector<UnweightedDist> > &dist_matrix,
        vector<VertexID> &active_queue,
        VertexID &end_active_queue,
        vector<VertexID> &once_candidated_queue,
        VertexID &end_once_candidated_queue,
        vector<bool> &once_candidated,
        VertexID b_id,
        VertexID roots_start,
        VertexID roots_size,
//        vector<IndexType> &L,
        const vector<bool> &used_bp_roots)
{
    MPI_Datatype VIDType = MPI_Instance::get_mpi_datatype<VertexID>();
    VertexID roots_bound = roots_start + roots_size;
    std::vector<VertexID> roots_master_local; // Roots which belongs to this host.
    for (VertexID r_global = roots_start; r_global < roots_bound; ++r_global) {
        if (G.get_master_host_id(r_global) == host_id && !used_bp_roots[r_global]) {
            roots_master_local.push_back(G.get_local_vertex_id(r_global));
        }
    }
//	init_start_reset_time -= WallTimer::get_time_mark();
    // TODO: parallel enqueue
    // Active_queue
    {
        for (VertexID r_local : roots_master_local) {
            active_queue[end_active_queue++] = r_local;
        }
//        for (VertexID r_real_id = roots_start; r_real_id < roots_bound; ++r_real_id) {
//            if (!used_bp_roots[r_real_id]) {
//                active_queue[end_active_queue++] = r_real_id;
//            }
//        }
    }
//	init_start_reset_time += WallTimer::get_time_mark();
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
            Lr.vertices.push_back(G.get_global_vertex_id(r_local));
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
    // Dist_matrix
    {
//        using Label = std::pair<VertexID, UnweightedDist>;
        // Deprecated Old method: unpack the IndexType structure before sending.
//        struct Label {
//            VertexID root_id;
//            VertexID label_global_id;
//            UnweightedDist dist;
//        };
//        std::vector<Label> buffer_send; // buffer for sending, (root, dis
//		IndexType &Lr;
        VertexID b_i_bound;
        VertexID id_offset;
        VertexID dist_start_index;
        VertexID dist_bound_index;
        VertexID v_start_index;
        VertexID v_bound_index;
        UnweightedDist dist;
//        for (VertexID r_id = 0; r_id < roots_size; ++r_id) {
//            if (used_bp_roots[r_id + roots_start]) {
//                continue;
//            }
//            IndexType &Lr = L[r_id + roots_start];
        for (VertexID r_local : roots_master_local) {
            // The distance table.
            IndexType &Lr = L[r_local];
            VertexID r_root_id = G.get_global_vertex_id(r_local) - roots_start;
            b_i_bound = Lr.batches.size();
            _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
            _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
            _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
            for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
                id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
                dist_start_index = Lr.batches[b_i].start_index;
                dist_bound_index = dist_start_index + Lr.batches[b_i].size;
                // Traverse dist_matrix
                for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                    v_start_index = Lr.distances[dist_i].start_index;
                    v_bound_index = v_start_index + Lr.distances[dist_i].size;
                    dist = Lr.distances[dist_i].dist;
                    for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
//                        VertexID label_global_id = Lr.vertices[v_i] + id_offset;
                        dist_matrix[r_root_id][Lr.vertices[v_i] + id_offset] = dist; // distance table
//                        buffer_send.emplace_back(r_root_id, label_global_id, dist); // buffer for sending
                    }
                }
            }
        }
//        }
    }
//	init_dist_matrix_time += WallTimer::get_time_mark();
    // Broadcast local roots labels
    {
        for (int loc = 0; loc < num_hosts - 1; ++loc) {
            int dest_host_id = G.buffer_send_list_loc_2_master_host_id(loc);
            // How many masters to be sent
            VertexID num_root_masters = roots_master_local.size();
            MPI_Send(&num_root_masters,
                    1,
                    VIDType,
                    dest_host_id,
                    SENDING_NUM_ROOT_MASTERS,
                    MPI_COMM_WORLD);
            // For every root master
            for (VertexID r_local : roots_master_local) {
                // Which root
                IndexType &Lr = L[r_local];
                VertexID r_root_id = G.get_global_vertex_id(r_local) - roots_start;
                MPI_Send(&r_root_id,
                        1,
                        VIDType,
                        dest_host_id,
                        SENDING_ROOT_ID,
                        MPI_COMM_WORLD);
                // The Batches array
                MPI_Send(Lr.batches.data(),
                         Lr.batches.size() * sizeof(Lr.batches[0]),
                         MPI_CHAR,
                         dest_host_id,
                         SENDING_INDEXTYPE_BATCH,
                         MPI_COMM_WORLD);
                // The Distances array
                MPI_Send(Lr.distances.data(),
                         Lr.distances.size() * sizeof(Lr.distances[0]),
                         MPI_CHAR,
                         dest_host_id,
                         SENDING_INDEXTYPE_DISTANCE,
                         MPI_COMM_WORLD);
                // The Vertices arrray
                MPI_Send(Lr.vertices.data(),
                         Lr.vertices.size() * sizeof(Lr.vertices[0]),
                         MPI_CHAR,
                         dest_host_id,
                         SENDING_INDEXTYPE_VERTICES,
                         MPI_COMM_WORLD);
            }
        }
    }

    // Receive labels from every other host
    {
        for (int h_i = 0; h_i < num_hosts - 1; ++h_i) {
            VertexID num_root_recieved;
//            MPI_Recv(&num_root_recieved,
//                    1,
//                    VIDType,
//                    )
        }
    }
}

// Function: (sequential version) pushes v_head's labels to v_head's every neighbor
template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::push_labels(
        VertexID v_head,
        VertexID roots_start,
        const DistGraph &G,
//        const vector<IndexType> &L,
        vector<ShortIndex> &short_index,
        vector<VertexID> &got_candidates_queue,
        VertexID &end_got_candidates_queue,
        vector<bool> &got_candidates,
        vector<VertexID> &once_candidated_queue,
        VertexID &end_once_candidated_queue,
        vector<bool> &once_candidated,
        const vector<bool> &used_bp_roots,
        UnweightedDist iter)
{
    const IndexType &Lv = L[v_head];
    // These 2 index are used for traversing v_head's last inserted labels
    VertexID l_i_start = Lv.distances.rbegin() -> start_index;
    VertexID l_i_bound = l_i_start + Lv.distances.rbegin() -> size;
    // Traverse v_head's every neighbor v_tail
    EdgeID e_i_start = G.vertices_idx[v_head];
    EdgeID e_i_bound = e_i_start + G.out_degrees[v_head];
    for (EdgeID e_i = e_i_start; e_i < e_i_bound; ++e_i) {
        idi v_tail = G.out_edges[e_i];

        if (used_bp_roots[v_tail]) {
            continue;
        }
//		if (used_bp_roots[v_head]) {
//			continue;
//		}

        if (v_tail < roots_start) { // v_tail has higher rank than any roots, then no roots can push new labels to it.
            return;
        }
//		if (v_tail <= Lv.vertices[l_i_start] + roots_start) { // v_tail has higher rank than any v_head's labels
//			return;
//		} // This condition cannot be used anymore since v_head's last inserted labels are not ordered from higher rank to lower rank now, because v_head's candidate set is a queue now rather than a bitmap. For a queue, its order of candidates are not ordered by ranks.
        const IndexType &L_tail = L[v_tail];
//        _mm_prefetch(&L_tail.bp_dist[0], _MM_HINT_T0);
//        _mm_prefetch(&L_tail.bp_sets[0][0], _MM_HINT_T0);
        // Traverse v_head's last inserted labels
        for (VertexID l_i = l_i_start; l_i < l_i_bound; ++l_i) {
            VertexID label_root_id = Lv.vertices[l_i];
            VertexID label_real_id = label_root_id + roots_start;
            if (v_tail <= label_real_id) {
                // v_tail has higher rank than all remaining labels
                // For candidates_que, this is not true any more!
//				break;
                continue;
            }
            ShortIndex &SI_v_tail = short_index[v_tail];
            if (SI_v_tail.indicator[label_root_id]) {
                // The label is already selected before
                continue;
            }
            // Record label_root_id as once selected by v_tail
            SI_v_tail.indicator.set(label_root_id);
            // Add into once_candidated_queue
            if (!once_candidated[v_tail]) {
                // If v_tail is not in the once_candidated_queue yet, add it in
                once_candidated[v_tail] = true;
                once_candidated_queue[end_once_candidated_queue++] = v_tail;
            }

//            // Bit Parallel Checking: if label_real_id to v_tail has shorter distance already
//            //			++total_check_count;
//            const IndexType &L_label = L[label_real_id];
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

            // Record vertex label_root_id as v_tail's candidates label
//			SI_v_tail.candidates.set(label_root_id);
            if (!SI_v_tail.is_candidate[label_root_id]) {
                SI_v_tail.is_candidate[label_root_id] = true;
                SI_v_tail.candidates_que[SI_v_tail.end_candidates_que++] = label_root_id;
            }

            // Add into got_candidates_queue
            if (!got_candidates[v_tail]) {
                // If v_tail is not in got_candidates_queue, add it in (prevent duplicate)
                got_candidates[v_tail] = true;
                got_candidates_queue[end_got_candidates_queue++] = v_tail;
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
        VertexID v_id,
        VertexID roots_start,
//        const vector<IndexType> &L,
        const vector< vector<UnweightedDist> > &dist_matrix,
        UnweightedDist iter)
{
//	++total_check_count;
//	++normal_check_count;
//	distance_query_time -= WallTimer::get_time_mark();
//	dist_query_ins_count.measure_start();

    VertexID cand_real_id = cand_root_id + roots_start;
    const IndexType &Lv = L[v_id];

    // Traverse v_id's all existing labels
    VertexID b_i_bound = Lv.batches.size();
    _mm_prefetch(&Lv.batches[0], _MM_HINT_T0);
    _mm_prefetch(&Lv.distances[0], _MM_HINT_T0);
    _mm_prefetch(&Lv.vertices[0], _MM_HINT_T0);
    //_mm_prefetch(&dist_matrix[cand_root_id][0], _MM_HINT_T0);
    for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
        VertexID id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
        VertexID dist_start_index = Lv.batches[b_i].start_index;
        VertexID dist_bound_index = dist_start_index + Lv.batches[b_i].size;
        // Traverse dist_matrix
        for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
            UnweightedDist dist = Lv.distances[dist_i].dist;
            if (dist >= iter) { // In a batch, the labels' distances are increasingly ordered.
                // If the half path distance is already greater than their targeted distance, jump to next batch
                break;
            }
            VertexID v_start_index = Lv.distances[dist_i].start_index;
            VertexID v_bound_index = v_start_index + Lv.distances[dist_i].size;
//            _mm_prefetch(&dist_matrix[cand_root_id][0], _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(dist_matrix[cand_root_id].data()), _MM_HINT_T0);
            for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                VertexID v = Lv.vertices[v_i] + id_offset; // v is a label hub of v_id
                if (v >= cand_real_id) {
                    // Vertex cand_real_id cannot have labels whose ranks are lower than it,
                    // in which case dist_matrix[cand_root_id][v] does not exist.
                    continue;
                }
                VertexID d_tmp = dist + dist_matrix[cand_root_id][v];
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
// update the distance buffer dist_matrix;
// but it only update the v_id's labels' vertices array;
template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::insert_label_only(
        VertexID cand_root_id,
        VertexID v_id,
        VertexID roots_start,
        VertexID roots_size,
//        vector<IndexType> &L,
        vector< vector<UnweightedDist> > &dist_matrix,
        UnweightedDist iter)
{
    L[v_id].vertices.push_back(cand_root_id);
    // Update the distance buffer if necessary
    VertexID v_root_id = v_id - roots_start;
    if (v_id >= roots_start && v_root_id < roots_size) {
        dist_matrix[v_root_id][cand_root_id + roots_start] = iter;
    }
}

// Function updates those index arrays in v_id's label only if v_id has been inserted new labels
template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::update_label_indices(
        VertexID v_id,
        VertexID inserted_count,
//        vector<IndexType> &L,
        vector<ShortIndex> &short_index,
        VertexID b_id,
        UnweightedDist iter)
{
    IndexType &Lv = L[v_id];
    // indicator[BATCH_SIZE + 1] is true, means v got some labels already in this batch
    if (short_index[v_id].indicator[BATCH_SIZE]) {
        // Increase the batches' last element's size because a new distance element need to be added
        ++(Lv.batches.rbegin() -> size);
    } else {
        short_index[v_id].indicator.set(BATCH_SIZE);
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
template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::reset_at_end(
        VertexID roots_start,
        VertexID roots_size,
//        vector<IndexType> &L,
        vector< vector<UnweightedDist> > &dist_matrix)
{
    VertexID b_i_bound;
    VertexID id_offset;
    VertexID dist_start_index;
    VertexID dist_bound_index;
    VertexID v_start_index;
    VertexID v_bound_index;
    for (VertexID r_id = 0; r_id < roots_size; ++r_id) {
        IndexType &Lr = L[r_id + roots_start];
        b_i_bound = Lr.batches.size();
        _mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
        _mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
        _mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
        for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
            id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
            dist_start_index = Lr.batches[b_i].start_index;
            dist_bound_index = dist_start_index + Lr.batches[b_i].size;
            // Traverse dist_matrix
            for (VertexID dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
                v_start_index = Lr.distances[dist_i].start_index;
                v_bound_index = v_start_index + Lr.distances[dist_i].size;
                for (VertexID v_i = v_start_index; v_i < v_bound_index; ++v_i) {
                    dist_matrix[r_id][Lr.vertices[v_i] + id_offset] = MAX_UNWEIGHTED_DIST;
                }
            }
        }

        // Cleanup for large graphs. 02/17/2019
//		Lr.cleanup();
    }
}

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
inline void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::batch_process(
        const DistGraph &G,
        VertexID b_id,
        VertexID roots_start, // start id of roots
        VertexID roots_size, // how many roots in the batch
//        vector<IndexType> &L,
        const vector<bool> &used_bp_roots,
        vector<VertexID> &active_queue,
        VertexID &end_active_queue,
        vector<VertexID> &got_candidates_queue,
        VertexID &end_got_candidates_queue,
        vector<ShortIndex> &short_index,
        vector< vector<UnweightedDist> > &dist_matrix,
        vector<bool> &got_candidates,
        vector<bool> &is_active,
        vector<VertexID> &once_candidated_queue,
        VertexID &end_once_candidated_queue,
        vector<bool> &once_candidated)
{

//	initializing_time -= WallTimer::get_time_mark();
    // At the beginning of a batch, initialize the labels L and distance buffer dist_matrix;
//	puts("Initializing...");//test
    initialization(
            G,
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
//            L,
            used_bp_roots);
//	puts("Initial done.");//test

    UnweightedDist iter = 0; // The iterator, also the distance for current iteration
//	initializing_time += WallTimer::get_time_mark();


    while (0 != end_active_queue) {
//		candidating_time -= WallTimer::get_time_mark();
//		candidating_ins_count.measure_start();
        ++iter;
//		printf("iter: %u\n", iter);//test
        // Traverse active vertices to push their labels as candidates
//		puts("Pushing...");//test
        for (VertexID i_queue = 0; i_queue < end_active_queue; ++i_queue) {
            VertexID v_head = active_queue[i_queue];
            is_active[v_head] = false; // reset is_active

            push_labels(
                    v_head,
                    roots_start,
                    G,
//                    L,
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
        end_active_queue = 0; // Set the active_queue empty
//		puts("Push done.");//test
//		candidating_ins_count.measure_stop();
//		candidating_time += WallTimer::get_time_mark();
//		adding_time -= WallTimer::get_time_mark();
//		adding_ins_count.measure_start();

        // Traverse vertices in the got_candidates_queue to insert labels
//		puts("Checking...");//test
        for (VertexID i_queue = 0; i_queue < end_got_candidates_queue; ++i_queue) {
            VertexID v_id = got_candidates_queue[i_queue];
            VertexID inserted_count = 0; //recording number of v_id's truly inserted candidates
            got_candidates[v_id] = false; // reset got_candidates
            // Traverse v_id's all candidates
//			total_candidates_num += roots_size;
//			for (VertexID cand_root_id = 0; cand_root_id < roots_size; ++cand_root_id) {
//				if (!short_index[v_id].candidates[cand_root_id]) {
//					// Root cand_root_id is not vertex v_id's candidate
//					continue;
//				}
//				++set_candidates_num;
//				short_index[v_id].candidates.reset(cand_root_id);
            VertexID bound_cand_i = short_index[v_id].end_candidates_que;
            for (VertexID cand_i = 0; cand_i < bound_cand_i; ++cand_i) {
                VertexID cand_root_id = short_index[v_id].candidates_que[cand_i];
                short_index[v_id].is_candidate[cand_root_id] = false;
                // Only insert cand_root_id into v_id's label if its distance to v_id is shorter than existing distance
                if ( distance_query(
                        cand_root_id,
                        v_id,
                        roots_start,
//                        L,
                        dist_matrix,
                        iter) ) {
                    if (!is_active[v_id]) {
                        is_active[v_id] = true;
                        active_queue[end_active_queue++] = v_id;
                    }
                    ++inserted_count;
                    // The candidate cand_root_id needs to be added into v_id's label
                    insert_label_only(
                            cand_root_id,
                            v_id,
                            roots_start,
                            roots_size,
//                            L,
                            dist_matrix,
                            iter);
                }
            }
            short_index[v_id].end_candidates_que = 0;
//			}
            if (0 != inserted_count) {
                // Update other arrays in L[v_id] if new labels were inserted in this iteration
                update_label_indices(
                        v_id,
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
    }

    // Reset the dist_matrix
//	initializing_time -= WallTimer::get_time_mark();
//	init_dist_matrix_time -= WallTimer::get_time_mark();
//	puts("Resetting...");//test
    reset_at_end(
            roots_start,
            roots_size,
//            L,
            dist_matrix);
//	puts("Reset done.");//test
//	init_dist_matrix_time += WallTimer::get_time_mark();
//	initializing_time += WallTimer::get_time_mark();


//	double total_time = time_can + time_add;
//	printf("Candidating time: %f (%f%%)\n", time_can, time_can / total_time * 100);
//	printf("Adding time: %f (%f%%)\n", time_add, time_add / total_time * 100);
}


template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::construct(const DistGraph &G)
{
    num_v = G.num_v;
    assert(num_v >= BATCH_SIZE);
    num_masters = G.num_masters;
    host_id = G.host_id;
    num_hosts = G.num_hosts;
//    L.resize(num_v);
    L.resize(num_masters);
    VertexID remainer = num_v % BATCH_SIZE;
    VertexID b_i_bound = num_v / BATCH_SIZE;
    vector<bool> used_bp_roots(num_v, false);
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

    vector<VertexID> active_queue(num_masters); // Any vertex v who is active should be put into this queue.
//    vector<VertexID> active_queue(num_v); // Any vertex v who is active should be put into this queue.
    VertexID end_active_queue = 0;
    vector<bool> is_active(num_masters, false);// is_active[v] is true means vertex v is in the active queue.
//    vector<bool> is_active(num_v, false);// is_active[v] is true means vertex v is in the active queue.
    vector<VertexID> got_candidates_queue(num_masters); // Any vertex v who got candidates should be put into this queue.
//    vector<VertexID> got_candidates_queue(num_v); // Any vertex v who got candidates should be put into this queue.
    VertexID end_got_candidates_queue = 0;
    vector<bool> got_candidates(num_masters, false); // got_candidates[v] is true means vertex v is in the queue got_candidates_queue
//    vector<bool> got_candidates(num_v, false); // got_candidates[v] is true means vertex v is in the queue got_candidates_queue
    vector<ShortIndex> short_index(num_masters);
//    vector<ShortIndex> short_index(num_v);
    vector< vector<UnweightedDist> > dist_matrix(BATCH_SIZE, vector<UnweightedDist>(num_v, MAX_UNWEIGHTED_DIST));


    vector<VertexID> once_candidated_queue(num_masters); // if short_index[v].indicator.any() is true, v is in the queue.
//    vector<VertexID> once_candidated_queue(num_v); // if short_index[v].indicator.any() is true, v is in the queue.
        // Used mainly for resetting short_index[v].indicator.
    VertexID end_once_candidated_queue = 0;
    vector<bool> once_candidated(num_masters, false);
//    vector<bool> once_candidated(num_v, false);

    //printf("b_i_bound: %u\n", b_i_bound);//test
    for (VertexID b_i = 0; b_i < b_i_bound; ++b_i) {
        printf("b_i: %u\n", b_i);//test
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
                dist_matrix,
                got_candidates,
                is_active,
                once_candidated_queue,
                end_once_candidated_queue,
                once_candidated);
    }
    if (remainer != 0) {
		printf("b_i: %u\n", b_i_bound);//test
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
                dist_matrix,
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
    printf("BATCH_SIZE: %u\n", BATCH_SIZE);
    printf("BP_Size: %u\n", BITPARALLEL_SIZE);
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

    printf("Total_labeling_time: %.2f seconds\n", time_labeling);
    // End test
}

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::store_index_to_file(
        const char *filename,
        const vector<VertexID> &rank)
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

        vector< std::pair<VertexID, UnweightedDist> > ordered_labels;
        // Traverse v_id's all existing labels
        for (VertexID b_i = 0; b_i < Lv.batches.size(); ++b_i) {
            VertexID id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
            VertexID dist_start_index = Lv.batches[b_i].start_index;
            VertexID dist_bound_index = dist_start_index + Lv.batches[b_i].size;
            // Traverse dist_matrix
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
        const vector<VertexID> &rank2id)
//        const vector<VertexID> &rank)
{
//    VertexID num_v = rank.size();
    vector< vector< std::pair<VertexID, UnweightedDist> > > ordered_L(num_v);
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
            // Traverse dist_matrix
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
//	vector<VertexID> &A = Ia.label_id;
//	vector<VertexID> &B = Ib.label_id;
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
        for (VertexID i = 0; i < Lu.size(); ++i) {
            markers[Lu[i].first] = Lu[i].second;
        }
        for (VertexID i = 0; i < Lv.size(); ++i) {
            const auto &tmp_l = markers.find(Lv[i].first);
            if (tmp_l == markers.end()) {
                continue;
            }
            int d = tmp_l->second + Lv[i].second;
            if (d < dist) {
                dist = d;
            }
        }
        if (dist == 255) {
            printf("2147483647\n");
        } else {
            printf("%u\n", dist);
        }
    }
}

template <VertexID BATCH_SIZE, VertexID BITPARALLEL_SIZE>
void DistBVCPLL<BATCH_SIZE, BITPARALLEL_SIZE>::switch_labels_to_old_id(
        const vector<VertexID> &rank2id)
//        const vector<VertexID> &rank)
{
    VertexID label_sum = 0;
    VertexID test_label_sum = 0;

//	VertexID num_v = rank2id.size();
//    VertexID num_v = rank.size();
    vector< vector< std::pair<VertexID, UnweightedDist> > > new_L(num_v);
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
            // Traverse dist_matrix
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

}


#endif //PADO_DPADO_H
