/*
 * pado.h
 *
 *  Created on: Sep 4, 2018
 *      Author: Zhen Peng
 */

#ifndef INCLUDES_PADO_H_
#define INCLUDES_PADO_H_

#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <iostream>
#include <limits.h>
#include <xmmintrin.h>
#include <bitset>
#include "globals.h"
#include "graph.h"
//#include "index.h"

using std::vector;
using std::unordered_map;
using std::map;
using std::bitset;
using std::stable_sort;
using std::min;
using std::fill;

namespace PADO {

const inti BATCH_SIZE = 1024; // The size for regular batch and bit array.
const inti BITPARALLEL_SIZE = 50;



//// Batch based processing, 09/11/2018
class VertexCentricPLL {
private:
	// Structure for the type of label
	struct IndexType {
		struct Batch {
			idi batch_id; // Batch ID
			idi start_index; // Index to the array distances where the batch starts
			inti size; // Number of distances element in this batch

			Batch(idi batch_id_, idi start_index_, inti size_):
						batch_id(batch_id_), start_index(start_index_), size(size_)
			{
				;
			}
		};

		struct DistanceIndexType {
			idi start_index; // Index to the array vertices where the same-ditance vertices start
			inti size; // Number of the same-distance vertices
			smalli dist; // The real distance

			DistanceIndexType(idi start_index_, inti size_, smalli dist_):
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
	} __attribute__((aligned(64)));

	// Structure for the type of temporary label
	struct ShortIndex {
		// I use BATCH_SIZE + 1 bit for indicator bit array.
		// The v.indicator[BATCH_SIZE] is set if in current batch v has got any new labels already.
		// In this way, when do initialization, only initialize those short_index[v] whose indicator[BATCH_SIZE] is set.
		bitset<BATCH_SIZE + 1> indicator; // Global indicator, indicator[r] (0 <= r < BATCH_SIZE) is set means root r once selected as candidate already
		bitset<BATCH_SIZE> candidates; // Candidates one iteration, candidates[r] is set means root r is candidate in this iteration
	} __attribute__((aligned(64)));

	vector<IndexType> L;
	void construct(const Graph &G);
	inline void bit_parallel_labeling(
			const Graph &G,
			vector<IndexType> &L,
			vector<bool> &used_bp_roots);
	inline bool bit_parallel_checking(
			idi v_id,
			idi w_id,
			const vector<IndexType> &L,
			smalli iter);

	inline void batch_process(
			const Graph &G,
			idi b_id,
			idi root_start,
			inti roots_size,
			vector<IndexType> &L,
			const vector<bool> &used_bp_roots);


	inline void initialize(
				vector<ShortIndex> &short_index,
				vector< vector<smalli> > &dist_matrix,
				vector<idi> &active_queue,
				idi &end_active_queue,
				vector<idi> &once_candidated_queue,
				idi &end_once_candidated_queue,
				vector<bool> &once_candidated,
				idi b_id,
				idi roots_start,
				inti roots_size,
				vector<IndexType> &L,
				const vector<bool> &used_bp_roots);
	inline void push_labels(
				idi v_head,
				idi roots_start,
				const Graph &G,
				const vector<IndexType> &L,
				vector<ShortIndex> &short_index,
				vector<idi> &candidate_queue,
				idi &end_candidate_queue,
				vector<bool> &got_candidates,
				vector<idi> &once_candidated_queue,
				idi &end_once_candidated_queue,
				vector<bool> &once_candidated,
				const vector<bool> &used_bp_roots,
				smalli iter);
	inline bool distance_query(
				idi cand_root_id,
				idi v_id,
				idi roots_start,
				const vector<IndexType> &L,
				const vector< vector<smalli> > &dist_matrix,
				smalli iter);
	inline void insert_label_only(
				idi cand_root_id,
				idi v_id,
				idi roots_start,
				inti roots_size,
				vector<IndexType> &L,
				vector< vector<smalli> > &dist_matrix,
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
				vector< vector<smalli> > &dist_matrix);

	// Test only
//	uint64_t normal_hit_count = 0;
	uint64_t bp_hit_count = 0;
//	uint64_t total_check_count = 0;
	uint64_t normal_check_count = 0;
	uint64_t total_candidates_num = 0;
	uint64_t set_candidates_num = 0;
	double initializing_time = 0;
	double candidating_time = 0;
	double adding_time = 0;
	double distance_query_time = 0;
//	double init_index_time = 0;
//	double init_dist_matrix_time = 0;
//	double init_start_reset_time = 0;
//	double init_indicators_time = 0;
//	L2CacheMissRate cache_miss;
//	TotalInstructsExe candidating_ins_count;
//	TotalInstructsExe adding_ins_count;
//	TotalInstructsExe bp_labeling_ins_count;
//	TotalInstructsExe bp_checking_ins_count;
//	TotalInstructsExe dist_query_ins_count;
	// End test



public:
	VertexCentricPLL() = default;
	VertexCentricPLL(const Graph &G);

	weighti query(
			idi u,
			idi v);

	void print();
	void switch_labels_to_old_id(
					const vector<idi> &rank2id,
					const vector<idi> &rank);

}; // class VertexCentricPLL

VertexCentricPLL::VertexCentricPLL(const Graph &G)
{
	construct(G);
}

inline void VertexCentricPLL::bit_parallel_labeling(
			const Graph &G,
			vector<IndexType> &L,
			vector<bool> &used_bp_roots)
{
	idi num_v = G.get_num_v();
	idi num_e = G.get_num_e();

	std::vector<smalli> tmp_d(num_v); // distances from the root to every v
	std::vector<std::pair<uint64_t, uint64_t> > tmp_s(num_v); // first is S_r^{-1}, second is S_r^{0}
	std::vector<idi> que(num_v); // active queue
	std::vector<std::pair<idi, idi> > sibling_es(num_e); // siblings, their distances to the root are equal (have difference of 0)
	std::vector<std::pair<idi, idi> > child_es(num_e); // child and father, their distances to the root have difference of 1.

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
		used_bp_roots[r] = true;

		fill(tmp_d.begin(), tmp_d.end(), SMALLI_MAX);
		fill(tmp_s.begin(), tmp_s.end(), std::make_pair(0, 0));

		idi que_t0 = 0, que_t1 = 0, que_h = 0;
		que[que_h++] = r;
		tmp_d[r] = 0;
		que_t1 = que_h;

		int ns = 0; // number of selected neighbor, default 64
		// the edge of one vertex in G is ordered decreasingly to rank, lower rank first, so here need to traverse edges backward
		idi i_bound = G.vertices[r] - 1;
		idi i_start = i_bound + G.out_degrees[r];
		for (idi i = i_start; i > i_bound; --i) {
			idi v = G.out_edges[i];
			if (!used_bp_roots[v]) {
				used_bp_roots[v] = true;
				// Algo3:line4: for every v in S_r, (dist[v], S_r^{-1}[v], S_r^{0}[v]) <- (1, {v}, empty_set)
				que[que_h++] = v;
				tmp_d[v] = 1;
				tmp_s[v].first = 1ULL << ns;
				if (++ns == 64) break;
			}
		}

		for (smalli d = 0; que_t0 < que_h; ++d) {
			idi num_sibling_es = 0, num_child_es = 0;

			for (idi que_i = que_t0; que_i < que_t1; ++que_i) {
				idi v = que[que_i];
				idi i_start = G.vertices[v];
				idi i_bound = i_start + G.out_degrees[v];
				for (idi i = i_start; i < i_bound; ++i) {
					idi tv = G.out_edges[i];
					smalli td = d + 1;

					if (d > tmp_d[tv]) {
						;
					}
					else if (d == tmp_d[tv]) {
						if (v < tv) { // ??? Why need v < tv !!! Because it's a undirected graph.
							sibling_es[num_sibling_es].first  = v;
							sibling_es[num_sibling_es].second = tv;
							++num_sibling_es;
						}
					} else { // d < tmp_d[tv]
						if (tmp_d[tv] == SMALLI_MAX) {
							que[que_h++] = tv;
							tmp_d[tv] = td;
						}
						child_es[num_child_es].first  = v;
						child_es[num_child_es].second = tv;
						++num_child_es;
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
				tmp_s[c].first  |= tmp_s[v].first;
				tmp_s[c].second |= tmp_s[v].second;
			}

			que_t0 = que_t1;
			que_t1 = que_h;
		}

		for (idi v = 0; v < num_v; ++v) {
			L[v].bp_dist[i_bpspt] = tmp_d[v];
			L[v].bp_sets[i_bpspt][0] = tmp_s[v].first; // S_r^{-1}
			L[v].bp_sets[i_bpspt][1] = tmp_s[v].second & ~tmp_s[v].first; // Only need those r's neighbors who are not already in S_r^{-1}
		}
	}

}

// Function bit parallel checking:
// return false if shortest distance exits in bp labels, return true if bp labels cannot cover the distance
inline bool VertexCentricPLL::bit_parallel_checking(
			idi v_id,
			idi w_id,
			const vector<IndexType> &L,
			smalli iter)
{
	// Bit Parallel Checking: if label_real_id to v_tail has shorter distance already
	const IndexType &Lv = L[v_id];
	const IndexType &Lw = L[w_id];

	_mm_prefetch(&Lv.bp_dist[0], _MM_HINT_T0);
	_mm_prefetch(&Lv.bp_sets[0][0], _MM_HINT_T0);
	_mm_prefetch(&Lw.bp_dist[0], _MM_HINT_T0);
	_mm_prefetch(&Lw.bp_sets[0][0], _MM_HINT_T0);
	for (inti i = 0; i < BITPARALLEL_SIZE; ++i) {
		inti td = Lv.bp_dist[i] + Lw.bp_dist[i];
		if (td - 2 <= iter) {
			td +=
				(Lv.bp_sets[i][0] & Lw.bp_sets[i][0]) ? -2 :
				((Lv.bp_sets[i][0] & Lw.bp_sets[i][1]) |
				 (Lv.bp_sets[i][1] & Lw.bp_sets[i][0]))
				? -1 : 0;
			if (td <= iter) {
				++bp_hit_count;
				return false;
			}
		}
	}
	return true;
}


// Function for initializing at the begin of a batch
// For a batch, initialize the temporary labels and real labels of roots;
// traverse roots' labels to initialize distance buffer;
// unset flag arrays is_active and got_labels
inline void VertexCentricPLL::initialize(
			vector<ShortIndex> &short_index,
			vector< vector<smalli> > &dist_matrix,
			vector<idi> &active_queue,
			idi &end_active_queue,
			vector<idi> &once_candidated_queue,
			idi &end_once_candidated_queue,
			vector<bool> &once_candidated,
			idi b_id,
			idi roots_start,
			inti roots_size,
			vector<IndexType> &L,
			const vector<bool> &used_bp_roots)
{
	idi roots_bound = roots_start + roots_size;
//	init_start_reset_time -= WallTimer::get_time_mark();
	// TODO: parallel enqueue
	{
		//active_queue
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
		for (idi v_i = 0; v_i < end_once_candidated_queue; ++v_i) {
			idi v = once_candidated_queue[v_i];
			short_index[v].indicator.reset();
			once_candidated[v] = false;
		}
		end_once_candidated_queue = 0;
		for (idi v = roots_start; v < roots_bound; ++v) {
			if (!used_bp_roots[v]) {
				short_index[v].indicator.set(v - roots_start);
				short_index[v].indicator.set(BATCH_SIZE); // v got labels
			}
		}
//		init_indicators_time += WallTimer::get_time_mark();
	}
//
	// Real Index
	{
//		IndexType &Lr = nullptr;
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
//	init_index_time += WallTimer::get_time_mark();
//	init_dist_matrix_time -= WallTimer::get_time_mark();
	// Dist_matrix
	{
//		IndexType &Lr;
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
//	init_dist_matrix_time += WallTimer::get_time_mark();
}

// Function that pushes v_head's labels to v_head's every neighbor
inline void VertexCentricPLL::push_labels(
				idi v_head,
				idi roots_start,
				const Graph &G,
				const vector<IndexType> &L,
				vector<ShortIndex> &short_index,
				vector<idi> &candidate_queue,
				idi &end_candidate_queue,
				vector<bool> &got_candidates,
				vector<idi> &once_candidated_queue,
				idi &end_once_candidated_queue,
				vector<bool> &once_candidated,
				const vector<bool> &used_bp_roots,
				smalli iter)
{
	const IndexType &Lv = L[v_head];
	// These 2 index are used for traversing v_head's last inserted labels
	idi l_i_start = Lv.distances.rbegin() -> start_index;
	idi l_i_bound = l_i_start + Lv.distances.rbegin() -> size;
	// Traverse v_head's every neighbor v_tail
	idi e_i_start = G.vertices[v_head];
	idi e_i_bound = e_i_start + G.out_degrees[v_head];
	for (idi e_i = e_i_start; e_i < e_i_bound; ++e_i) {
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
		if (v_tail <= Lv.vertices[l_i_start] + roots_start) { // v_tail has higher rank than any v_head's labels
			return;
		}
		const IndexType &L_tail = L[v_tail];
		_mm_prefetch(&L_tail.bp_dist[0], _MM_HINT_T0);
		_mm_prefetch(&L_tail.bp_sets[0][0], _MM_HINT_T0);
		// Traverse v_head's last inserted labels
		for (idi l_i = l_i_start; l_i < l_i_bound; ++l_i) {
			idi label_root_id = Lv.vertices[l_i];
			idi label_real_id = label_root_id + roots_start;
			if (v_tail <= label_real_id) {
				// v_tail has higher rank than all remaining labels
				break;
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

			// Bit Parallel Checking: if label_real_id to v_tail has shorter distance already
			//			++total_check_count;
			const IndexType &L_label = L[label_real_id];

			_mm_prefetch(&L_label.bp_dist[0], _MM_HINT_T0);
			_mm_prefetch(&L_label.bp_sets[0][0], _MM_HINT_T0);
//			bp_checking_ins_count.measure_start();
			bool no_need_add = false;
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
						++bp_hit_count;
						break;
					}
				}
			}
			if (no_need_add) {
//				bp_checking_ins_count.measure_stop();
				continue;
			}
//			bp_checking_ins_count.measure_stop();

			// Record vertex label_root_id as v_tail's candidates label
			SI_v_tail.candidates.set(label_root_id);

			// Add into candidate_queue
			if (!got_candidates[v_tail]) {
				// If v_tail is not in candidate_queue, add it in (prevent duplicate)
				got_candidates[v_tail] = true;
				candidate_queue[end_candidate_queue++] = v_tail;
			}
		}
	}
}

// Function for distance query;
// traverse vertex v_id's labels;
// return false if shorter distance exists already, return true if the cand_root_id can be added into v_id's label.
inline bool VertexCentricPLL::distance_query(
			idi cand_root_id,
			idi v_id,
			idi roots_start,
			const vector<IndexType> &L,
			const vector< vector<smalli> > &dist_matrix,
			smalli iter)
{
//	++total_check_count;
	++normal_check_count;
	distance_query_time -= WallTimer::get_time_mark();
//	dist_query_ins_count.measure_start();

	idi cand_real_id = cand_root_id + roots_start;
	const IndexType &Lv = L[v_id];

//	// Bit Parallel Checking: if label_real_id to v_tail has shorter distance already
//	++total_check_count;
//	const IndexType &L_tail = L[cand_real_id];
//
//	_mm_prefetch(&Lv.bp_dist[0], _MM_HINT_T0);
//	_mm_prefetch(&Lv.bp_sets[0][0], _MM_HINT_T0);
//	for (inti i = 0; i < BITPARALLEL_SIZE; ++i) {
//		inti td = Lv.bp_dist[i] + L_tail.bp_dist[i];
//		if (td - 2 <= iter) {
//			td +=
//				(Lv.bp_sets[i][0] & L_tail.bp_sets[i][0]) ? -2 :
//				((Lv.bp_sets[i][0] & L_tail.bp_sets[i][1]) |
//				 (Lv.bp_sets[i][1] & L_tail.bp_sets[i][0]))
//				? -1 : 0;
//			if (td <= iter) {
//				++bp_hit_count;
//				return false;
//			}
//		}
//	}

	// Traverse v_id's all existing labels
	inti b_i_bound = Lv.batches.size();
	_mm_prefetch(&Lv.batches[0], _MM_HINT_T0);
	_mm_prefetch(&Lv.distances[0], _MM_HINT_T0);
	_mm_prefetch(&Lv.vertices[0], _MM_HINT_T0);
	//_mm_prefetch(&dist_matrix[cand_root_id][0], _MM_HINT_T0);
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
			_mm_prefetch(&dist_matrix[cand_root_id][0], _MM_HINT_T0);
			for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
				idi v = Lv.vertices[v_i] + id_offset; // v is a label hub of v_id
				if (v >= cand_real_id) {
					// Vertex cand_real_id cannot have labels whose ranks are lower than it,
					// in which case dist_matrix[cand_root_id][v] does not exit.
					continue;
				}
				inti d_tmp = dist + dist_matrix[cand_root_id][v];
				if (d_tmp <= iter) {
					distance_query_time += WallTimer::get_time_mark();
//					dist_query_ins_count.measure_stop();
					return false;
				}
			}
		}
	}
	distance_query_time += WallTimer::get_time_mark();
//	dist_query_ins_count.measure_stop();
	return true;
}

// Function inserts candidate cand_root_id into vertex v_id's labels;
// update the distance buffer dist_matrix;
// but it only update the v_id's labels' vertices array;
inline void VertexCentricPLL::insert_label_only(
				idi cand_root_id,
				idi v_id,
				idi roots_start,
				inti roots_size,
				vector<IndexType> &L,
				vector< vector<smalli> > &dist_matrix,
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
inline void VertexCentricPLL::update_label_indices(
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
inline void VertexCentricPLL::reset_at_end(
				idi roots_start,
				inti roots_size,
				vector<IndexType> &L,
				vector< vector<smalli> > &dist_matrix)
{
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

inline void VertexCentricPLL::batch_process(
						const Graph &G,
						idi b_id,
						idi roots_start, // start id of roots
						inti roots_size, // how many roots in the batch
						vector<IndexType> &L,
						const vector<bool> &used_bp_roots)
{

	initializing_time -= WallTimer::get_time_mark();
	static const idi num_v = G.get_num_v();
	static vector<idi> active_queue(num_v);
	static idi end_active_queue = 0;
	static vector<idi> candidate_queue(num_v);
	static idi end_candidate_queue = 0;
	static vector<ShortIndex> short_index(num_v);
	static vector< vector<smalli> > dist_matrix(roots_size, vector<smalli>(num_v, SMALLI_MAX));
	static vector<bool> got_candidates(num_v, false); // got_candidates[v] is true means vertex v is in the queue candidate_queue
	static vector<bool> is_active(num_v, false);// is_active[v] is true means vertex v is in the active queue.

	static vector<idi> once_candidated_queue(num_v); // if short_index[v].indicator.any() is true, v is in the queue.
	static idi end_once_candidated_queue = 0;
	static vector<bool> once_candidated(num_v, false);

	// At the beginning of a batch, initialize the labels L and distance buffer dist_matrix;
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
	initializing_time += WallTimer::get_time_mark();


	while (0 != end_active_queue) {
		candidating_time -= WallTimer::get_time_mark();
//		candidating_ins_count.measure_start();
		++iter;
		// Traverse active vertices to push their labels as candidates
		for (idi i_queue = 0; i_queue < end_active_queue; ++i_queue) {
			idi v_head = active_queue[i_queue];
			is_active[v_head] = false; // reset is_active

			push_labels(
					v_head,
					roots_start,
					G,
					L,
					short_index,
					candidate_queue,
					end_candidate_queue,
					got_candidates,
					once_candidated_queue,
					end_once_candidated_queue,
					once_candidated,
					used_bp_roots,
					iter);
		}
		end_active_queue = 0; // Set the active_queue empty
//		candidating_ins_count.measure_stop();
		candidating_time += WallTimer::get_time_mark();
		adding_time -= WallTimer::get_time_mark();
//		adding_ins_count.measure_start();

		// Traverse vertices in the candidate_queue to insert labels
		for (idi i_queue = 0; i_queue < end_candidate_queue; ++i_queue) {
			idi v_id = candidate_queue[i_queue];
			inti inserted_count = 0; //recording number of v_id's truly inserted candidates
			got_candidates[v_id] = false; // reset got_candidates
			// Traverse v_id's all candidates
			total_candidates_num += roots_size;
			for (inti cand_root_id = 0; cand_root_id < roots_size; ++cand_root_id) {
				if (!short_index[v_id].candidates[cand_root_id]) {
					// Root cand_root_id is not vertex v_id's candidate
					continue;
				}
				++set_candidates_num;
				short_index[v_id].candidates.reset(cand_root_id);
				// Only insert cand_root_id into v_id's label if its distance to v_id is shorter than existing distance
				if ( distance_query(
								cand_root_id,
								v_id,
								roots_start,
								L,
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
							L,
							dist_matrix,
							iter);
				}
			}
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
		end_candidate_queue = 0; // Set the candidate_queue empty
//		adding_ins_count.measure_stop();
		adding_time += WallTimer::get_time_mark();
	}

	// Reset the dist_matrix
	initializing_time -= WallTimer::get_time_mark();
//	init_dist_matrix_time -= WallTimer::get_time_mark();
	reset_at_end(
			roots_start,
			roots_size,
			L,
			dist_matrix);
//	init_dist_matrix_time += WallTimer::get_time_mark();
	initializing_time += WallTimer::get_time_mark();


//	double total_time = time_can + time_add;
//	printf("Candidating time: %f (%f%%)\n", time_can, time_can / total_time * 100);
//	printf("Adding time: %f (%f%%)\n", time_add, time_add / total_time * 100);
}



void VertexCentricPLL::construct(const Graph &G)
{
	idi num_v = G.get_num_v();
	L.resize(num_v);
	idi remainer = num_v % BATCH_SIZE;
	idi b_i_bound = num_v / BATCH_SIZE;
	vector<bool> used_bp_roots(num_v, false);
//	cache_miss.measure_start();
	double time_labeling = -WallTimer::get_time_mark();

	double bp_labeling_time = -WallTimer::get_time_mark();
//	bp_labeling_ins_count.measure_start();
	bit_parallel_labeling(
				G,
				L,
				used_bp_roots);
//	bp_labeling_ins_count.measure_stop();
	bp_labeling_time += WallTimer::get_time_mark();

	for (idi b_i = 0; b_i < b_i_bound; ++b_i) {
//		printf("b_i: %u\n", b_i);//test
		batch_process(
				G,
				b_i,
				b_i * BATCH_SIZE,
				BATCH_SIZE,
				L,
				used_bp_roots);
//		batch_process(
//				G,
//				b_i,
//				b_i * BATCH_SIZE,
//				BATCH_SIZE,
//				L);
	}
	if (remainer != 0) {
//		printf("b_i: %u\n", b_i_bound);//test
		batch_process(
				G,
				b_i_bound,
				b_i_bound * BATCH_SIZE,
				remainer,
				L,
				used_bp_roots);
//		batch_process(
//				G,
//				b_i_bound,
//				b_i_bound * BATCH_SIZE,
//				remainer,
//				L);
	}
	time_labeling += WallTimer::get_time_mark();
//	cache_miss.measure_stop();

	// Test
	setlocale(LC_NUMERIC, "");
	printf("BATCH_SIZE: %u\n", BATCH_SIZE);
	printf("BP_Size: %u\n", BITPARALLEL_SIZE);
	printf("BP_labeling: %f %.2f%%\n", bp_labeling_time, bp_labeling_time / time_labeling * 100);
	printf("Initializing: %f %.2f%%\n", initializing_time, initializing_time / time_labeling * 100);
//		printf("\tinit_start_reset_time: %f (%f%%)\n", init_start_reset_time, init_start_reset_time / initializing_time * 100);
//		printf("\tinit_index_time: %f (%f%%)\n", init_index_time, init_index_time / initializing_time * 100);
//			printf("\t\tinit_indicators_time: %f (%f%%)\n", init_indicators_time, init_indicators_time / init_index_time * 100);
//		printf("\tinit_dist_matrix_time: %f (%f%%)\n", init_dist_matrix_time, init_dist_matrix_time / initializing_time * 100);
	printf("Candidating: %f %.2f%%\n", candidating_time, candidating_time / time_labeling * 100);
	printf("Adding: %f %.2f%%\n", adding_time, adding_time / time_labeling * 100);
		printf("distance_query_time: %f %.2f%%\n", distance_query_time, distance_query_time / time_labeling * 100);
		uint64_t total_check_count = bp_hit_count + normal_check_count;
		printf("total_check_count: %'llu\n", total_check_count);
		printf("bp_hit_count: %'llu %.2f%%\n",
						bp_hit_count,
						bp_hit_count * 100.0 / total_check_count);
		printf("normal_check_count: %'llu %.2f%%\n", normal_check_count, normal_check_count * 100.0 / total_check_count);
		printf("total_candidates_num: %'llu set_candidates_num: %'llu %.2f%%\n",
							total_candidates_num,
							set_candidates_num,
							set_candidates_num * 100.0 / total_candidates_num);
//		printf("\tnormal_hit_count (to total_check, to normal_check): %llu (%f%%, %f%%)\n",
//						normal_hit_count,
//						normal_hit_count * 100.0 / total_check_count,
//						normal_hit_count * 100.0 / (total_check_count - bp_hit_count));
//	cache_miss.print();
//	printf("Candidating: "); candidating_ins_count.print();
//	printf("Adding: "); adding_ins_count.print();
//	printf("BP_Labeling: "); bp_labeling_ins_count.print();
//	printf("BP_Checking: "); bp_checking_ins_count.print();
//	printf("distance_query: "); dist_query_ins_count.print();
	printf("Labeling_time: %.2f\n", time_labeling);
	// End test
}

void VertexCentricPLL::switch_labels_to_old_id(
								const vector<idi> &rank2id,
								const vector<idi> &rank)
{
	idi label_sum = 0;
	idi test_label_sum = 0;

//	idi num_v = rank2id.size();
	idi num_v = rank.size();
	vector< vector< pair<idi, weighti> > > new_L(num_v);
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
					new_L[new_v].push_back(make_pair(tail, dist));
					++test_label_sum;
				}
			}
		}
	}
	printf("Label sum: %u (%u), mean: %f\n", label_sum, test_label_sum, label_sum * 1.0 / num_v);

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

	// Try query
	idi u;
	idi v;
	while (std::cin >> u >> v) {
		weighti dist = WEIGHTI_MAX;
		// Bit Parallel Check
		const IndexType &idx_u = L[rank[u]];
		const IndexType &idx_v = L[rank[v]];

		for (inti i = 0; i < BITPARALLEL_SIZE; ++i) {
			int td = idx_v.bp_dist[i] + idx_u.bp_dist[i];
			if (td - 2 <= dist) {
				td +=
					(idx_v.bp_sets[i][0] & idx_u.bp_sets[i][0]) ? -2 :
					((idx_v.bp_sets[i][0] & idx_u.bp_sets[i][1])
							| (idx_v.bp_sets[i][1] & idx_u.bp_sets[i][0]))
							? -1 : 0;
				if (td < dist) {
					dist = td;
				}
			}
		}

		// Normal Index Check
		const auto &Lu = new_L[u];
		const auto &Lv = new_L[v];
//		unsorted_map<idi, weighti> markers;
		map<idi, weighti> markers;
		for (idi i = 0; i < Lu.size(); ++i) {
			markers[Lu[i].first] = Lu[i].second;
		}
		for (idi i = 0; i < Lv.size(); ++i) {
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

}
#endif /* INCLUDES_PADO_H_ */
