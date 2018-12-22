/*
 * pado.h
 *
 *  Created on: Dec 18, 2018
 *      Author: Zhen Peng
 */

#ifndef INCLUDES_PADO_WEIGHTED_H_
#define INCLUDES_PADO_WEIGHTED_H_

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

using std::vector;
using std::unordered_map;
using std::map;
using std::bitset;
using std::stable_sort;
using std::min;
using std::fill;

namespace PADO {

//const inti BATCH_SIZE = 1024; // The size for regular batch and bit array.
	idi BATCH_SIZE; // Here is for the whole graph, so make it non-const
//const inti BITPARALLEL_SIZE = 50; // Weighted graphs cannot use Bit Parallel technique



//// Batch based processing, 09/11/2018
class WeightedVertexCentricPLL {
private:
	// Structure for the type of label
	struct IndexType {
		vector<idi> vertices; // Vertices in the label, preresented as temperory ID
		vector<weighti> distances;
	} __attribute__((aligned(64)));
//	struct IndexType {
//		struct Batch {
//			idi batch_id; // Batch ID
//			idi start_index; // Index to the array distances where the batch starts
//			inti size; // Number of distances element in this batch
//
//			Batch(idi batch_id_, idi start_index_, inti size_):
//						batch_id(batch_id_), start_index(start_index_), size(size_)
//			{
//				;
//			}
//		};
//
//		struct DistanceIndexType {
//			idi start_index; // Index to the array vertices where the same-ditance vertices start
//			inti size; // Number of the same-distance vertices
//			weighti dist; // The real distance
//
//			DistanceIndexType(idi start_index_, inti size_, weighti dist_):
//						start_index(start_index_), size(size_), dist(dist_)
//			{
//				;
//			}
//		};
//
//		// No Bit Parallel Labels for weighted graphs
//
//		vector<Batch> batches; // Batch info
//		vector<DistanceIndexType> distances; // Distance info
//		vector<idi> vertices; // Vertices in the label, preresented as temperory ID
//	} __attribute__((aligned(64)));

	// Structure for the type of temporary label
	struct ShortIndex {
		// Use a queue to store candidates
		//vector<idi> candidates_que;
		//idi end_candidates_que = 0;
		vector<weighti> candidates_dists; // record the distances to candidates. If candidates_dists[c] = INF, then c is NOT a candidate
			// The candidates_dists is also used for distance query.

		// Use a queue to store last inserted distances
		//vector< pair<idi, weighti> > lasts_queue;
		//idi end_lasts_queue = 0;

		ShortIndex() = default;
		explicit ShortIndex(idi num_roots) {
			//candidates_que.resize(num_roots);
			candidates_dists.resize(num_roots, WEIGHTI_MAX);
			//lasts_queue.resize(num_roots);
		}
//		// I use BATCH_SIZE + 1 bit for indicator bit array.
//		// The v.indicator[BATCH_SIZE] is set if in current batch v has got any new labels already.
//		// In this way, when do initialization, only initialize those short_index[v] whose indicator[BATCH_SIZE] is set.
//		bitset<BATCH_SIZE + 1> indicator; // Global indicator, indicator[r] (0 <= r < BATCH_SIZE) is set means root r once selected as candidate already
////		bitset<BATCH_SIZE> candidates; // Candidates one iteration, candidates[r] is set means root r is candidate in this iteration
//
//		// Use a queue to store candidates
//		vector<inti> candidates_que = vector<inti>(BATCH_SIZE);
//		inti end_candidates_que = 0;
//		vector<bool> is_candidate = vector<bool>(BATCH_SIZE, false);

	} __attribute__((aligned(64)));

	vector<IndexType> L;

	void construct(const WeightedGraph &G);
	inline void batch_process(
			const WeightedGraph &G,
			//idi b_id,
			idi root_start,
			inti roots_size,
			vector<IndexType> &L);
	inline void initialize(
			vector<ShortIndex> &short_index,
			vector< vector<weighti> > &dists_table,
			vector< vector<weighti> > &labels_table,
			vector<idi> &active_queue,
			idi &end_active_queue,
			//vector<idi> &once_candidated_queue,
			//idi &end_once_candidated_queue,
			//vector<bool> &once_candidated,
			//idi b_id,
			idi roots_start,
			inti roots_size,
			vector<IndexType> &L);
	inline void push_labels(
			idi v_head,
			idi roots_start,
			const WeightedGraph &G,
			//const vector<IndexType> &L,
			vector< vector<weighti> > &dists_table,
			const vector< vector<weighti> > &labels_table,
			vector<ShortIndex> &short_index,
			vector<idi> &has_cand_queue,
			idi &end_has_cand_queue,
			vector<bool> &had_candidates);
	//vector<idi> &once_candidated_queue,
	//idi &end_once_candidated_queue,
	//vector<bool> &once_candidated)
	inline weighti distance_query(
			idi v_id,
			idi cand_root_id,
			const vector< vector<weighti> > &labels_table,
			const vector<ShortIndex> &short_index,
			idi roots_start,
			weighti tmp_dist_v_c);
	inline void update_index(
			vector<IndexType> &L,
			vector< vector<weighti> > &labels_table,
			idi roots_start,
			inti roots_size,
			idi num_v);
//	inline void insert_label_only(
//			idi cand_root_id,
//			idi v_id,
//			idi roots_start,
//			inti roots_size,
//			vector<IndexType> &L,
//			vector< vector<weighti> > &dist_matrix,
//			weighti iter);
//	inline void update_label_indices(
//			idi v_id,
//			idi inserted_count,
//			vector<IndexType> &L,
//			vector<ShortIndex> &short_index,
//			idi b_id,
//			weighti iter);
//	inline void reset_at_end(
//			idi roots_start,
//			inti roots_size,
//			vector<IndexType> &L,
//			vector< vector<weighti> > &dist_matrix);

	// Test only
//	uint64_t normal_hit_count = 0;
	uint64_t bp_hit_count = 0;
//	uint64_t total_check_count = 0;
	uint64_t normal_check_count = 0;
//	uint64_t total_candidates_num = 0;
//	uint64_t set_candidates_num = 0;
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
	WeightedVertexCentricPLL() = default;
	WeightedVertexCentricPLL(const WeightedGraph &G);

	weighti query(
			idi u,
			idi v);

	void print();
	void switch_labels_to_old_id(
					const vector<idi> &rank2id,
					const vector<idi> &rank);

}; // class WeightedVertexCentricPLL

WeightedVertexCentricPLL::WeightedVertexCentricPLL(const WeightedGraph &G)
{
	construct(G);
}

// Function for initializing at the begin of a batch
// For a batch, initialize the temporary labels and real labels of roots;
// traverse roots' labels to initialize distance buffer;
// unset flag arrays is_active and got_labels
inline void WeightedVertexCentricPLL::initialize(
			vector<ShortIndex> &short_index,
			//vector< vector<weighti> > &dist_matrix,
			vector< vector<weighti> > &dists_table,
			vector< vector<weighti> > &labels_table,
			vector<idi> &active_queue,
			idi &end_active_queue,
			//vector<idi> &once_candidated_queue,
			//idi &end_once_candidated_queue,
			//vector<bool> &once_candidated,
			//idi b_id,
			idi roots_start,
			inti roots_size,
			vector<IndexType> &L)
{
	idi roots_bound = roots_start + roots_size;
//	init_start_reset_time -= WallTimer::get_time_mark();
	// TODO: parallel enqueue
	// Active queue
	{
		for (idi r_real_id = roots_start; r_real_id < roots_bound; ++r_real_id) {
			active_queue[end_active_queue++] = r_real_id;
		}
	}
//	init_start_reset_time += WallTimer::get_time_mark();
//	init_index_time -= WallTimer::get_time_mark();
	// Short_index
	{
		// Every root puts itself into lasts_queue of its short_index
//		for (idi r_real_id = roots_start; r_real_id < roots_bound; ++r_real_id) {
//			short_index[r_real_id].lasts_queue[short_index[r_real_id].end_lasts_queue++] = make_pair(r_real_id, 0);
//		}
	//	init_indicators_time -= WallTimer::get_time_mark();
	//	for (idi v_i = 0; v_i < end_once_candidated_queue; ++v_i) {
	//		idi v = once_candidated_queue[v_i];
	//		short_index[v].indicator.reset();
	//		once_candidated[v] = false;
	//	}
	//	end_once_candidated_queue = 0;
	//	for (idi v = roots_start; v < roots_bound; ++v) {
	//		short_index[v].indicator.set(v - roots_start);
	//		short_index[v].indicator.set(BATCH_SIZE); // v got labels
	//	}
	//	init_indicators_time += WallTimer::get_time_mark();
	}
//
	// Real Index
	//{
//	//	IndexType &Lr = nullptr;
	//	for (idi r_id = 0; r_id < roots_size; ++r_id) {
	//		if (used_bp_roots[r_id + roots_start]) {
	//			continue;
	//		}
	//		IndexType &Lr = L[r_id + roots_start];
	//		Lr.batches.push_back(IndexType::Batch(
	//											b_id, // Batch ID
	//											Lr.distances.size(), // start_index
	//											1)); // size
	//		Lr.distances.push_back(IndexType::DistanceIndexType(
	//											Lr.vertices.size(), // start_index
	//											1, // size
	//											0)); // dist
	//		Lr.vertices.push_back(r_id);
	//	}
	//}
//	init_index_time += WallTimer::get_time_mark();
//	init_dist_matrix_time -= WallTimer::get_time_mark();
	// Distance Tables
	{
		// Traverse all roots
		for (idi r_id = 0; r_id < roots_size; ++r_id) {
			idi r_real_id = r_id + roots_start;
			const IndexType &Lr = L[r_real_id];
			idi l_i_bound = Lr.vertices.size();
			// Traverse r_real_id's all labels to initial dists_table
			_mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
			_mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
			for (idi l_i = 0; l_i < l_i_bound; ++l_i) {
				dists_table[Lr.vertices[l_i] - roots_start][r_real_id] = Lr.distances[l_i];
			}

			dists_table[r_id][r_real_id] = 0;
			labels_table[r_real_id][r_id] = 0;
		}
		//inti b_i_bound;
		//idi id_offset;
		//idi dist_start_index;
		//idi dist_bound_index;
		//idi v_start_index;
		//idi v_bound_index;
		//weighti dist;
		//for (idi r_id = 0; r_id < roots_size; ++r_id) {
		//	if (used_bp_roots[r_id + roots_start]) {
		//		continue;
		//	}
		//	IndexType &Lr = L[r_id + roots_start];
		//	b_i_bound = Lr.batches.size();
		//	_mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
		//	_mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
		//	_mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
		//	for (inti b_i = 0; b_i < b_i_bound; ++b_i) {
		//		id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
		//		dist_start_index = Lr.batches[b_i].start_index;
		//		dist_bound_index = dist_start_index + Lr.batches[b_i].size;
		//		// Traverse dist_matrix
		//		for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
		//			v_start_index = Lr.distances[dist_i].start_index;
		//			v_bound_index = v_start_index + Lr.distances[dist_i].size;
		//			dist = Lr.distances[dist_i].dist;
		//			for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
		//				dist_matrix[r_id][Lr.vertices[v_i] + id_offset] = dist;
		//			}
		//		}
		//	}
		//}
	}
//	init_dist_matrix_time += WallTimer::get_time_mark();
}

// Function that pushes v_head's labels to v_head's every neighbor
inline void WeightedVertexCentricPLL::push_labels(
				idi v_head,
				idi roots_start,
				const WeightedGraph &G,
				//const vector<IndexType> &L,
				vector< vector<weighti> > &dists_table,
				const vector< vector<weighti> > &labels_table,
				vector<ShortIndex> &short_index,
				vector<idi> &has_cand_queue,
				idi &end_has_cand_queue,
				vector<bool> &had_candidates)
				//vector<idi> &once_candidated_queue,
				//idi &end_once_candidated_queue,
				//vector<bool> &once_candidated)
{
	// Traverse v_head's every neighbor v_tail
	idi e_i_start = G.vertices[v_head];
	idi e_i_bound = e_i_start + G.out_degrees[v_head];
	for (idi e_i = e_i_start; e_i < e_i_bound; ++e_i) {
		idi v_tail = G.out_edges[e_i];
		ShortIndex &SI_v_tail = short_index[v_tail];
		weighti weight_h_t = G.out_weights[e_i];

		if (v_tail < roots_start) { // v_tail has higher rank than any roots, then no roots can push new labels to it.
			return;
		}
//		if (v_tail <= Lv.vertices[l_i_start] + roots_start) { // v_tail has higher rank than any v_head's labels
//			return;
//		}
		// Traverse v_head's last inserted labels
		// TODO: use a queue to store v_head's last inserted labels, maybe in the ShortIndex structure
		for (idi r = roots_start; r < v_tail; ++r) {
			// r is a potential candidate for v_tail, so r should has higher rank than v_tail
			idi r_root_id = r - roots_start;
			weighti dist_r_h = labels_table[v_head][r_root_id];
			if (WEIGHTI_MAX == dist_r_h) {
				continue;
			}
			weighti tmp_dist_r_t = dist_r_h + weight_h_t;
			if (tmp_dist_r_t < dists_table[r_root_id][v_tail] && tmp_dist_r_t < SI_v_tail.candidates_dists[r_root_id]) {
				// Makr r as a candidate of v_tail
				// TODO: also use a queue to store candidate IDs, in ShortIndex structure
				SI_v_tail.candidates_dists[r_root_id] = tmp_dist_r_t;
				dists_table[r_root_id][v_tail] = tmp_dist_r_t;
				if (!had_candidates[v_tail]) {
					// Add v_tail into has_cand_queue
					had_candidates[v_tail] = true;
					has_cand_queue[end_has_cand_queue++] = v_tail;
				}
			}
		}
	}
}

//Function: to see if v_id and cand_root_id can have other path already cover the candidate distance
// If there was other path, return the shortest distance. If there was not, return INF
inline weighti WeightedVertexCentricPLL::distance_query(
					idi v_id,
					idi cand_root_id,
					//const vector< vector<weighti> > &dists_table,
					const vector< vector<weighti> > &labels_table,
					const vector<ShortIndex> &short_index,
					idi roots_start,
					//inti roots_size,
					//idi num_v,
					weighti tmp_dist_v_c)
{
	// Traverse other roots
	idi cand_real_id = roots_start + cand_root_id;
	for (idi hop_root_id = 0; hop_root_id < cand_root_id; ++hop_root_id) {
		// Check label distances in this batch
		if (WEIGHTI_MAX == labels_table[v_id][hop_root_id] || WEIGHTI_MAX == labels_table[cand_root_id][hop_root_id]) {
			continue;
		}
		weighti dist_label_v_h = labels_table[v_id][hop_root_id];
		weighti dist_label_c_h = labels_table[cand_root_id][hop_root_id];
		weighti label_v_c = dist_label_v_h + dist_label_c_h;
		if (label_v_c <= tmp_dist_v_c) {
			return label_v_c;
		}

		// Check candidate distances
		if (WEIGHTI_MAX == short_index[v_id].candidates_dists[hop_root_id] 
				|| WEIGHTI_MAX == short_index[cand_real_id].candidates_dists[hop_root_id]) {
			continue;
		}
		weighti dist_cand_v_h = short_index[v_id].candidates_dists[hop_root_id];
		weighti dist_cand_c_h = short_index[cand_real_id].candidates_dists[hop_root_id];
		weighti cand_dist_v_c = dist_cand_v_h + dist_cand_c_h;
		if (cand_dist_v_c <= tmp_dist_v_c) {
			return cand_dist_v_c;
		}

		// Cross check
		weighti label_v_cand_c = dist_label_v_h + dist_cand_c_h;
		if (label_v_cand_c <= tmp_dist_v_c) {
			return label_v_cand_c;
		}
		weighti cand_v_label_c = dist_cand_v_h + dist_label_c_h;
		if (cand_v_label_c <= tmp_dist_v_c) {
			return cand_v_label_c;
		}
	}

	return WEIGHTI_MAX;
}
//// Function for distance query;
//// traverse vertex v_id's labels;
//// return false if shorter distance exists already, return true if the cand_root_id can be added into v_id's label.
//inline bool WeightedVertexCentricPLL::distance_query(
//			idi cand_root_id,
//			idi v_id,
//			idi roots_start,
//			const vector<IndexType> &L,
//			const vector< vector<weighti> > &dist_matrix,
//			weighti iter)
//{
////	++total_check_count;
//	++normal_check_count;
//	distance_query_time -= WallTimer::get_time_mark();
////	dist_query_ins_count.measure_start();
//
//	idi cand_real_id = cand_root_id + roots_start;
//	const IndexType &Lv = L[v_id];
//
//	// Traverse v_id's all existing labels
//	inti b_i_bound = Lv.batches.size();
//	_mm_prefetch(&Lv.batches[0], _MM_HINT_T0);
//	_mm_prefetch(&Lv.distances[0], _MM_HINT_T0);
//	_mm_prefetch(&Lv.vertices[0], _MM_HINT_T0);
//	//_mm_prefetch(&dist_matrix[cand_root_id][0], _MM_HINT_T0);
//	for (inti b_i = 0; b_i < b_i_bound; ++b_i) {
//		idi id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
//		idi dist_start_index = Lv.batches[b_i].start_index;
//		idi dist_bound_index = dist_start_index + Lv.batches[b_i].size;
//		// Traverse dist_matrix
//		for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
//			inti dist = Lv.distances[dist_i].dist;
//			if (dist >= iter) { // In a batch, the labels' distances are increasingly ordered.
//				// If the half path distance is already greater than their targeted distance, jump to next batch
//				break;
//			}
//			idi v_start_index = Lv.distances[dist_i].start_index;
//			idi v_bound_index = v_start_index + Lv.distances[dist_i].size;
//			_mm_prefetch(&dist_matrix[cand_root_id][0], _MM_HINT_T0);
//			for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
//				idi v = Lv.vertices[v_i] + id_offset; // v is a label hub of v_id
//				if (v >= cand_real_id) {
//					// Vertex cand_real_id cannot have labels whose ranks are lower than it,
//					// in which case dist_matrix[cand_root_id][v] does not exit.
//					continue;
//				}
//				inti d_tmp = dist + dist_matrix[cand_root_id][v];
//				if (d_tmp <= iter) {
//					distance_query_time += WallTimer::get_time_mark();
////					dist_query_ins_count.measure_stop();
//					return false;
//				}
//			}
//		}
//	}
//	distance_query_time += WallTimer::get_time_mark();
////	dist_query_ins_count.measure_stop();
//	return true;
//}

inline void WeightedVertexCentricPLL::update_index(
		vector<IndexType> &L,
		vector< vector<weighti> > &labels_table,
		idi roots_start,
		inti roots_size,
		idi num_v)
{
	idi roots_bound = roots_start + roots_size;
	for (idi v = 0; v < num_v; ++v) {
		for (idi r = roots_start; r < v; ++r) {
			idi r_root_id = r - roots_start;
			if (WEIGHTI_MAX != labels_table[v][r_root_id]) {
				L[v].vertices.push_back(r);
				L[v].distances.push_back(labels_table[v][r_root_id]);
			}
		}
	}
	for (idi r = roots_start; r < roots_bound; ++r) {
		L[r].vertices.push_back(r);
		L[r].distances.push_back(0);
	}
}

//// Function inserts candidate cand_root_id into vertex v_id's labels;
//// update the distance buffer dist_matrix;
//// but it only update the v_id's labels' vertices array;
//inline void WeightedVertexCentricPLL::insert_label_only(
//				idi cand_root_id,
//				idi v_id,
//				idi roots_start,
//				inti roots_size,
//				vector<IndexType> &L,
//				vector< vector<weighti> > &dist_matrix,
//				weighti iter)
//{
//	L[v_id].vertices.push_back(cand_root_id);
//	// Update the distance buffer if necessary
//	idi v_root_id = v_id - roots_start;
//	if (v_id >= roots_start && v_root_id < roots_size) {
//		dist_matrix[v_root_id][cand_root_id + roots_start] = iter;
//	}
//}

//// Function updates those index arrays in v_id's label only if v_id has been inserted new labels
//inline void WeightedVertexCentricPLL::update_label_indices(
//				idi v_id,
//				idi inserted_count,
//				vector<IndexType> &L,
//				vector<ShortIndex> &short_index,
//				idi b_id,
//				weighti iter)
//{
//	IndexType &Lv = L[v_id];
//	// indicator[BATCH_SIZE + 1] is true, means v got some labels already in this batch
//	if (short_index[v_id].indicator[BATCH_SIZE]) {
//		// Increase the batches' last element's size because a new distance element need to be added
//		++(Lv.batches.rbegin() -> size);
//	} else {
//		short_index[v_id].indicator.set(BATCH_SIZE);
//		// Insert a new Batch with batch_id, start_index, and size because a new distance element need to be added
//		Lv.batches.push_back(IndexType::Batch(
//									b_id,
//									Lv.distances.size(),
//									1));
//	}
//	// Insert a new distance element with start_index, size, and dist
//	Lv.distances.push_back(IndexType::DistanceIndexType(
//										Lv.vertices.size() - inserted_count,
//										inserted_count,
//										iter));
//}

//// Function to reset dist_matrix the distance buffer to INF
//// Traverse every root's labels to reset its distance buffer elements to INF.
//// In this way to reduce the cost of initialization of the next batch.
//inline void WeightedVertexCentricPLL::reset_at_end(
//				idi roots_start,
//				inti roots_size,
//				vector<IndexType> &L,
//				vector< vector<weighti> > &dist_matrix)
//{
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
//					dist_matrix[r_id][Lr.vertices[v_i] + id_offset] = WEIGHTI_MAX;
//				}
//			}
//		}
//	}
//}

inline void WeightedVertexCentricPLL::batch_process(
						const WeightedGraph &G,
						//idi b_id,
						idi roots_start, // start id of roots
						inti roots_size, // how many roots in the batch
						vector<IndexType> &L)
{
	printf("@630 batch_process\n"); fflush(stdout);//test
	initializing_time -= WallTimer::get_time_mark();
	static const idi num_v = G.get_num_v();
	// Active queue
	static vector<idi> active_queue(num_v);
	static idi end_active_queue = 0;
	static vector<bool> is_active(num_v, false);// is_active[v] is true means vertex v is in the active queue.
	// Queue of vertices having candidates
	static vector<idi> has_cand_queue(num_v);
	static idi end_has_cand_queue = 0;
	static vector<bool> had_candidates(num_v, false); // had_candidates[v] is true means vertex v is in the queue has_cand_queue.
//	// Qeueu of vertices having had candidates at least once. It is used for build index after the labels_table is finished.
//	static vector<idi> once_candidated_queue(num_v); // If v is in the queue, it means v had candidated at least once in the batch.
//	static idi end_once_candidated_queue = 0;
//	static vector<bool> once_candidated(num_v, false);
	// Every vertex has its candidate set and queue.
	printf("@646 batch_process\n"); fflush(stdout);//test
	static vector<ShortIndex> short_index(num_v, ShortIndex(roots_size));
	printf("@648 batch_process\n"); fflush(stdout);//test
	// Distance table of shortest distance from roots to other vertices.
	static vector< vector<weighti> > dists_table(BATCH_SIZE, vector<weighti>(num_v, WEIGHTI_MAX)); // dists_table[r][v]: shortest distances from r to v.
	// Labels distance table only record distances of labels.
	static vector< vector<weighti> > labels_table(num_v, vector<weighti>(BATCH_SIZE, WEIGHTI_MAX)); // labels_table[v][r]: if smaller than INF, label (r, labels-table[v][r]) should be added into L[v].

	// At the beginning of a batch, initialize the labels L and distance buffer dist_matrix;
	printf("initilizing...\n"); fflush(stdout);//test
	initialize(
			short_index,
			dists_table,
			labels_table,
			active_queue,
			end_active_queue,
			//once_candidated_queue,
			//end_once_candidated_queue,
			//once_candidated,
			//b_id,
			roots_start,
			roots_size,
			L);

//	weighti iter = 0; // The iterator, also the distance for current iteration
	initializing_time += WallTimer::get_time_mark();


	idi iter = 0;//test
	while (0 != end_active_queue) {
		printf("iter: %u\n", iter++); fflush(stdout);//test
		candidating_time -= WallTimer::get_time_mark();
//		candidating_ins_count.measure_start();
		// Traverse the active queue, every active vertex sends distances to its neighbors
		for (idi i_queue = 0; i_queue < end_active_queue; ++i_queue) {
			idi v_head = active_queue[i_queue];
			is_active[v_head] = false; // reset is_active

			push_labels(
					v_head,
					roots_start,
					G,
					dists_table,
					labels_table,
					short_index,
					has_cand_queue,
					end_has_cand_queue,
					had_candidates);
		}
		end_active_queue = 0; // Set the active_queue empty
//		candidating_ins_count.measure_stop();
		candidating_time += WallTimer::get_time_mark();
		adding_time -= WallTimer::get_time_mark();
//		adding_ins_count.measure_start();

		// Traverse vertices in the has_cand_queue to insert labels
		for (idi i_queue = 0; i_queue < end_has_cand_queue; ++i_queue) {
			idi v_id = has_cand_queue[i_queue];
			had_candidates[v_id] = false; // reset had_candidates
			bool need_activate = false;
			// Traverse v_id's all candidates
			for (inti cand_root_id = 0; cand_root_id < roots_size; ++cand_root_id) {
				weighti tmp_dist_v_c = short_index[v_id].candidates_dists[cand_root_id];
				if (WEIGHTI_MAX == tmp_dist_v_c) {
					continue;
				}
				//printf("(v_id: %u cand: %u dist: %u)\n", v_id, cand_root_id + roots_start, short_index[v_id].candidates_dists[cand_root_id]); fflush(stdout);//test
				short_index[v_id].candidates_dists[cand_root_id] = WEIGHTI_MAX;
				weighti query_dist_v_c;
				if (WEIGHTI_MAX == 
						(query_dist_v_c = distance_query(
										 v_id,
										 cand_root_id,
										 labels_table,
										 short_index,
										 roots_start,
										 tmp_dist_v_c))) {
					// Record the new distance in the labels_table
					labels_table[v_id][cand_root_id] = tmp_dist_v_c;
					need_activate = true;
				} else {
					// Update the dists_table
					dists_table[cand_root_id][v_id] = query_dist_v_c;
					// Need to send back the distance
					sending_back();
				}
			}
			if (need_activate) {
				if (!is_active[v_id]) {
					is_active[v_id] = true;
					active_queue[end_active_queue++] = v_id;
				}
			}
			//inti bound_cand_i = short_index[v_id].end_candidates_que;
			//for (inti cand_i = 0; cand_i < bound_cand_i; ++cand_i) {
			//	inti cand_root_id = short_index[v_id].candidates_que[cand_i];
			//	short_index[v_id].is_candidate[cand_root_id] = false;
			//	// Only insert cand_root_id into v_id's label if its distance to v_id is shorter than existing distance
			//	if ( distance_query(
			//					cand_root_id,
			//					v_id,
			//					roots_start,
			//					L,
			//					dist_matrix,
			//					iter) ) {
			//		if (!is_active[v_id]) {
			//			is_active[v_id] = true;
			//			active_queue[end_active_queue++] = v_id;
			//		}
			//		++inserted_count;
			//		// The candidate cand_root_id needs to be added into v_id's label
			//		insert_label_only(
			//				cand_root_id,
			//				v_id,
			//				roots_start,
			//				roots_size,
			//				L,
			//				dist_matrix,
			//				iter);
			//	}
			//}
			//short_index[v_id].end_candidates_que = 0;
			//if (0 != inserted_count) {
			//	// Update other arrays in L[v_id] if new labels were inserted in this iteration
			//	update_label_indices(
			//					v_id,
			//					inserted_count,
			//					L,
			//					short_index,
			//					b_id,
			//					iter);
			//}
		}
		end_has_cand_queue = 0; // Set the has_cand_queue empty
//		adding_ins_count.measure_stop();
		adding_time += WallTimer::get_time_mark();
	}

//	{// test
//		for (idi v = 0; v < num_v; ++v) {
//			for (idi r = roots_start; r < roots_start + roots_size; ++r) {
//				if (WEIGHTI_MAX != labels_table[v][r]) {
//					printf("(v: %u r: %u d: %u)\n", v, r, labels_table[v][r]);
//				}
//			}
//		}
//
//	}

	// Update the index according to labels_table
	update_index(
			L,
			labels_table,
			roots_start,
			roots_size,
			num_v);

	// Reset the dist_matrix
	initializing_time -= WallTimer::get_time_mark();
//	init_dist_matrix_time -= WallTimer::get_time_mark();
//	reset_at_end(
//			roots_start,
//			roots_size,
//			L,
//			dist_matrix);
//	init_dist_matrix_time += WallTimer::get_time_mark();
	initializing_time += WallTimer::get_time_mark();


//	double total_time = time_can + time_add;
//	printf("Candidating time: %f (%f%%)\n", time_can, time_can / total_time * 100);
//	printf("Adding time: %f (%f%%)\n", time_add, time_add / total_time * 100);
}



void WeightedVertexCentricPLL::construct(const WeightedGraph &G)
{
	idi num_v = G.get_num_v();
	L.resize(num_v);
//	idi remainer = num_v % BATCH_SIZE;
//	idi b_i_bound = num_v / BATCH_SIZE;
//	cache_miss.measure_start();

	// Naive version: activate the whole graph
	BATCH_SIZE = num_v;

	double time_labeling = -WallTimer::get_time_mark();
	printf("batch_processing...\n"); fflush(stdout);//test
	batch_process(
			G,
			0 * BATCH_SIZE,
			BATCH_SIZE,
			L);

//	for (idi b_i = 0; b_i < b_i_bound; ++b_i) {
//		batch_process(
//				G,
//				b_i,
//				b_i * BATCH_SIZE,
//				BATCH_SIZE,
//				L);
//	}
//	if (remainer != 0) {
//		batch_process(
//				G,
//				b_i_bound,
//				b_i_bound * BATCH_SIZE,
//				remainer,
//				L);
//	}
	time_labeling += WallTimer::get_time_mark();
//	cache_miss.measure_stop();

	// Test
	setlocale(LC_NUMERIC, ""); // For print large number with comma
	printf("BATCH_SIZE: %u\n", BATCH_SIZE);
//	printf("BP_Size: %u\n", BITPARALLEL_SIZE);
	printf("Initializing: %f %.2f%%\n", initializing_time, initializing_time / time_labeling * 100);
//		printf("\tinit_start_reset_time: %f (%f%%)\n", init_start_reset_time, init_start_reset_time / initializing_time * 100);
//		printf("\tinit_index_time: %f (%f%%)\n", init_index_time, init_index_time / initializing_time * 100);
//			printf("\t\tinit_indicators_time: %f (%f%%)\n", init_indicators_time, init_indicators_time / init_index_time * 100);
//		printf("\tinit_dist_matrix_time: %f (%f%%)\n", init_dist_matrix_time, init_dist_matrix_time / initializing_time * 100);
	printf("Candidating: %f %.2f%%\n", candidating_time, candidating_time / time_labeling * 100);
	printf("Adding: %f %.2f%%\n", adding_time, adding_time / time_labeling * 100);
//		printf("distance_query_time: %f %.2f%%\n", distance_query_time, distance_query_time / time_labeling * 100);
//		uint64_t total_check_count = bp_hit_count + normal_check_count;
//		printf("total_check_count: %'lu\n", total_check_count);
//		printf("bp_hit_count: %'lu %.2f%%\n",
//						bp_hit_count,
//						bp_hit_count * 100.0 / total_check_count);
//		printf("normal_check_count: %'lu %.2f%%\n", normal_check_count, normal_check_count * 100.0 / total_check_count);
//		printf("total_candidates_num: %'lu set_candidates_num: %'lu %.2f%%\n",
//							total_candidates_num,
//							set_candidates_num,
//							set_candidates_num * 100.0 / total_candidates_num);
//		printf("\tnormal_hit_count (to total_check, to normal_check): %lu (%f%%, %f%%)\n",
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

void WeightedVertexCentricPLL::switch_labels_to_old_id(
								const vector<idi> &rank2id,
								const vector<idi> &rank)
{
	idi label_sum = 0;
	idi test_label_sum = 0;

//	idi num_v = rank2id.size();
	idi num_v = rank.size();
	vector< vector< pair<idi, weighti> > > new_L(num_v);

	for (idi v_id = 0; v_id < num_v; ++v_id) {
		idi new_v = rank2id[v_id];
		const IndexType &Lv = L[v_id];
		idi l_i_bound = Lv.vertices.size();
		label_sum += l_i_bound;
		for (idi l_i = 0; l_i < l_i_bound; ++l_i) {
			idi tail = Lv.vertices[l_i];
			weighti dist = Lv.distances[l_i];
			new_L[new_v].push_back(make_pair(tail, dist));
			++test_label_sum;
		}
	}
//	for (idi v_id = 0; v_id < num_v; ++v_id) {
//		idi new_v = rank2id[v_id];
//		const IndexType &Lv = L[v_id];
//		// Traverse v_id's all existing labels
//		for (inti b_i = 0; b_i < Lv.batches.size(); ++b_i) {
//			idi id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
//			idi dist_start_index = Lv.batches[b_i].start_index;
//			idi dist_bound_index = dist_start_index + Lv.batches[b_i].size;
//			// Traverse dist_matrix
//			for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
//				label_sum += Lv.distances[dist_i].size;
//				idi v_start_index = Lv.distances[dist_i].start_index;
//				idi v_bound_index = v_start_index + Lv.distances[dist_i].size;
//				inti dist = Lv.distances[dist_i].dist;
//				for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
//					idi tail = Lv.vertices[v_i] + id_offset;
////					idi new_tail = rank2id[tail];
////					new_L[new_v].push_back(make_pair(new_tail, dist));
//					new_L[new_v].push_back(make_pair(tail, dist));
//					++test_label_sum;
//				}
//			}
//		}
//	}
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

//	// Try query
//	idi u;
//	idi v;
//	while (std::cin >> u >> v) {
//		weighti dist = WEIGHTI_MAX;
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
#endif /* INCLUDES_PADO_WEIGHTED_H_ */
