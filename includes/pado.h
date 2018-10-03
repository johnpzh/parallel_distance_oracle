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
using std::bitset;
using std::stable_sort;
using std::min;
using std::fill;

namespace PADO {

const inti BATCH_SIZE = 1024; // The size for regular batch and bit array.



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

			}
		};

		struct DistanceIndexType {
			idi start_index; // Index to the array vertices where the same-ditance vertices start
			inti size; // Number of the same-distance vertices
			smalli dist; // The real distance

			DistanceIndexType(idi start_index_, inti size_, smalli dist_):
						start_index(start_index_), size(size_), dist(dist_)
			{

			}
		};

		vector<Batch> batches; // Batch info
		vector<DistanceIndexType> distances; // Distance info
		vector<idi> vertices; // Vertices in the label, preresented as temperory ID
	};

	// Structure for the type of temporary label
	struct ShortIndex {
		bitset<BATCH_SIZE> indicator; // Global indicator, indicator[r] is set means root r once selected as candidate already
		bitset<BATCH_SIZE> candidates; // Candidates one iteration, candidates[r] is set means root r is candidate in this iteration
	};

	vector<IndexType> L;
	void construct(const Graph &G);
	inline void batch_process(
			const Graph &G,
			idi b_id,
			idi root_start,
			inti roots_size,
			vector<IndexType> &L);

	inline void initialize(
				vector<ShortIndex> &short_index,
				vector< vector<smalli> > &dist_matrix,
				vector<idi> &active_queue,
				inti &end_active_queue,
				vector<bool> &got_labels,
				idi b_id,
				idi roots_start,
				inti roots_size,
				vector<IndexType> &L,
				idi num_v);
	inline void push_labels(
				idi v_head,
				idi roots_start,
				const Graph &G,
				const vector<IndexType> &L,
				vector<ShortIndex> &short_index,
				vector<idi> &candidate_queue,
				inti &end_candidate_queue,
				vector<bool> &got_candidates);
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
				vector<bool> &got_labels,
				idi b_id,
				smalli iter);

	// Test only
	uint64_t check_count = 0;
	double initializing_time = 0;
	double candidating_time = 0;
	double adding_time = 0;
	double distance_query_time = 0;
//	double distance_query_final_check_time = 0;
	double init_index_time = 0;
	double init_dist_matrix_time = 0;
	double init_start_reset_time = 0;
	// End test



public:
	VertexCentricPLL() = default;
	VertexCentricPLL(const Graph &G);

	weighti query(
			idi u,
			idi v);

	void print();
	void switch_labels_to_old_id(const vector<idi> &rank2id);

}; // class VertexCentricPLL

VertexCentricPLL::VertexCentricPLL(const Graph &G)
{
	construct(G);
}

// Function for initializing at the begin of a batch
// For a batch, initialize the temporary labels and real labels of roots;
// traverse roots' labels to initialize distance buffer;
// unset flag arrays is_active and got_labels
inline void VertexCentricPLL::initialize(
			vector<ShortIndex> &short_index,
			vector< vector<smalli> > &dist_matrix,
			vector<idi> &active_queue,
			inti &end_active_queue,
			vector<bool> &got_labels,
			idi b_id,
			idi roots_start,
			inti roots_size,
			vector<IndexType> &L,
			idi num_v)
{
	init_start_reset_time -= WallTimer::get_time_mark();
	idi roots_bound = roots_start + roots_size;
	{
		// got_labels
//		fill(got_labels.begin(), got_labels.end(), 0);
//		got_labels[r_real_id] = true;

		fill(got_labels.begin(), got_labels.begin() + roots_start, 0);
		fill(got_labels.begin() + roots_start, got_labels.begin() + roots_bound, 1);
		fill(got_labels.begin() + roots_bound, got_labels.end(), 0);

		//active_queue
//		active_queue[end_active_queue++] = r_real_id;
		for (idi r_real_id = roots_start; r_real_id < roots_bound; ++r_real_id) {
			active_queue[end_active_queue++] = r_real_id;
		}
	}
	init_start_reset_time += WallTimer::get_time_mark();
	init_index_time -= WallTimer::get_time_mark();
	// Short_index
	{
		idi v = 0;
		for ( ; v < roots_start; ++v) {
			short_index[v].indicator.reset();
		}
		for ( ; v < roots_bound; ++v) {
			short_index[v].indicator.reset();
			short_index[v].indicator.set(v - roots_start);
		}
		for (; v < num_v; ++v) {
			short_index[v].indicator.reset();
		}
	}
//
	// Real Index
	{
//		IndexType &Lr = nullptr;
//		IndexType *Lr = nullptr;
		for (idi r_id = 0; r_id < roots_size; ++r_id) {
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
	init_index_time += WallTimer::get_time_mark();
	init_dist_matrix_time -= WallTimer::get_time_mark();
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
			IndexType &Lr = L[r_id + roots_start];
			fill(dist_matrix[r_id].begin(),
					dist_matrix[r_id].begin() + r_id + roots_start + 1,
					SMALLI_MAX);
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
	init_dist_matrix_time += WallTimer::get_time_mark();

//	init_start_reset_time -= WallTimer::get_time_mark();
//	for (idi v = 0; v < num_v; ++v) {
//		short_index[v].indicator.reset();
//	}
//	fill(got_labels.begin(), got_labels.end(), 0);

//	init_start_reset_time += WallTimer::get_time_mark();

//	// Initialize roots labels and distance matrix
//	for (idi r_id = 0; r_id < roots_size; ++r_id) {
//		init_index_time -= WallTimer::get_time_mark();
//		// Initialize roots labels
//		idi r_real_id = r_id + roots_start;
//		// Short Index
//		short_index[r_real_id].indicator.set(r_id);
//		// Real Index
//		IndexType &Lr = L[r_real_id];
//		Lr.batches.push_back(IndexType::Batch(
//								b_id, // Batch ID
//								Lr.distances.size(), // start_index
//								1)); // size
//		Lr.distances.push_back(IndexType::DistanceIndexType(
//								Lr.vertices.size(), // start_index
//								1, // size
//								0)); // dist
//		Lr.vertices.push_back(r_id);
		// Push r_real_id to active queue
//		active_queue[end_active_queue++] = r_real_id;
//		got_labels[r_real_id] = true;
//		init_index_time += WallTimer::get_time_mark();
//
//		init_dist_matrix_time -= WallTimer::get_time_mark();
//		// Initialize distance matrix
//		fill(dist_matrix[r_id].begin(),
//				dist_matrix[r_id].begin() + r_real_id + 1,
//				SMALLI_MAX);
//		inti b_i_bound = Lr.batches.size();
//		_mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
//		_mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
//		_mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
//		for (inti b_i = 0; b_i < b_i_bound; ++b_i) {
//			idi id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
//			idi dist_start_index = Lr.batches[b_i].start_index;
//			idi dist_bound_index = dist_start_index + Lr.batches[b_i].size;
//			// Traverse dist_matrix
//			for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
//				idi v_start_index = Lr.distances[dist_i].start_index;
//				idi v_bound_index = v_start_index + Lr.distances[dist_i].size;
//				smalli dist = Lr.distances[dist_i].dist;
//				for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
//					dist_matrix[r_id][Lr.vertices[v_i] + id_offset] = dist;
//				}
//			}
//		}
//		init_dist_matrix_time += WallTimer::get_time_mark();
//	}
}


//inline void VertexCentricPLL::initialize(
//			vector<ShortIndex> &short_index,
//			vector< vector<smalli> > &dist_matrix,
//			vector<idi> &active_queue,
//			inti &end_active_queue,
//			vector<bool> &got_labels,
//			idi b_id,
//			idi roots_start,
//			inti roots_size,
//			vector<IndexType> &L,
//			idi num_v)
//{
//	init_start_reset_time -= WallTimer::get_time_mark();
//	for (idi v = 0; v < num_v; ++v) {
//		short_index[v].indicator.reset();
//	}
//	fill(got_labels.begin(), got_labels.end(), 0);
//	init_start_reset_time += WallTimer::get_time_mark();
//
//	// Initialize roots labels and distance matrix
//	for (idi r_id = 0; r_id < roots_size; ++r_id) {
//		init_index_time -= WallTimer::get_time_mark();
//		// Initialize roots labels
//		idi r_real_id = r_id + roots_start;
//		// Short Index
//		short_index[r_real_id].indicator.set(r_id);
//		// Real Index
//		IndexType &Lr = L[r_real_id];
//		Lr.batches.push_back(IndexType::Batch(
//								b_id, // Batch ID
//								Lr.distances.size(), // start_index
//								1)); // size
//		Lr.distances.push_back(IndexType::DistanceIndexType(
//								Lr.vertices.size(), // start_index
//								1, // size
//								0)); // dist
//		Lr.vertices.push_back(r_id);
//		// Push r_real_id to active queue
//		active_queue[end_active_queue++] = r_real_id;
//		got_labels[r_real_id] = true;
//		init_index_time += WallTimer::get_time_mark();
//
//		init_dist_matrix_time -= WallTimer::get_time_mark();
//		// Initialize distance matrix
//		fill(dist_matrix[r_id].begin(),
//				dist_matrix[r_id].begin() + r_real_id + 1,
//				SMALLI_MAX);
//		inti b_i_bound = Lr.batches.size();
//		_mm_prefetch(&Lr.batches[0], _MM_HINT_T0);
//		_mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
//		_mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
//		for (inti b_i = 0; b_i < b_i_bound; ++b_i) {
//			idi id_offset = Lr.batches[b_i].batch_id * BATCH_SIZE;
//			idi dist_start_index = Lr.batches[b_i].start_index;
//			idi dist_bound_index = dist_start_index + Lr.batches[b_i].size;
//			// Traverse dist_matrix
//			for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
//				idi v_start_index = Lr.distances[dist_i].start_index;
//				idi v_bound_index = v_start_index + Lr.distances[dist_i].size;
//				smalli dist = Lr.distances[dist_i].dist;
//				for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
//					dist_matrix[r_id][Lr.vertices[v_i] + id_offset] = dist;
//				}
//			}
//		}
//		init_dist_matrix_time += WallTimer::get_time_mark();
//	}
//}

// Function that pushes v_head's labels to v_head's every neighbor
inline void VertexCentricPLL::push_labels(
				idi v_head,
				idi roots_start,
				const Graph &G,
				const vector<IndexType> &L,
				vector<ShortIndex> &short_index,
				vector<idi> &candidate_queue,
				inti &end_candidate_queue,
				vector<bool> &got_candidates)
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
		if (v_tail < roots_start) { // v_tail has higher rank than any roots, then no roots can push new labels to it.
			return;
		}
		if (v_tail <= Lv.vertices[l_i_start] + roots_start) { // v_tail has higher rank than any v_head's labels
			return;
		}
		// Traverse v_head's last inserted labels
		for (idi l_i = l_i_start; l_i < l_i_bound; ++l_i) {
			idi label_root_id = Lv.vertices[l_i];
			if (v_tail <= label_root_id + roots_start) {
				// v_tail has higher rank than all remaining labels
				break;
			}
			ShortIndex &SI_v_tail = short_index[v_tail];
			if (SI_v_tail.indicator[label_root_id]) {
				// The label is alreay selected before
				continue;
			}
			SI_v_tail.indicator.set(label_root_id);
			// Record vertex label_root_id as v_tail's candidates label
			SI_v_tail.candidates.set(label_root_id);
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
// return the distance between v_id and cand_root_id based on existing labels.
// return false if shorter distance exists already, return true if the cand_root_id can be added into v_id's label.
inline bool VertexCentricPLL::distance_query(
			idi cand_root_id,
			idi v_id,
			idi roots_start,
			const vector<IndexType> &L,
			const vector< vector<smalli> > &dist_matrix,
			smalli iter)
{
	distance_query_time -= WallTimer::get_time_mark();

	idi cand_real_id = cand_root_id + roots_start;
	const IndexType &Lv = L[v_id];
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
//				distance_query_final_check_time -= WallTimer::get_time_mark();
				inti d_tmp = dist + dist_matrix[cand_root_id][v];
//				++check_count;
				if (d_tmp <= iter) {
					distance_query_time += WallTimer::get_time_mark();
					return false;
				}
//				distance_query_final_check_time += WallTimer::get_time_mark();
			}
		}
	}
	distance_query_time += WallTimer::get_time_mark();
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
				vector<bool> &got_labels,
				idi b_id,
				smalli iter)
{
	IndexType &Lv = L[v_id];
	if (got_labels[v_id]) {
		// Increase the batches' last element's size because a new distance element need to be added
		++(Lv.batches.rbegin() -> size);
	} else {
		got_labels[v_id] = true;
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

inline void VertexCentricPLL::batch_process(
						const Graph &G,
						idi b_id,
						idi roots_start, // start id of roots
						inti roots_size, // how many roots in the batch
						vector<IndexType> &L)
{
//	double time_can = 0;
//	double time_add = 0;

	initializing_time -= WallTimer::get_time_mark();
	static const idi num_v = G.get_num_v();
	static vector<idi> active_queue(num_v);
	static idi end_active_queue = 0;
	static vector<idi> candidate_queue(num_v);
	static idi end_candidate_queue = 0;
	static vector<ShortIndex> short_index(num_v);
//	vector<ShortIndex> short_index(num_v);
	static vector< vector<smalli> > dist_matrix(roots_size, vector<smalli>(num_v));
	static vector<bool> got_labels(num_v, false); // got_labels[v] is true means vertex v got new labels in this batch
//	vector<bool> got_labels(num_v, false); // got_labels[v] is true means vertex v got new labels in this batch
	static vector<bool> got_candidates(num_v, false); // got_candidates[v] is true means vertex v is in the queue candidate_queue
	static vector<bool> is_active(num_v, false);// is_active[v] is true means vertex v is in the active queue.

	// At the beginning of a batch, initialize the labels L and distance buffer dist_matrix;
	initialize(
			short_index,
			dist_matrix,
			active_queue,
			end_active_queue,
			got_labels,
			b_id,
			roots_start,
			roots_size,
			L,
			num_v);

	smalli iter = 0; // The iterator, also the distance for current iteration
	initializing_time += WallTimer::get_time_mark();


	while (0 != end_active_queue) {
		candidating_time -= WallTimer::get_time_mark();
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
					got_candidates);
		}
		end_active_queue = 0; // Set the active_queue empty
		candidating_time += WallTimer::get_time_mark();
		adding_time -= WallTimer::get_time_mark();

		// Traverse vertices in the candidate_queue to insert labels
		for (idi i_queue = 0; i_queue < end_candidate_queue; ++i_queue) {
			idi v_id = candidate_queue[i_queue];
			inti inserted_count = 0; //recording number of v_id's truly inserted candidates
			got_candidates[v_id] = false; // reset got_candidates
			// Traverse v_id's all candidates
			for (inti cand_root_id = 0; cand_root_id < roots_size; ++cand_root_id) {
				if (!short_index[v_id].candidates[cand_root_id]) {
					// Root cand_root_id is not vertex v_id's candidate
					continue;
				}
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
								got_labels,
								b_id,
								iter);
			}
		}
		end_candidate_queue = 0; // Set the candidate_queue empty
		adding_time += WallTimer::get_time_mark();
	}
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
	double time_labeling = -WallTimer::get_time_mark();

	for (idi b_i = 0; b_i < b_i_bound; ++b_i) {
//		printf("b_i: %u\n", b_i);//test
		batch_process(
				G,
				b_i,
				b_i * BATCH_SIZE,
				BATCH_SIZE,
				L);
	}
	if (remainer != 0) {
//		printf("b_i: %u\n", b_i_bound);//test
		batch_process(
				G,
				b_i_bound,
				b_i_bound * BATCH_SIZE,
				remainer,
				L);
	}
	time_labeling += WallTimer::get_time_mark();

	// Test
	printf("check_count: %llu\n", check_count);
	double total_time = initializing_time + candidating_time + adding_time;
	printf("Initializing: %f (%f%%)\n", initializing_time, initializing_time / total_time * 100);
	printf("Candidating: %f (%f%%)\n", candidating_time, candidating_time / total_time * 100);
	printf("Adding: %f (%f%%)\n", adding_time, adding_time / total_time * 100);
	printf("Labeling: %f\n", time_labeling);
//	printf("10%% Labeling: %f (%f%%)\n", time_ten, time_ten/time_labeling * 100);
	printf("distance_query_time: %f (%f%%)\n", distance_query_time, distance_query_time / time_labeling * 100);
//	printf("distance_query_final_check_time: %f (%f%%)\n", distance_query_final_check_time, distance_query_final_check_time / time_labeling * 100);
	printf("init_start_reset_time: %f (%f%%)\n", init_start_reset_time, init_start_reset_time / initializing_time * 100);
	printf("init_index_time: %f (%f%%)\n", init_index_time, init_index_time / initializing_time * 100);
	printf("init_dist_matrix_time: %f (%f%%)\n", init_dist_matrix_time, init_dist_matrix_time / initializing_time * 100);
	// End test
}

void VertexCentricPLL::switch_labels_to_old_id(const vector<idi> &rank2id)
{
	idi label_sum = 0;
	idi test_label_sum = 0;

	idi num_v = rank2id.size();
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
					idi new_tail = rank2id[tail];
					new_L[new_v].push_back(make_pair(new_tail, dist));
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

//	// Try query
//	idi u;
//	idi v;
//	while (std::cin >> u >> v) {
//		const auto &Lu = new_L[u];
//		const auto &Lv = new_L[v];
//		weighti dist = WEIGHTI_MAX;
//		unordered_map<idi, weighti> markers;
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
