/*
 * pado.h
 *
 *  Created on: Dec 18, 2018
 *      Author: Zhen Peng
 */

#ifndef INCLUDES_PADO_WEIGHTED_UNV_UNP_H_
#define INCLUDES_PADO_WEIGHTED_UNV_UNP_H_

#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <iostream>
#include <climits>
#include <xmmintrin.h>
#include <immintrin.h>
#include <bitset>
#include <cmath>
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
//	inti BATCH_SIZE = 512; // Here is for the whole graph, so make it non-const
//const inti BITPARALLEL_SIZE = 50; // Weighted graphs cannot use Bit Parallel technique

//	// AVX-512 constant variables
//	static const inti NUM_P_INT = 16;
//	static const __m512i INF_v = _mm512_set1_epi32(WEIGHTI_MAX);
//	static const __m512i UNDEF_i32_v = _mm512_undefined_epi32();
//	static const __m512i LOWEST_BYTE_MASK = _mm512_set1_epi32(0xFF);
//	static const __m128i INF_v_128i = _mm_set1_epi8(-1);

//// Batch based processing, 09/11/2018
template <inti BATCH_SIZE = 512>
class WeightedVertexCentricPLL {
private:
	idi num_v_ = 0;
	// Structure for the type of label
	struct IndexType {
		vector<idi> vertices; // Vertices in the label, preresented as temperory ID
		vector<weighti> distances;
	}; //__attribute__((aligned(64)));

	// Structure for the type of temporary label
	struct ShortIndex {
		// Use a queue to store candidates
		vector<inti> candidates_que;
		inti end_candidates_que = 0;
		// Use a array to store distances of candidates; length of roots_size
		vector<weighti> candidates_dists; // record the distances to candidates. If candidates_dists[c] = INF, then c is NOT a candidate
			// The candidates_dists is also used for distance query.

		// Use a queue to store temporary labels in this batch; use it so don't need to traverse vertices_dists.
		vector<inti> vertices_que; // Elements in vertices_que are roots_id, not real id
		idi end_vertices_que = 0;
		// Use an array to store distances to roots; length of roots_size
		vector<weighti> vertices_dists; // labels_table

		// Usa a queue to record which roots have reached this vertex in this batch.
		// It is used for reset the dists_table
		vector<inti> reached_roots_que;
		idi end_reached_roots_que = 0;

		// Use a queue to store last inserted labels (IDs); distances are stored in vertices_dists.
		vector<inti> last_new_roots;
		idi end_last_new_roots = 0;

		// Monotone checker
		bool are_dists_monotone = true;
		weighti max_dist_batch = 0;

		ShortIndex() = default;
		explicit ShortIndex(idi num_roots) {
			candidates_que.resize(num_roots);
			candidates_dists.resize(num_roots, WEIGHTI_MAX);
			last_new_roots.resize(num_roots);
			vertices_que.resize(num_roots);
			vertices_dists.resize(num_roots, WEIGHTI_MAX);
			reached_roots_que.resize(num_roots);
		}
	}; //__attribute__((aligned(64)));

	// Structure of the public ordered index for distance queries.
	struct IndexOrdered {
		vector<idi> label_id;
		vector<weighti> label_dists;
	};

	vector<IndexType> L;
	vector<IndexOrdered> Index; // Ordered labels for original vertex ID

	void construct(const WeightedGraph &G);
	inline void vertex_centric_labeling_in_batches(
			const WeightedGraph &G,
			idi b_id,
			idi root_start,
			inti roots_size,
			vector<IndexType> &L,
			vector<idi> &active_queue,
			idi end_active_queue,
			vector<bool> &is_active,
			vector<idi> &has_cand_queue,
			idi end_has_cand_queue,
			vector<bool> &has_candidates,
			vector< vector<weighti> > &dists_table,
			vector< vector<weighti> > &roots_labels_buffer,
			vector<ShortIndex> &short_index,
			vector<idi> &has_new_labels_queue,
			idi end_has_new_labels_queue,
			vector<bool> &has_new_labels);
	inline void initialize_tables(
			vector<ShortIndex> &short_index,
			vector< vector<weighti> > &roots_labels_buffer,
			vector<idi> &active_queue,
			idi &end_active_queue,
			idi roots_start,
			inti roots_size,
			vector<IndexType> &L,
			vector<idi> &has_new_labels_queue,
			idi &end_has_new_labels_queue,
			vector<bool> &has_new_labels);
	inline void send_messages(
			idi v_head,
			idi roots_start,
			const WeightedGraph &G,
			vector< vector<weighti> > &dists_table,
			vector<ShortIndex> &short_index,
			vector<idi> &has_cand_queue,
			idi &end_has_cand_queue,
			vector<bool> &has_candidates);
	inline weighti distance_query(
			idi v_id,
			idi cand_root_id,
			const vector< vector<weighti> > &roots_labels_buffer,
			const vector<ShortIndex> &short_index,
			const vector<IndexType> &L,
			idi roots_start,
			weighti tmp_dist_v_c);
	inline void reset_tables(
			vector<ShortIndex> &short_index,
			idi roots_start,
			inti roots_size,
			const vector<IndexType> &L,
			vector< vector<weighti> > &dists_table,
			vector< vector<weighti> > &roots_labels_buffer,
			const vector<idi> &has_new_labels_queue,
			idi end_has_new_labels_queue);
	inline void update_index(
			vector<IndexType> &L,
			vector<ShortIndex> &short_index,
			idi roots_start,
			vector<idi> &has_new_labels_queue,
			idi &end_has_new_labels_queue,
			vector<bool> &has_new_labels);
	inline void send_back(
			idi s,
			idi r_root_id,
			const WeightedGraph &G,
			vector< vector<weighti> > &dists_table,
			vector<ShortIndex> &short_index,
			idi roots_start);
	inline void filter_out_labels(
			const vector<idi> &has_new_labels_queue,
			idi end_has_new_labels_queue,
			vector<ShortIndex> &short_index,
			idi roots_start);
	inline void int_i32scatter_epi8(
			weighti *base_addr, 
			__m512i vindex, 
			weighti a);
	inline void int_mask_i32scatter_epi8(
			weighti *base_addr, 
			__mmask16 k, 
			__m512i vindex, 
			weighti a);
	inline void epi32_mask_i32scatter_epi8(
			weighti *base_addr,
			__mmask16 mask,
			__m512i vindex,
			__m512i a);
	inline void epi32_mask_unpack_into_queue(
			vector<inti> &queue,
			inti &end_queue,
			__mmask16 mask,
			__m512i a);


	// Statistics
//	uint64_t normal_hit_count = 0;
	//uint64_t bp_hit_count = 0;
//	uint64_t total_check_count = 0;
	//uint64_t normal_check_count = 0;
	//uint64_t check_count = 0;
	//uint64_t l_l_hit_count = 0;
	//uint64_t vl_cl_hit_count = 0;
	//uint64_t vl_cc_hit_count = 0;
	//uint64_t vc_cl_hit_count = 0;
	//uint64_t vc_cc_hit_count = 0;
//	uint64_t total_candidates_num = 0;
//	uint64_t set_candidates_num = 0;
//	uint64_t monotone_count = 0;
//	uint64_t checked_count = 0;
//	double initializing_time = 0;
//	double candidating_time = 0;
//	double adding_time = 0;
//	double distance_query_time = 0;
//	double correction_time = 0;
//	double init_index_time = 0;
//	double init_dist_matrix_time = 0;
//	double init_start_reset_time = 0;
//	double init_indicators_time = 0;
//	pair<uint64_t, uint64_t> simd_util = make_pair(0, 0); // .first is numerator, .second is denominator
//	pair<uint64_t, uint64_t> simd_full_count = make_pair(0, 0);
//	L2CacheMissRate cache_miss;
//	TotalInstructsExe candidating_ins_count;
//	TotalInstructsExe adding_ins_count;
//	TotalInstructsExe bp_labeling_ins_count;
//	TotalInstructsExe bp_checking_ins_count;
//	TotalInstructsExe dist_query_ins_count;
	// End Statistics



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
	void store_index_to_file(
			const char *filename,
			const vector<idi> &rank2id);
	void load_index_from_file(
			const char *filename);
	void order_labels(
			const vector<idi> &rank2id,
			const vector<idi> &rank);
	weighti query_distance(
			idi a,
			idi b);

}; // class WeightedVertexCentricPLL
template <inti BATCH_SIZE>
WeightedVertexCentricPLL<BATCH_SIZE>::WeightedVertexCentricPLL(const WeightedGraph &G)
{
	construct(G);
}

// Function for initializing at the begin of a batch
// For a batch, initialize the temporary labels and real labels of roots;
// traverse roots' labels to initialize distance buffer;
// unset flag arrays is_active and got_labels
template <inti BATCH_SIZE>
inline void WeightedVertexCentricPLL<BATCH_SIZE>::initialize_tables(
			vector<ShortIndex> &short_index,
			vector< vector<weighti> > &roots_labels_buffer,
			vector<idi> &active_queue,
			idi &end_active_queue,
			idi roots_start,
			inti roots_size,
			vector<IndexType> &L,
			vector<idi> &has_new_labels_queue,
			idi &end_has_new_labels_queue,
			vector<bool> &has_new_labels)
{
	idi roots_bound = roots_start + roots_size;
//	init_dist_matrix_time -= WallTimer::get_time_mark();
	// Distance Tables
	{
		// Traverse all roots
		for (idi r_root_id = 0; r_root_id < roots_size; ++r_root_id) {
			idi r_real_id = r_root_id + roots_start;
			const IndexType &Lr = L[r_real_id];
			// Traverse r_real_id's all labels to initial roots_labels_buffer
//			// Vectorized Version
//			inti remainder_simd = Lr.vertices.size() % NUM_P_INT;
//			idi bound_l_i = Lr.vertices.size() - remainder_simd;
//			for (idi l_i = 0; l_i < bound_l_i; l_i += NUM_P_INT) {
//				__m512i v_id_v = _mm512_loadu_epi32(&Lr.vertices[l_i]);
//				__m128i tmp_dists_128i = _mm_loadu_epi8(&Lr.distances[l_i]);
//				//__m512i dists_v = _mm512_cvtepi8_epi32(tmp_dists_128i);
//				epi8_i32scatter_epi8();
//			}
//			if (remainder_simd) {
//				__mmask16 in_m = (__mmask16) ((uint16_t) 0xFFFF >> (NUM_P_INT - remainder_simd));
//			}
			// Sequential Version
			idi l_i_bound = Lr.vertices.size();
			_mm_prefetch(&Lr.vertices[0], _MM_HINT_T0);
			_mm_prefetch(&Lr.distances[0], _MM_HINT_T0);
			for (idi l_i = 0; l_i < l_i_bound; ++l_i) {
				roots_labels_buffer[r_root_id][Lr.vertices[l_i]] = Lr.distances[l_i];
			}

			roots_labels_buffer[r_root_id][r_real_id] = 0;
		}
	}
//	init_dist_matrix_time += WallTimer::get_time_mark();
//	init_index_time -= WallTimer::get_time_mark();
	// Short_index
	{
		for (idi r_real_id = roots_start; r_real_id < roots_bound; ++r_real_id) {
			ShortIndex &SI_r = short_index[r_real_id];
			idi r_root_id = r_real_id - roots_start;
			// Record itself as last inserted label distance
			SI_r.vertices_que[SI_r.end_vertices_que++] = r_root_id;
			SI_r.vertices_dists[r_root_id] = 0;
			SI_r.last_new_roots[SI_r.end_last_new_roots++] = r_root_id;
		}
	}
//
//	init_index_time += WallTimer::get_time_mark();
//	init_start_reset_time -= WallTimer::get_time_mark();
	// TODO: parallel enqueue
	// Active queue
	{
		for (idi r_real_id = roots_start; r_real_id < roots_bound; ++r_real_id) {
			active_queue[end_active_queue++] = r_real_id;
		}
	}
//	init_start_reset_time += WallTimer::get_time_mark();

	// has_new_labels_queue: Put all roots into the has_new_labels_queue
	{
		for (idi r_real_id = roots_start; r_real_id < roots_bound; ++r_real_id) {
			has_new_labels_queue[end_has_new_labels_queue++] = r_real_id;
			has_new_labels[r_real_id] = true;
		}
	}
}

//// Function that pushes v_head's labels to v_head's every neighbor
//inline void WeightedVertexCentricPLL::send_messages(
//				idi v_head,
//				idi roots_start,
//				const WeightedGraph &G,
//				vector< vector<weighti> > &dists_table,
//				vector<ShortIndex> &short_index,
//				vector<idi> &has_cand_queue,
//				idi &end_has_cand_queue,
//				vector<bool> &has_candidates)
//{
//	ShortIndex &SI_v_head = short_index[v_head];
//	// Traverse v_head's last inserted labels
//	idi bound_r_i = SI_v_head.end_last_new_roots;
//	for (idi r_i = 0; r_i < bound_r_i; ++r_i) {
//		idi r_root_id = SI_v_head.last_new_roots[r_i]; // last inserted label r_root_id of v_head
//		idi r_real_id = r_root_id + roots_start;
//		idi dist_r_h = SI_v_head.vertices_dists[r_root_id];
//		// Traverse v_head's every neighbor v_tail
//		idi e_i_start = G.vertices[v_head];
//		idi e_i_bound = e_i_start + G.out_degrees[v_head];
//		for (idi e_i = e_i_start; e_i < e_i_bound; ++e_i) {
//			idi v_tail = G.out_edges[e_i];
//			if (v_tail <= r_real_id) {
//				// v_tail has higher rank, and so do those following neighbors
//				break;
//			}
//			weighti weight_h_t = G.out_weights[e_i];
//			ShortIndex &SI_v_tail = short_index[v_tail];
//			weighti tmp_dist_r_t = dist_r_h + weighti_h_t;
//			if (tmp_dist_r_t < dists_table[r_root_id][v_tail] &&
//					tmp_dist_r_t < SI_v_tail.candidates_dists[r_root_id]) {
//				// Mark r_root_id as a candidate of v_tail
//				if (WEIGHTI_MAX == SI_v_tail.candidates_dists[r_root_id]) {
//					// Add r_root_id into v_tail's candidates_que
//					SI_v_tail.candidates_que[SI_v_tail.end_candidates_que++] = r_root_id;
//				}
//				SI_v_tail.candidates_dists[r_root_id] = tmp_dist_r_t;
//				if (WEIGHTI_MAX == dists_table[r_root_id][v_tail]) {
//					// Add r_root_id into v_tail's reached_roots_que so to reset dists_table[r_root_id][v_tail] at the end of this batch
//					SI_v_tail.reached_roots_que[SI_v_tail.end_reached_roots_que++] = r_root_id;
//				}
//				dists_table[r_root_id][v_tail] = tmp_dist_r_t; // For filtering out longer distances coming later
//			}
//		}
//	}
//}

// Function that pushes v_head's labels to v_head's every neighbor
template <inti BATCH_SIZE>
inline void WeightedVertexCentricPLL<BATCH_SIZE>::send_messages(
				idi v_head,
				idi roots_start,
				const WeightedGraph &G,
				vector< vector<weighti> > &dists_table,
				vector<ShortIndex> &short_index,
				vector<idi> &has_cand_queue,
				idi &end_has_cand_queue,
				vector<bool> &has_candidates)
{
	ShortIndex &SI_v_head = short_index[v_head];
	// Traverse v_head's every neighbor v_tail
	idi e_i_start = G.vertices[v_head];
	idi e_i_bound = e_i_start + G.out_degrees[v_head];
	for (idi e_i = e_i_start; e_i < e_i_bound; ++e_i) {
		idi v_tail = G.out_edges[e_i];
		if (v_tail <= roots_start) { // v_tail has higher rank than any roots, then no roots can push new labels to it.
			break;
		}
		weighti weight_h_t = G.out_weights[e_i];
		ShortIndex &SI_v_tail = short_index[v_tail];
		bool got_candidates = false; // A flag indicates if v_tail got new candidates.
		// Traverse v_head's last inserted labels
//		// Vectorized Version
//		__m512i v_tail_v = _mm512_set1_epi32(v_tail);
//		__m512i roots_start_v = _mm512_set1_epi32(roots_start);
//		__m512i weight_h_t_v = _mm512_set1_epi32(weight_h_t);
//		inti remainder_simd = SI_v_head.end_last_new_roots % NUM_P_INT;
//		idi bound_r_i = SI_v_head.end_last_new_roots - remainder_simd;
//		for (idi r_i = 0; r_i < bound_r_i; r_i += NUM_P_INT) {
//			__m512i r_root_id_v = _mm512_loadu_epi32(&SI_v_head.last_new_roots[r_i]);
//			__mmask16 is_r_higher_ranked_m = _mm512_cmplt_epi32_mask(
//												_mm512_add_epi32(r_root_id_v, roots_start_v),
//												v_tail_v);
//			if (!is_r_higher_ranked_m) {
//				continue;
//			}
//			__m512i dists_r_h_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_v_head.vertices_dists[0], sizeof(weighti));
//			dists_r_h_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_r_h_v, LOWEST_BYTE_MASK);
//			// Dist from r to tail (label dist)
//			__m512i tmp_dist_r_t_v = _mm512_mask_add_epi32(INF_v, is_r_higher_ranked_m, dists_r_h_v, weight_h_t_v);
//			// Dist from r to tail (table dist)
//			__m512i table_dists_r_t_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &dists_table[v_tail][0], sizeof(weighti));
//			table_dists_r_t_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, table_dists_r_t_v, LOWEST_BYTE_MASK);
//			__mmask16 is_tmp_dist_shorter_m = _mm512_mask_cmplt_epi32_mask(is_r_higher_ranked_m, tmp_dist_r_t_v, table_dists_r_t_v);
//			if (!is_tmp_dist_shorter_m) {
//				continue;
//			}
//			// Dist from r to tail (candidate dist)
//			__m512i cand_dists_r_t_v = _mm512_mask_i32gather_epi32(INF_v, is_tmp_dist_shorter_m, r_root_id_v, &SI_v_tail.candidates_dists[0], sizeof(weighti));
//			cand_dists_r_t_v = _mm512_mask_and_epi32(INF_v, is_tmp_dist_shorter_m, cand_dists_r_t_v, LOWEST_BYTE_MASK);
//			is_tmp_dist_shorter_m = _mm512_mask_cmplt_epi32_mask(is_tmp_dist_shorter_m, tmp_dist_r_t_v, cand_dists_r_t_v);
//			if (!is_tmp_dist_shorter_m) {
//				continue;
//			}
//			// Mark r_root_id as a v_tail's candidate
//			__mmask16 not_in_cand_que = _mm512_mask_cmpeq_epi32_mask(is_tmp_dist_shorter_m, cand_dists_r_t_v, INF_v);
//			// Add r_root_id into v_tail's candidates_que
//			epi32_mask_unpack_into_queue(
//					SI_v_tail.candidates_que,
//					SI_v_tail.end_candidates_que,
//					not_in_cand_que,
//					r_root_id_v); // r_root_id_v to SI_v_tail.candidates_que
//			epi32_mask_i32scatter_epi8(
//					&SI_v_tail.candidates_dists[0],
//					is_tmp_dist_shorter_m,
//					r_root_id_v,
//					tmp_dist_r_t_v); // tmp_dist_r_t_v to SI_v_tail.candidates_dists
//			// Update v_tail's reached_roots_que
//			__mmask16 not_in_reached_roots_que = _mm512_mask_cmpeq_epi32_mask(is_tmp_dist_shorter_m, table_dists_r_t_v, INF_v);
//			epi32_mask_unpack_into_queue(
//					SI_v_tail.reached_roots_que,
//					SI_v_tail.end_reached_roots_que,
//					not_in_reached_roots_que,
//					r_root_id_v); // r_root_id_v to SI_v_tail.reached_roots_que
//			epi32_mask_i32scatter_epi8(
//					&dists_table[v_tail][0],
//					is_tmp_dist_shorter_m,
//					r_root_id_v,
//					tmp_dist_r_t_v); // tmp_dist_r_t_v to dists_table[v_tail]
//			got_candidates = true;
//		}
//		if (remainder_simd) {
//			__mmask16 in_m = (__mmask16) ((uint16_t) 0xFFFF >> (NUM_P_INT - remainder_simd));
//			__m512i r_root_id_v = _mm512_mask_loadu_epi32(UNDEF_i32_v, in_m, &SI_v_head.last_new_roots[bound_r_i]);
//			__mmask16 is_r_higher_ranked_m = _mm512_mask_cmplt_epi32_mask(
//												in_m,
//												_mm512_add_epi32(r_root_id_v, roots_start_v),
//												v_tail_v);
//			if (!is_r_higher_ranked_m) {
//				continue;
//			}
//			__m512i dists_r_h_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_v_head.vertices_dists[0], sizeof(weighti));
//			dists_r_h_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_r_h_v, LOWEST_BYTE_MASK);
//			// Dist from r to tail (label dist)
//			__m512i tmp_dist_r_t_v = _mm512_mask_add_epi32(INF_v, is_r_higher_ranked_m, dists_r_h_v, weight_h_t_v);
//			// Dist from r to tail (table dist)
//			__m512i table_dists_r_t_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &dists_table[v_tail][0], sizeof(weighti));
//			table_dists_r_t_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, table_dists_r_t_v, LOWEST_BYTE_MASK);
//			__mmask16 is_tmp_dist_shorter_m = _mm512_mask_cmplt_epi32_mask(is_r_higher_ranked_m, tmp_dist_r_t_v, table_dists_r_t_v);
//			if (!is_tmp_dist_shorter_m) {
//				continue;
//			}
//			// Dist from r to tail (candidate dist)
//			__m512i cand_dists_r_t_v = _mm512_mask_i32gather_epi32(INF_v, is_tmp_dist_shorter_m, r_root_id_v, &SI_v_tail.candidates_dists[0], sizeof(weighti));
//			cand_dists_r_t_v = _mm512_mask_and_epi32(INF_v, is_tmp_dist_shorter_m, cand_dists_r_t_v, LOWEST_BYTE_MASK);
//			is_tmp_dist_shorter_m = _mm512_mask_cmplt_epi32_mask(is_tmp_dist_shorter_m, tmp_dist_r_t_v, cand_dists_r_t_v);
//			if (!is_tmp_dist_shorter_m) {
//				continue;
//			}
//			// Mark r_root_id as a v_tail's candidate
//			__mmask16 not_in_cand_que = _mm512_mask_cmpeq_epi32_mask(is_tmp_dist_shorter_m, cand_dists_r_t_v, INF_v);
//			// Add r_root_id into v_tail's candidates_que
//			epi32_mask_unpack_into_queue(
//					SI_v_tail.candidates_que,
//					SI_v_tail.end_candidates_que,
//					not_in_cand_que,
//					r_root_id_v); // r_root_id_v to SI_v_tail.candidates_que
//			epi32_mask_i32scatter_epi8(
//					&SI_v_tail.candidates_dists[0],
//					is_tmp_dist_shorter_m,
//					r_root_id_v,
//					tmp_dist_r_t_v); // tmp_dist_r_t_v to SI_v_tail.candidates_dists
//			// Update v_tail's reached_roots_que
//			__mmask16 not_in_reached_roots_que = _mm512_mask_cmpeq_epi32_mask(is_tmp_dist_shorter_m, table_dists_r_t_v, INF_v);
//			epi32_mask_unpack_into_queue(
//					SI_v_tail.reached_roots_que,
//					SI_v_tail.end_reached_roots_que,
//					not_in_reached_roots_que,
//					r_root_id_v); // r_root_id_v to SI_v_tail.reached_roots_que
//			epi32_mask_i32scatter_epi8(
//					&dists_table[v_tail][0],
//					is_tmp_dist_shorter_m,
//					r_root_id_v,
//					tmp_dist_r_t_v); // tmp_dist_r_t_v to dists_table[v_tail]
//			got_candidates = true;
//		}

		// Sequential Version
		idi bound_r_i = SI_v_head.end_last_new_roots;
		for (idi r_i = 0; r_i < bound_r_i; ++r_i) {
			idi r_root_id = SI_v_head.last_new_roots[r_i]; // last inserted label r_root_id of v_head
			if (v_tail <= r_root_id + roots_start) {
				continue;
			}
			weighti tmp_dist_r_t = SI_v_head.vertices_dists[r_root_id] + weight_h_t;
			if (tmp_dist_r_t < dists_table[v_tail][r_root_id] && tmp_dist_r_t < SI_v_tail.candidates_dists[r_root_id]) {
				// Mark r_root_id as a candidate of v_tail
				if (WEIGHTI_MAX == SI_v_tail.candidates_dists[r_root_id]) {
					// Add r_root_id into v_tail's candidates_que
					SI_v_tail.candidates_que[SI_v_tail.end_candidates_que++] = r_root_id;
				}
				SI_v_tail.candidates_dists[r_root_id] = tmp_dist_r_t;
				if (WEIGHTI_MAX == dists_table[v_tail][r_root_id]) {
					// Add r_root_id into v_tail's reached_roots_que so to reset dists_table[r_root_id][v_tail] at the end of this batch
					SI_v_tail.reached_roots_que[SI_v_tail.end_reached_roots_que++] = r_root_id;
				}
				dists_table[v_tail][r_root_id] = tmp_dist_r_t; // For filtering out longer distances coming later
				got_candidates = true;
			}
		}


		if (got_candidates && !has_candidates[v_tail]) {
			// Put v_tail into has_cand_queue
			has_candidates[v_tail] = true;
			has_cand_queue[end_has_cand_queue++] = v_tail;
		}
	}
	SI_v_head.end_last_new_roots = 0; // clear v_head's last_new_roots
}

//Function: to see if v_id and cand_root_id can have other path already cover the candidate distance
// If there was other path, return the shortest distance. If there was not, return INF
template <inti BATCH_SIZE>
inline weighti WeightedVertexCentricPLL<BATCH_SIZE>::distance_query(
					idi v_id,
					idi cand_root_id,
					//const vector< vector<weighti> > &dists_table,
					const vector< vector<weighti> > &roots_labels_buffer,
					const vector<ShortIndex> &short_index,
					const vector<IndexType> &L,
					idi roots_start,
					weighti tmp_dist_v_c)
{
	//++check_count;
//	distance_query_time -= WallTimer::get_time_mark();
	//static const __m512i INF_v = _mm512_set1_epi32(WEIGHTI_MAX);
	//static const __m512i UNDEF_i32_v = _mm512_undefined_epi32();
	//static const __m512i LOWEST_BYTE_MASK = _mm512_set1_epi32(0xFF);
	//static const __m128i INF_v_128i = _mm_set1_epi8(-1);
	//static const inti NUM_P_INT = 16;
	// Traverse all available hops of v, to see if they reach c
	// 1. Labels in L[v]
	idi cand_real_id = cand_root_id + roots_start;
	const IndexType &Lv = L[v_id];

	// 1. Labels in L[v_id]
//	// Vectorization Version
//	__m512i cand_real_id_v = _mm512_set1_epi32(cand_real_id);
//	__m512i tmp_dist_v_c_v = _mm512_set1_epi32(tmp_dist_v_c);
//	inti remainder_simd = Lv.vertices.size() % NUM_P_INT;
//	idi bound_i_l = Lv.vertices.size() - remainder_simd;
//	for (idi i_l = 0; i_l < bound_i_l; i_l += NUM_P_INT) {
////		simd_util.first += NUM_P_INT;
////		simd_util.second += NUM_P_INT;
////		++simd_full_count.first;
////		++simd_full_count.second;
//		// Labels IDs
//		__m512i r_v = _mm512_loadu_epi32(&Lv.vertices[i_l]);
//		__mmask16 is_r_higher_ranked_m = _mm512_cmplt_epi32_mask(r_v, cand_real_id_v);
//		if (!is_r_higher_ranked_m) {
//			continue;
//		}
//		// Labels dists
//		__m512i dists_c_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_v, &roots_labels_buffer[cand_root_id][0], sizeof(weighti));
//		dists_c_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_c_r_v, LOWEST_BYTE_MASK); // The least significant byte is the weight.
//		__mmask16 is_not_INF_m = _mm512_cmpneq_epi32_mask(dists_c_r_v, INF_v);
//		if (!is_not_INF_m) {
//			continue;
//		}
//
//		// Dist from v to c through label r
//		__mmask16 valid_lanes_m = is_not_INF_m;
//		__m128i tmp_dists_v_r_128i = _mm_mask_loadu_epi8(INF_v_128i, valid_lanes_m, &Lv.distances[i_l]); // Weighti is 8-bit
//		__m512i dists_v_r_v = _mm512_mask_cvtepi8_epi32(INF_v, valid_lanes_m, tmp_dists_v_r_128i); // Convert to 32-bit
//		__m512i label_dist_v_c_v = _mm512_mask_add_epi32(INF_v, valid_lanes_m, dists_c_r_v, dists_v_r_v);
//		__mmask16 is_label_dist_shorter_m = _mm512_mask_cmple_epi32_mask(valid_lanes_m, label_dist_v_c_v, tmp_dist_v_c_v);
//		if (is_label_dist_shorter_m) {
//			// Need to return the shorter distance (might be equal)
//			inti index = (inti) (log2( (double) ((uint16_t) is_label_dist_shorter_m) ) ); // Get the index as the most significant bit which is set as 1. Every "1" is okay actually.
//			distance_query_time += WallTimer::get_time_mark();
//			//++l_l_hit_count;
//			return ((int *) &label_dist_v_c_v)[index]; // Return the distance
//		}
//	}
//	if (remainder_simd) {
////		simd_util.first += remainder_simd;
////		simd_util.second += NUM_P_INT;
////		++simd_full_count.second;
//		__mmask16 in_m = (__mmask16) ((uint16_t) 0xFFFF >> (NUM_P_INT - remainder_simd));
//		// Labels IDs
//		__m512i r_v = _mm512_mask_loadu_epi32(UNDEF_i32_v, in_m, &Lv.vertices[bound_i_l]);
//		__mmask16 is_r_higher_ranked_m = _mm512_mask_cmplt_epi32_mask(in_m, r_v, cand_real_id_v);
//		if (is_r_higher_ranked_m) {
//			// Labels dists
//			__m512i dists_c_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_v, &roots_labels_buffer[cand_root_id][0], sizeof(weighti));
//			dists_c_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_c_r_v, LOWEST_BYTE_MASK); // The least significant byte is the weight.
//			__mmask16 is_not_INF_m = _mm512_cmpneq_epi32_mask(dists_c_r_v, INF_v);
//			if (is_not_INF_m) {
//				// Dist from v to c through label r
//				__mmask16 valid_lanes_m = is_not_INF_m;
//				__m128i tmp_dists_v_r_128i = _mm_mask_loadu_epi8(INF_v_128i, valid_lanes_m, &Lv.distances[bound_i_l]); // Weighti is 8-bit
//				__m512i dists_v_r_v = _mm512_mask_cvtepi8_epi32(INF_v, valid_lanes_m, tmp_dists_v_r_128i); // Convert to 32-bit
//				__m512i label_dist_v_c_v = _mm512_mask_add_epi32(INF_v, valid_lanes_m, dists_c_r_v, dists_v_r_v);
//				__mmask16 is_label_dist_shorter_m = _mm512_mask_cmple_epi32_mask(valid_lanes_m, label_dist_v_c_v, tmp_dist_v_c_v);
//				if (is_label_dist_shorter_m) {
//					// Need to return the shorter distance (might be equal)
//					inti index = (inti) (log2( (double) ((uint16_t) is_label_dist_shorter_m) ) ); // Get the index as the most significant bit which is set as 1. Every "1" is okay actually.
//					distance_query_time += WallTimer::get_time_mark();
//					//++l_l_hit_count;
//					return ((int *) &label_dist_v_c_v)[index]; // Return the distance
//				}
//			}
//
//		}
//	}

	// Sequential Version
	idi bound_i_l = Lv.vertices.size();
	for (idi i_l = 0; i_l < bound_i_l; ++i_l) {
		idi r = Lv.vertices[i_l];
		if (cand_real_id <= r || WEIGHTI_MAX == roots_labels_buffer[cand_root_id][r]) {
			continue;
		}
		weighti label_dist_v_c = Lv.distances[i_l] + roots_labels_buffer[cand_root_id][r];
		if (label_dist_v_c <= tmp_dist_v_c) {
//			distance_query_time += WallTimer::get_time_mark();
			//++l_l_hit_count;
			return label_dist_v_c;
		}
	}
	// 2. Labels in short_index[v_id].vertices_que
	const ShortIndex &SI_v = short_index[v_id];
	const ShortIndex &SI_c = short_index[cand_root_id];

//	const __m512i roots_start_v = _mm512_set1_epi32(roots_start);
//	remainder_simd = SI_v.end_vertices_que % NUM_P_INT;
//	idi bound_i_que = SI_v.end_vertices_que - remainder_simd;
//	for (inti i_que = 0; i_que < bound_i_que; i_que += NUM_P_INT) {
////		simd_util.first += NUM_P_INT;
////		simd_util.second += NUM_P_INT;
////		++simd_full_count.first;
////		++simd_full_count.second;
//		__m512i r_root_id_v = _mm512_loadu_epi32(&SI_v.vertices_que[i_que]);
//		__m512i r_real_id_v = _mm512_add_epi32(r_root_id_v, roots_start_v);
//		__mmask16 is_r_higher_ranked_m = _mm512_cmplt_epi32_mask(r_real_id_v, cand_real_id_v);
//		if (!is_r_higher_ranked_m) {
//			continue;
//		}
//		// Dists from v to r
//		__m512i dists_v_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_v.vertices_dists[0], sizeof(weighti));
//		dists_v_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_v_r_v, LOWEST_BYTE_MASK);
//		// Check r_root_id in cand_root_id's labels inserted in this batch
//		{
//			// Dists from c to r
//			__m512i dists_c_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_c.vertices_dists[0], sizeof(weighti));
//			dists_c_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_c_r_v, LOWEST_BYTE_MASK);
//			__mmask16 is_not_INF_m = _mm512_mask_cmpneq_epi32_mask(is_r_higher_ranked_m, dists_c_r_v, INF_v);
//			if (is_not_INF_m) {
//				// Label dist from v to c
//				__m512i label_dist_v_c_v = _mm512_mask_add_epi32(INF_v, is_not_INF_m, dists_v_r_v, dists_c_r_v);
//				// Compare label dist to tmp dist
//				__mmask16 is_label_dist_shorter_m = _mm512_mask_cmple_epi32_mask(is_not_INF_m, label_dist_v_c_v, tmp_dist_v_c_v);
//				if (is_label_dist_shorter_m) {
//					// Need to return the shorter distance (might be equal)
//					inti index = (inti) (log2( (double) ((uint16_t) is_label_dist_shorter_m) ) ); // Get the index as the most significant bit which is set as 1. Every "1" is okay actually.
//					distance_query_time += WallTimer::get_time_mark();
//					//++vl_cl_hit_count;
//					return ((int *) &label_dist_v_c_v)[index]; // Return the distance
//				}
//			}
//		}
////		// Check r_root_id in cand_root_id's candidates in this iteration
////		{
////			// Dists from c to r
////			__m512i dists_c_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_c.candidates_dists[0], sizeof(weighti));
////			dists_c_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_c_r_v, LOWEST_BYTE_MASK);
////			__mmask16 is_not_INF_m = _mm512_mask_cmpneq_epi32_mask(is_r_higher_ranked_m, dists_c_r_v, INF_v);
////			if (is_not_INF_m) {
////				// Label dist from v to c
////				__m512i label_dist_v_c_v = _mm512_mask_add_epi32(INF_v, is_not_INF_m, dists_v_r_v, dists_c_r_v);
////				// Compare label dist to tmp dist
////				__mmask16 is_label_dist_shorter_m = _mm512_mask_cmple_epi32_mask(is_not_INF_m, label_dist_v_c_v, tmp_dist_v_c_v);
////				if (is_label_dist_shorter_m) {
////					// Need to return the shorter distance (might be equal)
////					inti index = (inti) (log2( (double) ((uint16_t) is_label_dist_shorter_m) ) ); // Get the index as the most significant bit which is set as 1. Every "1" is okay actually.
////					distance_query_time += WallTimer::get_time_mark();
////					//++vl_cc_hit_count;
////					return ((int *) &label_dist_v_c_v)[index]; // Return the distance
////				}
////			}
////		}
//	}
//	if (remainder_simd) {
////		simd_util.first += remainder_simd;
////		simd_util.second += NUM_P_INT;
////		++simd_full_count.second;
//		__mmask16 in_m = (__mmask16) ((uint16_t) 0xFFFF >> (NUM_P_INT - remainder_simd));
//		__m512i r_root_id_v = _mm512_mask_loadu_epi32(UNDEF_i32_v, in_m, &SI_v.vertices_que[bound_i_que]);
//		__m512i r_real_id_v = _mm512_mask_add_epi32(UNDEF_i32_v, in_m, r_root_id_v, roots_start_v);
//		__mmask16 is_r_higher_ranked_m = _mm512_mask_cmplt_epi32_mask(in_m, r_real_id_v, cand_real_id_v);
//		if (is_r_higher_ranked_m) {
//			// Dists from v to r
//			__m512i dists_v_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_v.vertices_dists[0], sizeof(weighti));
//			dists_v_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_v_r_v, LOWEST_BYTE_MASK);
//			// Check r_root_id in cand_root_id's labels inserted in this batch
//			{
//				// Dists from c to r
//				__m512i dists_c_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_c.vertices_dists[0], sizeof(weighti));
//				dists_c_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_c_r_v, LOWEST_BYTE_MASK);
//				__mmask16 is_not_INF_m = _mm512_mask_cmpneq_epi32_mask(is_r_higher_ranked_m, dists_c_r_v, INF_v);
//				if (is_not_INF_m) {
//					// Label dist from v to c
//					__m512i label_dist_v_c_v = _mm512_mask_add_epi32(INF_v, is_not_INF_m, dists_v_r_v, dists_c_r_v);
//					// Compare label dist to tmp dist
//					__mmask16 is_label_dist_shorter_m = _mm512_mask_cmple_epi32_mask(is_not_INF_m, label_dist_v_c_v, tmp_dist_v_c_v);
//					if (is_label_dist_shorter_m) {
//						// Need to return the shorter distance (might be equal)
//						inti index = (inti) (log2( (double) ((uint16_t) is_label_dist_shorter_m) ) ); // Get the index as the most significant bit which is set as 1. Every "1" is okay actually.
//						distance_query_time += WallTimer::get_time_mark();
//						//++vl_cl_hit_count;
//						return ((int *) &label_dist_v_c_v)[index]; // Return the distance
//					}
//				}
//			}
////			// Check r_root_id in cand_root_id's candidates in this iteration
////			{
////				// Dists from c to r
////				__m512i dists_c_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_c.candidates_dists[0], sizeof(weighti));
////				dists_c_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_c_r_v, LOWEST_BYTE_MASK);
////				__mmask16 is_not_INF_m = _mm512_mask_cmpneq_epi32_mask(is_r_higher_ranked_m, dists_c_r_v, INF_v);
////				if (is_not_INF_m) {
////					// Label dist from v to c
////					__m512i label_dist_v_c_v = _mm512_mask_add_epi32(INF_v, is_not_INF_m, dists_v_r_v, dists_c_r_v);
////					// Compare label dist to tmp dist
////					__mmask16 is_label_dist_shorter_m = _mm512_mask_cmple_epi32_mask(is_not_INF_m, label_dist_v_c_v, tmp_dist_v_c_v);
////					if (is_label_dist_shorter_m) {
////						// Need to return the shorter distance (might be equal)
////						inti index = (inti) (log2( (double) ((uint16_t) is_label_dist_shorter_m) ) ); // Get the index as the most significant bit which is set as 1. Every "1" is okay actually.
////						distance_query_time += WallTimer::get_time_mark();
////						//++vl_cc_hit_count;
////						return ((int *) &label_dist_v_c_v)[index]; // Return the distance
////					}
////				}
////			}
//		}
//	}

	// Sequential Version
	inti bound_i_que = SI_v.end_vertices_que;
//	_mm_prefetch(&SI_v.vertices_dists[0], _MM_HINT_T0);
//	_mm_prefetch(&SI_c.vertices_dists[0], _MM_HINT_T0);
//	_mm_prefetch(&SI_c.candidates_dists[0], _MM_HINT_T0);
	for (inti i_que = 0; i_que < bound_i_que; ++i_que) {
		idi r_root_id = SI_v.vertices_que[i_que];
		if (cand_real_id <= r_root_id + roots_start) {
			continue;
		}
		// Check r_root_id in cand_root_id's labels inserted in this batch
		if (WEIGHTI_MAX != SI_c.vertices_dists[r_root_id]) {
			weighti label_dist_v_c = SI_v.vertices_dists[r_root_id] + SI_c.vertices_dists[r_root_id];
			if (label_dist_v_c <= tmp_dist_v_c) {
//				distance_query_time += WallTimer::get_time_mark();
				//++vl_cl_hit_count;
				return label_dist_v_c;
			}
		}
//		// Check r_root_id in cand_root_id's candidates in this iteration
//		if (WEIGHTI_MAX != SI_c.candidates_dists[r_root_id]) {
//			weighti label_dist_v_c = SI_v.vertices_dists[r_root_id] + SI_c.candidates_dists[r_root_id];
//			if (label_dist_v_c <= tmp_dist_v_c) {
//				distance_query_time += WallTimer::get_time_mark();
//				//++vl_cc_hit_count;
//				return label_dist_v_c;
//			}
//		}
	}

//	// 3. Labels in short_index[v_id].candidates_que
//	remainder_simd = SI_v.end_candidates_que % NUM_P_INT;
//	bound_i_que = SI_v.end_candidates_que - remainder_simd;
//	for (inti i_que = 0; i_que < bound_i_que; i_que += NUM_P_INT) {
//		__m512i r_root_id_v = _mm512_loadu_epi32(&SI_v.candidates_que[i_que]);
//		__m512i r_real_id_v = _mm512_add_epi32(r_root_id_v, roots_start_v);
//		__mmask16 is_r_higher_ranked_m = _mm512_cmplt_epi32_mask(r_real_id_v, cand_real_id_v);
//		if (!is_r_higher_ranked_m) {
//			continue;
//		}
//		// Dists from v to r
//		__m512i dists_v_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_v.candidates_dists[0], sizeof(weighti));
//		dists_v_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_v_r_v, LOWEST_BYTE_MASK);
//		// Check r_root_id in cand_root_id's labels inserted in this batch
//		{
//			// Dists from c to r
//			__m512i dists_c_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_c.vertices_dists[0], sizeof(weighti));
//			dists_c_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_c_r_v, LOWEST_BYTE_MASK);
//			__mmask16 is_not_INF_m = _mm512_mask_cmpneq_epi32_mask(is_r_higher_ranked_m, dists_c_r_v, INF_v);
//			if (is_not_INF_m) {
//				// Label dist from v to c
//				__m512i label_dist_v_c_v = _mm512_mask_add_epi32(INF_v, is_not_INF_m, dists_v_r_v, dists_c_r_v);
//				// Compare label dist to tmp dist
//				__mmask16 is_label_dist_shorter_m = _mm512_mask_cmple_epi32_mask(is_not_INF_m, label_dist_v_c_v, tmp_dist_v_c_v);
//				if (is_label_dist_shorter_m) {
//					// Need to return the shorter distance (might be equal)
//					inti index = (inti) (log2( (double) ((uint16_t) is_label_dist_shorter_m) ) ); // Get the index as the most significant bit which is set as 1. Every "1" is okay actually.
//					distance_query_time += WallTimer::get_time_mark();
//					//++vc_cl_hit_count;
//					return ((int *) &label_dist_v_c_v)[index]; // Return the distance
//				}
//			}
//		}
//		// Check r_root_id in cand_root_id's candidates in this iteration
//		{
//			// Dists from c to r
//			__m512i dists_c_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_c.candidates_dists[0], sizeof(weighti));
//			dists_c_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_c_r_v, LOWEST_BYTE_MASK);
//			__mmask16 is_not_INF_m = _mm512_mask_cmpneq_epi32_mask(is_r_higher_ranked_m, dists_c_r_v, INF_v);
//			if (is_not_INF_m) {
//				// Label dist from v to c
//				__m512i label_dist_v_c_v = _mm512_mask_add_epi32(INF_v, is_not_INF_m, dists_v_r_v, dists_c_r_v);
//				// Compare label dist to tmp dist
//				__mmask16 is_label_dist_shorter_m = _mm512_mask_cmple_epi32_mask(is_not_INF_m, label_dist_v_c_v, tmp_dist_v_c_v);
//				if (is_label_dist_shorter_m) {
//					// Need to return the shorter distance (might be equal)
//					inti index = (inti) (log2( (double) ((uint16_t) is_label_dist_shorter_m) ) ); // Get the index as the most significant bit which is set as 1. Every "1" is okay actually.
//					distance_query_time += WallTimer::get_time_mark();
//					//++vc_cc_hit_count;
//					return ((int *) &label_dist_v_c_v)[index]; // Return the distance
//				}
//			}
//		}
//	}
//	if (remainder_simd) {
//		__mmask16 in_m = (__mmask16) ((uint16_t) 0xFFFF >> (NUM_P_INT - remainder_simd));
//		__m512i r_root_id_v = _mm512_mask_loadu_epi32(UNDEF_i32_v, in_m, &SI_v.candidates_que[bound_i_que]);
//		__m512i r_real_id_v = _mm512_mask_add_epi32(UNDEF_i32_v, in_m, r_root_id_v, roots_start_v);
//		__mmask16 is_r_higher_ranked_m = _mm512_mask_cmplt_epi32_mask(in_m, r_real_id_v, cand_real_id_v);
//		if (is_r_higher_ranked_m) {
//			// Dists from v to r
//			__m512i dists_v_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_v.candidates_dists[0], sizeof(weighti));
//			dists_v_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_v_r_v, LOWEST_BYTE_MASK);
//			// Check r_root_id in cand_root_id's labels inserted in this batch
//			{
//				// Dists from c to r
//				__m512i dists_c_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_c.vertices_dists[0], sizeof(weighti));
//				dists_c_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_c_r_v, LOWEST_BYTE_MASK);
//				__mmask16 is_not_INF_m = _mm512_mask_cmpneq_epi32_mask(is_r_higher_ranked_m, dists_c_r_v, INF_v);
//				if (is_not_INF_m) {
//					// Label dist from v to c
//					__m512i label_dist_v_c_v = _mm512_mask_add_epi32(INF_v, is_not_INF_m, dists_v_r_v, dists_c_r_v);
//					// Compare label dist to tmp dist
//					__mmask16 is_label_dist_shorter_m = _mm512_mask_cmple_epi32_mask(is_not_INF_m, label_dist_v_c_v, tmp_dist_v_c_v);
//					if (is_label_dist_shorter_m) {
//						// Need to return the shorter distance (might be equal)
//						inti index = (inti) (log2( (double) ((uint16_t) is_label_dist_shorter_m) ) ); // Get the index as the most significant bit which is set as 1. Every "1" is okay actually.
//						distance_query_time += WallTimer::get_time_mark();
//						//++vc_cl_hit_count;
//						return ((int *) &label_dist_v_c_v)[index]; // Return the distance
//					}
//				}
//			}
//			// Check r_root_id in cand_root_id's candidates in this iteration
//			{
//				// Dists from c to r
//				__m512i dists_c_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_c.candidates_dists[0], sizeof(weighti));
//				dists_c_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_c_r_v, LOWEST_BYTE_MASK);
//				__mmask16 is_not_INF_m = _mm512_mask_cmpneq_epi32_mask(is_r_higher_ranked_m, dists_c_r_v, INF_v);
//				if (is_not_INF_m) {
//					// Label dist from v to c
//					__m512i label_dist_v_c_v = _mm512_mask_add_epi32(INF_v, is_not_INF_m, dists_v_r_v, dists_c_r_v);
//					// Compare label dist to tmp dist
//					__mmask16 is_label_dist_shorter_m = _mm512_mask_cmple_epi32_mask(is_not_INF_m, label_dist_v_c_v, tmp_dist_v_c_v);
//					if (is_label_dist_shorter_m) {
//						// Need to return the shorter distance (might be equal)
//						inti index = (inti) (log2( (double) ((uint16_t) is_label_dist_shorter_m) ) ); // Get the index as the most significant bit which is set as 1. Every "1" is okay actually.
//						distance_query_time += WallTimer::get_time_mark();
//						//++vc_cc_hit_count;
//						return ((int *) &label_dist_v_c_v)[index]; // Return the distance
//					}
//				}
//			}
//		}
//	}

	// Sequential Version
//	bound_i_que = SI_v.end_candidates_que;
////	_mm_prefetch(&SI_v.candidates_dists[0], _MM_HINT_T0);
////	_mm_prefetch(&SI_c.vertices_dists[0], _MM_HINT_T0);
////	_mm_prefetch(&SI_c.candidates_dists[0], _MM_HINT_T0);
//	for (inti i_que = 0; i_que < bound_i_que; ++i_que) {
//		idi r_root_id = SI_v.candidates_que[i_que];
//		if (cand_real_id <= r_root_id + roots_start) {
//			continue;
//		}
//		// Check r_root_id in cand_root_id's labels inserted in this batch
//		if (WEIGHTI_MAX != SI_c.vertices_dists[r_root_id]) {
//			weighti label_dist_v_c = SI_v.candidates_dists[r_root_id] + SI_c.vertices_dists[r_root_id];
//			if (label_dist_v_c <= tmp_dist_v_c) {
//				distance_query_time += WallTimer::get_time_mark();
//				++vc_cl_hit_count;
//				return label_dist_v_c;
//			}
//		}
//		// Check r_root_id in cand_root_id's candidates in this iteration
//		if (WEIGHTI_MAX != SI_c.candidates_dists[r_root_id]) {
//			weighti label_dist_v_c = SI_v.candidates_dists[r_root_id] + SI_c.candidates_dists[r_root_id];
//			if (label_dist_v_c <= tmp_dist_v_c) {
//				distance_query_time += WallTimer::get_time_mark();
//				++vc_cc_hit_count;
//				return label_dist_v_c;
//			}
//		}
//	}

//	distance_query_time += WallTimer::get_time_mark();
	return WEIGHTI_MAX;
}

// Function: reset distance table dists_table
template <inti BATCH_SIZE>
inline void WeightedVertexCentricPLL<BATCH_SIZE>::reset_tables(
		vector<ShortIndex> &short_index,
		idi roots_start,
		inti roots_size,
		const vector<IndexType> &L,
		vector< vector<weighti> > &dists_table,
		vector< vector<weighti> > &roots_labels_buffer,
		const vector<idi> &has_new_labels_queue,
		idi end_has_new_labels_queue)
{
	// Reset roots_labels_buffer according to L (old labels)
	for (idi r_roots_id = 0; r_roots_id < roots_size; ++r_roots_id) {
		idi r_real_id = r_roots_id + roots_start;
		// Traverse labels of r
		const IndexType &Lr = L[r_real_id];
		idi bound_i_l = Lr.vertices.size();
		for (idi i_l = 0; i_l < bound_i_l; ++i_l) {
			roots_labels_buffer[r_roots_id][Lr.vertices[i_l]] = WEIGHTI_MAX;
		}
		roots_labels_buffer[r_roots_id][r_real_id] = WEIGHTI_MAX;
	}

	// Reset dists_table according to short_index[v].reached_roots_que
	for (idi i_q = 0; i_q < end_has_new_labels_queue; ++i_q) {
		idi v_id = has_new_labels_queue[i_q];
		ShortIndex &SI_v = short_index[v_id];
		// Traverse roots which have reached v_id
		inti bound_i_r = SI_v.end_reached_roots_que;
		for (inti i_r = 0; i_r < bound_i_r; ++i_r) {
			dists_table[v_id][SI_v.reached_roots_que[i_r]] = WEIGHTI_MAX;
		}
		SI_v.end_reached_roots_que = 0; // Clear v_id's reached_roots_que
	}
}

// Function: after finishing the label tables in the short_index, build the index according to it.
// And also reset the has_new_labels_queue
template <inti BATCH_SIZE>
inline void WeightedVertexCentricPLL<BATCH_SIZE>::update_index(
		vector<IndexType> &L,
		vector<ShortIndex> &short_index,
		idi roots_start,
		vector<idi> &has_new_labels_queue,
		idi &end_has_new_labels_queue,
		vector<bool> &has_new_labels)
{
	// Traverse has_new_labels_queue for every vertex who got new labels in this batch
	for (idi i_q = 0; i_q < end_has_new_labels_queue; ++i_q) {
		idi v_id = has_new_labels_queue[i_q];
		has_new_labels[v_id] = false; // Reset has_new_labels
		IndexType &Lv = L[v_id];
		ShortIndex &SI_v = short_index[v_id];
		inti bound_i_r = SI_v.end_vertices_que;
		for (inti i_r = 0; i_r < bound_i_r; ++i_r) {
			idi r_root_id = SI_v.vertices_que[i_r];
			weighti dist = SI_v.vertices_dists[r_root_id];
			if (WEIGHTI_MAX == dist) {
				continue;
			}
			SI_v.vertices_dists[r_root_id] = WEIGHTI_MAX; // Reset v_id's vertices_dists
			Lv.vertices.push_back(r_root_id + roots_start);
			Lv.distances.push_back(dist);
		}
		SI_v.end_vertices_que = 0; // Clear v_id's vertices_que
	}
	end_has_new_labels_queue = 0; // Clear has_new_labels_queue
}

// Function:vertex s noticed that it received a distance between t but the distance is longer than what s can get,
// so s starts to send its shorter distance between s and t to all its neighbors. According to the distance sent,
// all active neighbors will update their distance table elements and reset their label table elements. The 
// process continue untill no active vertices.
template <inti BATCH_SIZE>
inline void WeightedVertexCentricPLL<BATCH_SIZE>::send_back(
		idi s,
		idi r_root_id,
		const WeightedGraph &G,
		vector< vector<weighti> > &dists_table,
		vector<ShortIndex> &short_index,
		idi roots_start)
{
	idi r_real_id = r_root_id + roots_start;
	static idi num_v = G.get_num_v();
	// Active queue
	static vector<idi> active_queue(num_v);
	static idi end_active_queue = 0;
	static vector<bool> is_active(num_v, false);
	// Temporary Active queue
	static vector<idi> tmp_active_queue(num_v);
	static idi end_tmp_active_queue = 0;
	static vector<bool> tmp_is_active(num_v, false);

	active_queue[end_active_queue++] = s;
	while (0 != end_active_queue) {
		// Traverse active queue, get every vertex and its distance to the target
		for (idi i_q = 0; i_q < end_active_queue; ++i_q) {
			idi v = active_queue[i_q];
			is_active[v] = false; // reset flag
			weighti dist_r_v = dists_table[r_root_id][v];
			// Traverse all neighbors of vertex v
			idi e_i_start = G.vertices[v];
			idi e_i_bound = e_i_start + G.out_degrees[v];
			for (idi e_i = e_i_start; e_i < e_i_bound; ++e_i) {
				idi w = G.out_edges[e_i];
				if (w <= r_real_id) {
					// Neighbors are ordered by ranks from low to high
					break;
				}
				weighti tmp_dist_r_w = dist_r_v + G.out_weights[e_i];
				if (tmp_dist_r_w <= dists_table[r_root_id][w]) {
					dists_table[r_root_id][w] = tmp_dist_r_w;
					short_index[w].vertices_dists[r_root_id] = WEIGHTI_MAX;
					if (!tmp_is_active[w]) {
						tmp_is_active[w] = true;
						tmp_active_queue[end_tmp_active_queue++] = w;
					}
				}
			}
		}
		end_active_queue = end_tmp_active_queue;
		end_tmp_active_queue = 0;
		active_queue.swap(tmp_active_queue);
		is_active.swap(tmp_is_active);
	}
}

//// Function: at the end of every batch, filter out wrong and redundant labels
//inline void WeightedVertexCentricPLL::filter_out_labels(
//		const vector<idi> &has_new_labels_queue,
//		idi end_has_new_labels_queue,
//		vector<ShortIndex> &short_index,
//		idi roots_start)
//{
//	// Traverse has_new_labels_queue for every vertex who got new labels in this batch
//	for (idi i_q = 0; i_q < end_has_new_labels_queue; ++i_q) {
//		idi v_id = has_new_labels_queue[i_q];
//		IndexType &Lv = L[v_id];
//		ShortIndex &SI_v = short_index[v_id];
//		inti bound_label_i = SI_v.end_vertices_que;
//		// Traverse v_id's every new label
//		for (inti i_c = 0; i_c < bound_label_i; ++i_c) {
//			idi c_root_id = SI_v.vertices_que[i_c];
//			weighti dist_v_c = SI_v.vertices_dists[c_root_id];
//			if (WEIGHTI_MAX == dist_v_c) {
//				continue;
//			}
//			ShortIndex &SI_c = short_index[c_root_id + roots_start];
//			// Traverse v_id's other new label to see if has shorter distance
//			for (inti i_r = 0; i_r < bound_label_i; ++i_r) {
////				if (i_r == i_c) {
////					continue;
////				}
//				// r_root_id is the middle hop.
//				idi r_root_id = SI_v.vertices_que[i_r];
//				if (c_root_id <= r_root_id) {
//					continue;
//				}
//				weighti dist_v_r = SI_v.vertices_dists[r_root_id];
//				if (WEIGHTI_MAX == dist_v_r) {
//					continue;
//				}
//				weighti dist_c_r = SI_c.vertices_dists[r_root_id];
//				if (WEIGHTI_MAX == dist_c_r) {
//					continue;
//				}
//				if (dist_v_r + dist_c_r <= dist_v_c) {
//					// Shorter distance exists, then label c is not necessary
//					SI_v.vertices_dists[c_root_id] = WEIGHTI_MAX; // Filter out c_root_id from v_id's labels
//				}
//			}
//		}
//	}
//}

// Function: at the end of every batch, filter out wrong and redundant labels
template <inti BATCH_SIZE>
inline void WeightedVertexCentricPLL<BATCH_SIZE>::filter_out_labels(
		const vector<idi> &has_new_labels_queue,
		idi end_has_new_labels_queue,
		vector<ShortIndex> &short_index,
		idi roots_start)
{
	// Traverse has_new_labels_queue for every vertex who got new labels in this batch
	for (idi i_q = 0; i_q < end_has_new_labels_queue; ++i_q) {
		idi v_id = has_new_labels_queue[i_q];
		IndexType &Lv = L[v_id];
		ShortIndex &SI_v = short_index[v_id];

		// Monotone check
//		++checked_count;
		if (SI_v.are_dists_monotone) {
//			++monotone_count;
			SI_v.max_dist_batch = 0;
			continue;
		} else {
			SI_v.are_dists_monotone = true;
			SI_v.max_dist_batch = 0;
		}

		inti bound_label_i = SI_v.end_vertices_que;
		// Traverse v_id's every new label
		for (inti i_c = 0; i_c < bound_label_i; ++i_c) {
			idi c_root_id = SI_v.vertices_que[i_c];
			weighti dist_v_c = SI_v.vertices_dists[c_root_id];
//			if (WEIGHTI_MAX == dist_v_c) {
//				continue;
//			}
			ShortIndex &SI_c = short_index[c_root_id + roots_start];

//			// Vectorized Version
//			//static inti NUM_P_INT = 16;
//			//static const __m512i INF_v = _mm512_set1_epi32(WEIGHTI_MAX);
//			//static const __m512i UNDEF_i32_v = _mm512_undefined_epi32();
//			//static const __m512i LOWEST_BYTE_MASK = _mm512_set1_epi32(0xFF);
//			//static const __m128i INF_v_128i = _mm_set1_epi8(-1);
//			const __m512i c_root_id_v = _mm512_set1_epi32(c_root_id);
//			const __m512i dist_v_c_v = _mm512_set1_epi32(dist_v_c);
//			inti remainder_simd = bound_label_i % NUM_P_INT;
//			idi bound_i_r = bound_label_i - remainder_simd;
//			for (idi i_r = 0; i_r < bound_i_r; i_r += NUM_P_INT) {
//				// Other label r
//				__m512i r_root_id_v = _mm512_loadu_epi32(&SI_v.vertices_que[i_r]);
//				__mmask16 is_r_higher_ranked_m = _mm512_cmplt_epi32_mask(r_root_id_v, c_root_id_v);
//				if (!is_r_higher_ranked_m) {
//					continue;
//				}
//				// Distance v to r
//				__m512i dists_v_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_v.vertices_dists[0], sizeof(weighti));
//				dists_v_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_v_r_v, LOWEST_BYTE_MASK);
//				__mmask16 is_shorter_m = _mm512_cmple_epi32_mask(dists_v_r_v, dist_v_c_v);
//				if (!is_shorter_m) {
//					continue;
//				}
//				// Distance c to r
//				__m512i dists_c_r_v = _mm512_mask_i32gather_epi32(INF_v, is_shorter_m, r_root_id_v, &SI_c.vertices_dists[0], sizeof(weighti));
//				dists_c_r_v = _mm512_mask_and_epi32(INF_v, is_shorter_m, dists_c_r_v, LOWEST_BYTE_MASK);
//				is_shorter_m = _mm512_cmple_epi32_mask(dists_c_r_v, dist_v_c_v);
//				if (!is_shorter_m) {
//					continue;
//				}
//				// Distance compare
//				is_shorter_m = _mm512_mask_cmple_epi32_mask(is_shorter_m, _mm512_mask_add_epi32(INF_v, is_shorter_m, dists_v_r_v, dists_c_r_v), dist_v_c_v);
//				if (is_shorter_m) {
//					int_mask_i32scatter_epi8(
//							&SI_v.vertices_dists[0],
//							is_shorter_m,
//							c_root_id_v,
//							WEIGHTI_MAX);
//				}
//			}
//			if (remainder_simd) {
//				__mmask16 in_m = (__mmask16) ((uint16_t) 0xFFFF >> (NUM_P_INT - remainder_simd));
//				// Other label r
//				__m512i r_root_id_v = _mm512_mask_loadu_epi32(UNDEF_i32_v, in_m, &SI_v.vertices_que[bound_i_r]);
//
//				__mmask16 is_r_higher_ranked_m = _mm512_mask_cmplt_epi32_mask(in_m, r_root_id_v, c_root_id_v);
//				if (!is_r_higher_ranked_m) {
//					continue;
//				}
//				// Distance v to r
//				__m512i dists_v_r_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, r_root_id_v, &SI_v.vertices_dists[0], sizeof(weighti));
//				dists_v_r_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_v_r_v, LOWEST_BYTE_MASK);
//				__mmask16 is_shorter_m = _mm512_cmple_epi32_mask(dists_v_r_v, dist_v_c_v);
//				if (!is_shorter_m) {
//					continue;
//				}
//				// Distance c to r
//				__m512i dists_c_r_v = _mm512_mask_i32gather_epi32(INF_v, is_shorter_m, r_root_id_v, &SI_c.vertices_dists[0], sizeof(weighti));
//				dists_c_r_v = _mm512_mask_and_epi32(INF_v, is_shorter_m, dists_c_r_v, LOWEST_BYTE_MASK);
//				is_shorter_m = _mm512_cmple_epi32_mask(dists_c_r_v, dist_v_c_v);
//				if (!is_shorter_m) {
//					continue;
//				}
//				// Distance compare
//				is_shorter_m = _mm512_mask_cmple_epi32_mask(is_shorter_m, _mm512_mask_add_epi32(INF_v, is_shorter_m, dists_v_r_v, dists_c_r_v), dist_v_c_v);
//				if (is_shorter_m) {
//					int_mask_i32scatter_epi8(
//							&SI_v.vertices_dists[0],
//							is_shorter_m,
//							c_root_id_v,
//							WEIGHTI_MAX);
//				}
//			}

			// Sequential Version
			// Traverse v_id's other new label to see if has shorter distance
			for (inti i_r = 0; i_r < bound_label_i; ++i_r) {
			//for (inti i_r = i_c + 1; i_r < bound_label_i; ++i_r) {}
				// r_root_id is the middle hop.
				idi r_root_id = SI_v.vertices_que[i_r];
				if (c_root_id <= r_root_id) {
					continue;
				}
				weighti dist_v_r = SI_v.vertices_dists[r_root_id];
				if (dist_v_r > dist_v_c) {
					continue;
				}
				weighti dist_c_r = SI_c.vertices_dists[r_root_id];
				if (dist_c_r > dist_v_c) {
					continue;
				}
				if (dist_v_r + dist_c_r <= dist_v_c) {
					// Shorter distance exists, then label c is not necessary
					SI_v.vertices_dists[c_root_id] = WEIGHTI_MAX; // Filter out c_root_id from v_id's labels
				}
			}
		}
	}
}
template <inti BATCH_SIZE>
inline void WeightedVertexCentricPLL<BATCH_SIZE>::vertex_centric_labeling_in_batches(
						const WeightedGraph &G,
						idi b_id,
						idi roots_start, // start id of roots
						inti roots_size, // how many roots in the batch
						vector<IndexType> &L,
						vector<idi> &active_queue,
						idi end_active_queue,
						vector<bool> &is_active,
						vector<idi> &has_cand_queue,
						idi end_has_cand_queue,
						vector<bool> &has_candidates,
						vector< vector<weighti> > &dists_table,
						vector< vector<weighti> > &roots_labels_buffer,
						vector<ShortIndex> &short_index,
						vector<idi> &has_new_labels_queue,
						idi end_has_new_labels_queue,
						vector<bool> &has_new_labels)
{
//	initializing_time -= WallTimer::get_time_mark();

	// At the beginning of a batch, initialize the labels L and distance buffer dist_matrix;
	initialize_tables(
			short_index,
			roots_labels_buffer,
			active_queue,
			end_active_queue,
			roots_start,
			roots_size,
			L,
			has_new_labels_queue,
			end_has_new_labels_queue,
			has_new_labels);

//	weighti iter = 0; // The iterator, also the distance for current iteration
//	initializing_time += WallTimer::get_time_mark();


	while (0 != end_active_queue) {
		// First stage, sending distances.
//		candidating_time -= WallTimer::get_time_mark();
//		candidating_ins_count.measure_start();
		// Traverse the active queue, every active vertex sends distances to its neighbors
		for (idi i_queue = 0; i_queue < end_active_queue; ++i_queue) {
			idi v_head = active_queue[i_queue];
			is_active[v_head] = false; // reset is_active

			send_messages(
					v_head,
					roots_start,
					G,
					dists_table,
					short_index,
					has_cand_queue,
					end_has_cand_queue,
					has_candidates);
		}
		end_active_queue = 0; // Set the active_queue empty
//		candidating_ins_count.measure_stop();
//		candidating_time += WallTimer::get_time_mark();
//		adding_time -= WallTimer::get_time_mark();
//		adding_ins_count.measure_start();

		// Traverse vertices in the has_cand_queue to insert labels
		for (idi i_queue = 0; i_queue < end_has_cand_queue; ++i_queue) {
			idi v_id = has_cand_queue[i_queue];
			bool need_activate = false;
			ShortIndex &SI_v = short_index[v_id];
			// Traverse v_id's all candidates
			inti bound_cand_i = SI_v.end_candidates_que;
			for (inti cand_i = 0; cand_i < bound_cand_i; ++cand_i) {
				inti cand_root_id = SI_v.candidates_que[cand_i];
				weighti tmp_dist_v_c = SI_v.candidates_dists[cand_root_id];
				// Distance check for pruning
				weighti query_dist_v_c;
				if (WEIGHTI_MAX == 
						(query_dist_v_c = distance_query(
										 v_id,
										 cand_root_id,
										 //dists_table,
										 roots_labels_buffer,
										 short_index,
										 L,
										 roots_start,
										 tmp_dist_v_c))) {
					if (WEIGHTI_MAX == SI_v.vertices_dists[cand_root_id]) {
						// Record cand_root_id as v_id's label
						SI_v.vertices_que[SI_v.end_vertices_que++] = cand_root_id;
						
					}
					// Monotone check
					if (SI_v.are_dists_monotone) {
						if (tmp_dist_v_c > SI_v.max_dist_batch) {
							SI_v.max_dist_batch = tmp_dist_v_c;
						} else {
							SI_v.are_dists_monotone = false;
						}
					}
					// Record the new distance in the label table
					SI_v.vertices_dists[cand_root_id] = tmp_dist_v_c;
					SI_v.last_new_roots[SI_v.end_last_new_roots++] = cand_root_id;
					need_activate = true;
				} else if (query_dist_v_c < tmp_dist_v_c){
					// Update the dists_table
					dists_table[v_id][cand_root_id] = query_dist_v_c;
					// Need to send back the distance
					// DEPRECATED! The correction should not be done here, because some shorter distance does not mean wrong label distances.
//					send_back(
//							v_id,
//							cand_root_id,
//							G,
//							dists_table,
//							short_index,
//							roots_start);
				}
			}
			if (need_activate) {
				if (!is_active[v_id]) {
					is_active[v_id] = true;
					active_queue[end_active_queue++] = v_id;
				}
				if (!has_new_labels[v_id]) {
					has_new_labels[v_id] = true;
					has_new_labels_queue[end_has_new_labels_queue++] = v_id;
				}
			}
		}
		// Reset vertices' candidates_que and candidates_dists
		for (idi i_queue = 0; i_queue < end_has_cand_queue; ++i_queue) {
			idi v_id = has_cand_queue[i_queue];
			has_candidates[v_id] = false; // reset has_candidates
			ShortIndex &SI_v = short_index[v_id];

//			// Semi-Vectorized Version
//			inti remainder_simd = SI_v.end_candidates_que % NUM_P_INT;
//			inti bound_cand_i = SI_v.end_candidates_que - remainder_simd;
//			for (inti cand_i = 0; cand_i < bound_cand_i; cand_i += NUM_P_INT) {
//				__m512i cand_root_id_v = _mm512_loadu_epi32(&SI_v.candidates_que[cand_i]);
//				int_i32scatter_epi8(
//						&SI_v.candidates_dists[0],
//						cand_root_id_v,
//						WEIGHTI_MAX);
//			}
//			if (remainder_simd) {
//				__mmask16 in_m = (__mmask16) ((uint16_t) 0xFFFF >> (NUM_P_INT - remainder_simd));
//				__m512i cand_root_id_v = _mm512_mask_loadu_epi32(UNDEF_i32_v, in_m, &SI_v.candidates_que[bound_cand_i]);
//				int_mask_i32scatter_epi8(
//						&SI_v.candidates_dists[0],
//						in_m,
//						cand_root_id_v,
//						WEIGHTI_MAX);
//			}

			// Sequential Version
			inti bound_cand_i = SI_v.end_candidates_que;
			for (inti cand_i = 0; cand_i < bound_cand_i; ++cand_i) {
				inti cand_root_id = SI_v.candidates_que[cand_i];
				SI_v.candidates_dists[cand_root_id] = WEIGHTI_MAX; // Reset candidates_dists after using in distance_query.
			}
			SI_v.end_candidates_que = 0; // Clear v_id's candidates_que
		}
		end_has_cand_queue = 0; // Set the has_cand_queue empty
//		adding_ins_count.measure_stop();
//		adding_time += WallTimer::get_time_mark();
	}

	// Filter out wrong and redundant labels in this batch
//	correction_time -= WallTimer::get_time_mark();
	filter_out_labels(
			has_new_labels_queue,
			end_has_new_labels_queue,
			short_index,
			roots_start);
//	correction_time += WallTimer::get_time_mark();

	// Reset dists_table and short_index
//	initializing_time -= WallTimer::get_time_mark();
	reset_tables(
			short_index,
			roots_start,
			roots_size,
			L,
			dists_table,
			roots_labels_buffer,
			has_new_labels_queue,
			end_has_new_labels_queue);

	// Update the index according to labels_table
	update_index(
			L,
			short_index,
			roots_start,
			has_new_labels_queue,
			end_has_new_labels_queue,
			has_new_labels);

	// Reset the dist_matrix
//	init_dist_matrix_time -= WallTimer::get_time_mark();
//	reset_at_end(
//			roots_start,
//			roots_size,
//			L,
//			dist_matrix);
//	init_dist_matrix_time += WallTimer::get_time_mark();
//	initializing_time += WallTimer::get_time_mark();


//	double total_time = time_can + time_add;
//	printf("Candidating time: %f (%f%%)\n", time_can, time_can / total_time * 100);
//	printf("Adding time: %f (%f%%)\n", time_add, time_add / total_time * 100);
}


template <inti BATCH_SIZE>
void WeightedVertexCentricPLL<BATCH_SIZE>::construct(const WeightedGraph &G)
{
	idi num_v = G.get_num_v();
	num_v_ = num_v;
	L.resize(num_v);
	idi remainer = num_v % BATCH_SIZE;
	idi b_i_bound = num_v / BATCH_SIZE;
//	cache_miss.measure_start();
	// Active queue
	static vector<idi> active_queue(num_v);
	static idi end_active_queue = 0;
	static vector<bool> is_active(num_v, false);// is_active[v] is true means vertex v is in the active queue.
	// Queue of vertices having candidates
	static vector<idi> has_cand_queue(num_v);
	static idi end_has_cand_queue = 0;
	static vector<bool> has_candidates(num_v, false); // has_candidates[v] is true means vertex v is in the queue has_cand_queue.
	// Distance table of shortest distance from roots to other vertices.
	static vector< vector<weighti> > dists_table(num_v, vector<weighti>(BATCH_SIZE, WEIGHTI_MAX)); 
		// The distance table is roots_sizes by N. 
		// 1. record the shortest distance so far from every root to every vertex; (from v to root r)
		// 2. (DEPRECATED! Replaced by roots_labels_buffer) The distance buffer, recording old label distances of every root. It needs to be initialized every batch by labels of roots. (dists_table[r][l], r > l)
	// A tabel for roots' old label distances (which are stored in the dists_table in the last version)
	static vector< vector<weighti> > roots_labels_buffer(BATCH_SIZE, vector<weighti>(num_v, WEIGHTI_MAX)); // r's old label distance from r to v
	// Every vertex has a ShortIndex object; the previous labels_table is now in ShortIndex structure
	static vector<ShortIndex> short_index(num_v, ShortIndex(BATCH_SIZE)); // Here the size of short_index actually is fixed because it is static.
		// Temporary distance table, recording in the current iteration the traversing distance from a vertex to a root.
		// The candidate table is replaced by the ShortIndex structure: every vertex has a queue and a distance array;
   		// 1. the queue records last inserted labels.
		// 2. the distance array acts like a bitmap but restores distances.

	// A queue to store vertices which have got new labels in this batch. This queue is used for reset dists_table.
	static vector<idi> has_new_labels_queue(num_v);
	static idi end_has_new_labels_queue = 0;
	static vector<bool> has_new_labels(num_v, false);

	double time_labeling = -WallTimer::get_time_mark();

	for (idi b_i = 0; b_i < b_i_bound; ++b_i) {
		vertex_centric_labeling_in_batches(
				G,
				b_i,
				b_i * BATCH_SIZE,
				BATCH_SIZE,
				L,
				active_queue,
				end_active_queue,
				is_active,
				has_cand_queue,
				end_has_cand_queue,
				has_candidates,
				dists_table,
				roots_labels_buffer,
				short_index,
				has_new_labels_queue,
				end_has_new_labels_queue,
				has_new_labels);
	}
	if (0 != remainer) {
		vertex_centric_labeling_in_batches(
				G,
				b_i_bound,
				b_i_bound * BATCH_SIZE,
				remainer,
				L,
				active_queue,
				end_active_queue,
				is_active,
				has_cand_queue,
				end_has_cand_queue,
				has_candidates,
				dists_table,
				roots_labels_buffer,
				short_index,
				has_new_labels_queue,
				end_has_new_labels_queue,
				has_new_labels);
	}
	time_labeling += WallTimer::get_time_mark();
//	cache_miss.measure_stop();

	// Statistics Print Out
	setlocale(LC_NUMERIC, ""); // For print large number with comma
	printf("BATCH_SIZE: %u\n", BATCH_SIZE);
//	printf("BP_Size: %u\n", BITPARALLEL_SIZE);
//	printf("Initializing: %f %.2f%%\n", initializing_time, initializing_time / time_labeling * 100);
//		printf("\tinit_start_reset_time: %f (%f%%)\n", init_start_reset_time, init_start_reset_time / initializing_time * 100);
//		printf("\tinit_index_time: %f (%f%%)\n", init_index_time, init_index_time / initializing_time * 100);
//			printf("\t\tinit_indicators_time: %f (%f%%)\n", init_indicators_time, init_indicators_time / init_index_time * 100);
//		printf("\tinit_dist_matrix_time: %f (%f%%)\n", init_dist_matrix_time, init_dist_matrix_time / initializing_time * 100);
//	printf("Candidating: %f %.2f%%\n", candidating_time, candidating_time / time_labeling * 100);
//	printf("Adding: %f %.2f%%\n", adding_time, adding_time / time_labeling * 100);
//		printf("distance_query_time: %f %.2f%%\n", distance_query_time, distance_query_time / time_labeling * 100);
//		printf("SIMD_utilization: %.2f%% %'lu %'lu\n", 100.0 * simd_util.first / simd_util.second, simd_util.first, simd_util.second);
//		printf("SIMD_full_ratio: %.2f%% %'lu %'lu\n", 100.0 * simd_full_count.first / simd_full_count.second, simd_full_count.first, simd_full_count.second);
		//printf("check_count: %'lu\n", check_count);
		//uint64_t total_hit_count = l_l_hit_count + vl_cl_hit_count + vl_cc_hit_count + vc_cl_hit_count + vc_cc_hit_count;
		//printf("l_l_hit_count: %'lu %.2f%% %.2f%%\n", l_l_hit_count, 100.0 * l_l_hit_count / total_hit_count, 100.0 * l_l_hit_count / check_count);
		//printf("vl_cl_hit_count: %'lu %.2f%% %.2f%%\n", vl_cl_hit_count, 100.0 * vl_cl_hit_count / total_hit_count, 100.0 * vl_cl_hit_count / check_count);
		//printf("vl_cc_hit_count: %'lu %.2f%% %.2f%%\n", vl_cc_hit_count, 100.0 * vl_cc_hit_count / total_hit_count, 100.0 * vl_cc_hit_count / check_count);
		//printf("vc_cl_hit_count: %'lu %.2f%% %.2f%%\n", vc_cl_hit_count, 100.0 * vc_cl_hit_count / total_hit_count, 100.0 * vc_cl_hit_count / check_count);
		//printf("vc_cc_hit_count: %'lu %.2f%% %.2f%%\n", vc_cc_hit_count, 100.0 * vc_cc_hit_count / total_hit_count, 100.0 * vc_cc_hit_count / check_count);
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
//	printf("Correction: %f %.2f%%\n", correction_time, 100.0 * correction_time / time_labeling);
//	printf("Monotone_count: %'lu %.2f%%\n", monotone_count, 100.0 * monotone_count / checked_count);
	printf("Total_labeling_time: %.2f seconds\n", time_labeling);
}

// Function: assign lanes of vindex in base_addr as int a.
template <inti BATCH_SIZE>
inline void WeightedVertexCentricPLL<BATCH_SIZE>::int_i32scatter_epi8(
		weighti *base_addr, 
		__m512i vindex, 
		weighti a)
{
	int *tmp_vindex = (int *) &vindex;
	for (inti i = 0; i < 16; ++i) {
		base_addr[tmp_vindex[i]] = a;
	}
}
// Function: according to mask k, assign lanes of vindex in base_addr as a
template <inti BATCH_SIZE>
inline void WeightedVertexCentricPLL<BATCH_SIZE>::int_mask_i32scatter_epi8(
		weighti *base_addr, 
		__mmask16 k, 
		__m512i vindex, 
		weighti a)
{
	int *tmp_vindex = (int *) &vindex;
	for (inti i = 0; i < 16; ++i) {
		uint16_t bit = (uint16_t) pow(2.0, i);
		if (k & bit) {
			base_addr[tmp_vindex[i]] = a;
		}
	}
}

//epi32_mask_unpack_into_queue(); // r_root_id_v to SI_v_tail.candidates_que
//epi32_mask_i32scatter_epi8(); // tmp_dist_r_t_v to SI_v_tail.candidates_dists

// Function: according to mask, assign corresponding lanes in 'a' into base_addr and plus index offsets in vindex
template <inti BATCH_SIZE>
inline void WeightedVertexCentricPLL<BATCH_SIZE>::epi32_mask_i32scatter_epi8(
		weighti *base_addr,
		__mmask16 mask,
		__m512i vindex,
		__m512i a)
{
	int *tmp_vindex = (int *) &vindex;
	int *tmp_a = (int *) &a;
	for (inti i = 0; i < 16; ++i) {
		uint16_t bit = (uint16_t) pow(2.0, i);
		if (mask & bit) {
			base_addr[tmp_vindex[i]] = tmp_a[i];
		}
	}
}

// Function: according to mask, put corresponding lanes in 'a' into the queue.
template <inti BATCH_SIZE>
inline void WeightedVertexCentricPLL<BATCH_SIZE>::epi32_mask_unpack_into_queue(
		vector<inti> &queue,
		inti &end_queue,
		__mmask16 mask,
		__m512i a)
{
	int *tmp_a = (int *) &a;
	for (inti i = 0; i < 16; ++i) {
		uint16_t bit = (uint16_t) pow(2.0, i);
		if (mask & bit) {
			queue[end_queue++] = tmp_a[i];
		}
	}
}

//// Have not been tested yet.
//inline void WeightedVertexCentricPLL::epi8_i32scatter_epi8(
//		uint8_t *base_addr,
//		__m512i vindex,
//		__m128i a)
//{
//	uint8_t *tmp_a = (uint8_t *) &a;
//	for (inti i = 0; i < 16; ++i) {
//		base_addr[vindex[i]] = tmp_a[i];
//	}
//}
//
//inline void WeightedVertexCentricPLL::epi8_mask_i32scatter_epi8(
//		uint8_t *base_addr,
//		__mmask16 mask,
//		__m512i vindex,
//		__m128i a)
//{
//	uint8_t *tmp_a = (uint8_t *) &a;
//	for (inti i = 0; i < 16; ++i) {
//		base_addr[vindex[i]] = tmp_a[i];
//	}
//}


template <inti BATCH_SIZE>
void WeightedVertexCentricPLL<BATCH_SIZE>::store_index_to_file(
								const char *filename,
								const vector<idi> &rank2id)
{
	ofstream fout(filename);
	if (!fout.is_open()) {
		fprintf(stderr, "Error: cannot open file %s\n", filename);
		exit(EXIT_FAILURE);
	}
	idi num_v = rank2id.size();
	vector< vector< pair<idi, weighti> > > ordered_L(num_v);
	Index.resize(num_v);
	// Store into file the number of vertices and the number of bit-parallel roots.
	fout.write((char *) &num_v, sizeof(num_v));

	// Traverse the L, put them into Index (ordered labels)
	for (idi v_id = 0; v_id < num_v; ++v_id) {
		idi new_v = rank2id[v_id];
		IndexOrdered & Iv = Index[new_v];
		const IndexType &Lv = L[v_id];
		auto &OLv = ordered_L[new_v];

		// Normal Labels
		// Traverse v_id's all existing labels
//		for (inti b_i = 0; b_i < Lv.batches.size(); ++b_i) {
//			idi id_offset = Lv.batches[b_i].batch_id * BATCH_SIZE;
//			idi dist_start_index = Lv.batches[b_i].start_index;
//			idi dist_bound_index = dist_start_index + Lv.batches[b_i].size;
//			// Traverse dist_matrix
//			for (idi dist_i = dist_start_index; dist_i < dist_bound_index; ++dist_i) {
//				idi v_start_index = Lv.distances[dist_i].start_index;
//				idi v_bound_index = v_start_index + Lv.distances[dist_i].size;
//				inti dist = Lv.distances[dist_i].dist;
//				for (idi v_i = v_start_index; v_i < v_bound_index; ++v_i) {
//					idi tail = Lv.vertices[v_i] + id_offset;
////					idi new_tail = rank2id[tail];
////					new_L[new_v].push_back(make_pair(new_tail, dist));
//					OLv.push_back(make_pair(tail, dist));
//				}
//			}
//		}

		for (idi l_i = 0; l_i < Lv.vertices.size(); ++l_i) {
			OLv.push_back(make_pair(Lv.vertices[l_i], Lv.distances[l_i]));
		}
		// Sort
		sort(OLv.begin(), OLv.end());
		// Store into Index
		inti size_labels = OLv.size();
		Iv.label_id.resize(size_labels + 1); // Adding one for Sentinel
		Iv.label_dists.resize(size_labels + 1); // Adding one for Sentinel
		for (inti l_i = 0; l_i < size_labels; ++l_i) {
			Iv.label_id[l_i] = OLv[l_i].first;
			Iv.label_dists[l_i] = OLv[l_i].second;
		}
		Iv.label_id[size_labels] = num_v; // Sentinel
		Iv.label_dists[size_labels] = WEIGHTI_MAX; // Sentinel
	}

	uint64_t labels_count = 0;
	// Traverse the Index, store labels into file
	for (idi v_id = 0; v_id < num_v; ++v_id) {
		IndexOrdered & Iv = Index[v_id];

		// Normal Labels
		// Store Labels into file.
		idi size_labels = Iv.label_id.size() - 1; // Remove the Sentinel
		labels_count += size_labels;
		fout.write((char *) &size_labels, sizeof(size_labels));
		for (idi l_i = 0; l_i < size_labels; ++l_i) {
			idi l = Iv.label_id[l_i];
			weighti d = Iv.label_dists[l_i];
			fout.write((char *) &l, sizeof(l));
			fout.write((char *) &d, sizeof(d));
		}
	}
	printf("Label_size: %'lu mean: %f\n", labels_count, static_cast<double>(labels_count) / num_v);
	fout.close();
}

template <inti BATCH_SIZE>
void WeightedVertexCentricPLL<BATCH_SIZE>::load_index_from_file(
								const char *filename)
{
	ifstream fin(filename);
	if (!fin.is_open()) {
		fprintf(stderr, "Error: cannot open file %s\n", filename);
		exit(EXIT_FAILURE);
	}
	idi num_v;
	// Load from file the number of vertices and the number of bit-parallel roots.
	fin.read((char *) &num_v, sizeof(num_v));
	num_v_ = num_v;
	Index.resize(num_v);
	uint64_t labels_count = 0;
	// Load labels for every vertex
	for (idi v_id = 0; v_id < num_v; ++v_id) {
		IndexOrdered &Iv = Index[v_id];
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


template <inti BATCH_SIZE>
void WeightedVertexCentricPLL<BATCH_SIZE>::order_labels(
								const vector<idi> &rank2id,
								const vector<idi> &rank)
{
	idi num_v = rank.size();
	vector< vector< pair<idi, weighti> > > ordered_L(num_v);
	idi labels_count = 0;
	Index.resize(num_v);

	// Traverse the L, put them into Index (ordered labels)
	for (idi v_id = 0; v_id < num_v; ++v_id) {
		idi new_v = rank2id[v_id];
		IndexOrdered & Iv = Index[new_v];
		const IndexType &Lv = L[v_id];
		auto &OLv = ordered_L[new_v];
		// Bit-parallel Labels

		// Normal Labels
		// Traverse v_id's all existing labels
		for (idi l_i = 0; l_i < Lv.vertices.size(); ++l_i) {
			OLv.push_back(make_pair(Lv.vertices[l_i], Lv.distances[l_i]));
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

template <inti BATCH_SIZE>
weighti WeightedVertexCentricPLL<BATCH_SIZE>::query_distance(
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
	for (idi i1 = 0, i2 = 0; ; ) {
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

template <inti BATCH_SIZE>
void WeightedVertexCentricPLL<BATCH_SIZE>::switch_labels_to_old_id(
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
			new_L[new_v].push_back(make_pair(rank2id[tail], dist));
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
	printf("Label_sum: %'u %'u mean: %f\n", label_sum, test_label_sum, label_sum * 1.0 / num_v);

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
