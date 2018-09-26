/*
 * pado.h
 *
 *  Created on: Sep 4, 2018
 *      Author: Zhen Peng
 * 
 * I want to label the roots of the batch at first, then push their labels forward.
 * In this way, the time complexity is almost only O(n+m) rather than O(n(n+m)).
 * However, when I label the roots, I cannot guarantee the shortest distance if
 * I only visit roots during traverse. This can not be torlerated as the distance
 * is not correct. 09/18/2018
 */

#ifndef INCLUDES_PADO_H_
#define INCLUDES_PADO_H_

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <limits.h>
#include "globals.h"
#include "graph.h"
#include "index.h"

using std::vector;
using std::unordered_map;
using std::stable_sort;
using std::min;

namespace PADO {

class BitArray {
private:
	uint64_t *bits = nullptr;
	inti size = 0;

public:
	BitArray(unsigned len);
	BitArray(unsigned len, uint64_t num) : BitArray(len)
	{
		*bits = num;
	}
	BitArray(const BitArray &right);
	BitArray &operator=(const BitArray &right) = delete;
	BitArray(BitArray &&right)
	{
//		puts("BitArray Move..."); fflush(stdout);//test
		bits = right.bits;
		size = right.size;

		right.bits = nullptr;
		right.size = 0;
	}
	BitArray &operator=(BitArray &&right) = delete;

	~BitArray()
	{
//		puts("BitArray Destructor...");fflush(stdout);//test
		free(bits);
		size = 0;
	}

	// Execute fun for every set bit in bits (bit array)
	template<typename T> void process_every_bit(T fun); // fun is an expected function
	vector<inti> get_all_locs_set(inti bound) const; // return the vector contains all locations which are set
	inti is_bit_set(inti loc) const;
	void set_bit(inti loc)
	{
		if (loc >= size) {
			fprintf(stderr, "Error: BitArray::set_bit: loc %u is larger than size %u.\n", loc, size);
			return;
		}
		inti remain = loc % 64;
		inti count = loc / 64;
		*(bits + count) |= ((uint64_t) 1 << remain);
//		*((uint64_t *) ((uint8_t *) bits + count * 8)) |= ((uint64_t) 1 << remain);
	}
	void unset_bit(inti loc)
	{
		if (loc >= size) {
			fprintf(stderr, "Error: BitArray::unset_bit: loc %u is larger than size %u.\n", loc, size);
			return;
		}
		inti remain = loc % 64;
		inti count = loc / 64;
		*(bits + count) &= (~((uint64_t) 1 << remain));
//		*((uint64_t *) ((uint8_t *) bits + count * 8)) &= (~((uint64_t) 1 << remain));
	}

	void unset_all()
	{
		memset(bits, 0, size / 64 * sizeof(uint64_t));
	}

	inti get_size() const
	{
		return size;
	}

};

BitArray::BitArray(inti len)
{
//	puts("BitArray Constructor...");fflush(stdout);//test
	if (len == 0 || len % 64 != 0) {
		fprintf(stderr, "Error: BitArray: the length should be divisible by 64\n");
		return;
	}
	size = len;
	inti b_l = len / 64;
	bits = (uint64_t *) calloc(b_l, sizeof(uint64_t));
	if (NULL == bits) {
		fprintf(stderr, "Error: BitArray: no enough memory for %u bytes.\n", b_l * 8);
		return;
	}
}

BitArray::BitArray(const BitArray &right)
{
//	puts("BitArray Copy...");fflush(stdout);//test
	size = right.size;
	inti b_l = size / 64;
	bits = (uint64_t *) calloc(b_l, sizeof(uint64_t));
	if (NULL == bits) {
		fprintf(stderr, "Error: BitArray: no enough memory for %u bytes.\n", b_l * 8);
		return;
	}
	memcpy(bits, right.bits, b_l * sizeof(uint64_t));
}

template<typename T>
void BitArray::process_every_bit(T fun)
{
	inti count = size / 64;
	for (inti c = 0; c < count; ++c) {
		uint64_t *num_64 = bits + c;
		if (0 == *num_64) {
			continue;
		}
		for (inti i_32 = 0; i_32 < 8; i_32 += 4) {
			uint32_t *num_32 = (uint32_t *) ((uint8_t *) num_64 + i_32);
			if (0 == *num_32) {
				continue;
			}
			for (inti i_16 = 0; i_16 < 4; i_16 += 2) {
				uint16_t *num_16 = (uint16_t *) ((uint8_t *) num_32 + i_16);
				if (0 == *num_16) {
					continue;
				}
				for (inti i_8 = 0; i_8 < 2; i_8 += 1) {
					uint8_t *num_8 = (uint8_t *) ((uint8_t *) num_16 + i_8);
//					printf("num_8: %u\n", *num_8);//test
					if (0 == *num_8) {
						continue;
					}
					inti offset = (i_8 + i_16 + i_32) * 8 + c * 64;
					for (inti i_1 = 0; i_1 < 8; ++i_1) {
						if (*num_8 & (1 << i_1)) {
							fun(i_1 + offset);
						}
					}
				}
			}
		}
	}
}

vector<inti> BitArray::get_all_locs_set(inti bound) const
{
	vector<inti> locs;
	inti count = size / 64;
//	if (bound < size) {
//		count = bound / 64;
//	}
	for (inti c = 0; c < count; ++c) {
		uint64_t *num_64 = bits + c;
		if (0 == *num_64) {
			continue;
		}
		for (inti i_32 = 0; i_32 < 2; ++i_32) {
			uint32_t *num_32 = (uint32_t *) num_64 + i_32;
			if (0 == *num_32) {
				continue;
			}
			for (inti i_16 = 0; i_16 < 2; ++i_16) {
				uint16_t *num_16 = (uint16_t *) num_32 + i_16;
				if (0 == *num_16) {
					continue;
				}
				for (inti i_8 = 0; i_8 < 2; ++i_8) {
					uint8_t *num_8 = (uint8_t *) num_16 + i_8;
					if (0 == *num_8) {
						continue;
					}
//					inti offset = c * 64 + i_32 * 32 + i_16 * 16 + i_8 * 8;
					inti offset = (c << 6) + (i_32 << 5) + (i_16 << 4) + (i_8 << 3);
					for (inti i_1 = 0; i_1 < 8; ++i_1) {
						inti index = i_1 + offset;
						if (index >= bound) {
							return locs;
						}
						if (*num_8 & (1 << i_1)) {
							locs.push_back(index);
						}
					}
				}
			}
		}
	}
//	for (inti c = 0; c < count; ++c) {
//		uint64_t *num_64 = bits + c;
//		if (0 == *num_64) {
//			continue;
//		}
//		for (inti i_32 = 0; i_32 < 8; i_32 += 4) {
//			uint32_t *num_32 = (uint32_t *) ((uint8_t *) num_64 + i_32);
//			if (0 == *num_32) {
//				continue;
//			}
//			for (inti i_16 = 0; i_16 < 4; i_16 += 2) {
//				uint16_t *num_16 = (uint16_t *) ((uint8_t *) num_32 + i_16);
//				if (0 == *num_16) {
//					continue;
//				}
//				for (inti i_8 = 0; i_8 < 2; i_8 += 1) {
//					uint8_t *num_8 = (uint8_t *) ((uint8_t *) num_16 + i_8);
//					if (0 == *num_8) {
//						continue;
//					}
//					inti offset = (i_8 + i_16 + i_32) * 8 + c * 64;
//					if (offset >= bound) {
//						return locs;
//					}
//					for (inti i_1 = 0; i_1 < 8; ++i_1) {
//						if (*num_8 & (1 << i_1)) {
////							fun(i_1 + offset);
//							locs.push_back(i_1 + offset);
//						}
//					}
//				}
//			}
//		}
//	}
	return locs;
}

inti BitArray::is_bit_set(inti loc) const
{
	if (loc >= size) {
		fprintf(stderr, "Error: BitArray::is_bit_set: loc %u is larger than size %u.\n", loc, size);
		return 0;
	}
	inti remain_64 = loc % 64;
	inti count_64 = loc / 64;
	uint64_t *num_64 = bits + count_64;
	if (0 == *num_64) {
		return 0;
	}
	inti remain_32 = remain_64 % 32;
	inti count_32 = remain_64 / 32;
	uint32_t *num_32 = (uint32_t *) num_64 + count_32;
	if (0 == *num_32) {
		return 0;
	}
	inti remain_16 = remain_32 % 16;
	inti count_16 = remain_32 / 16;
	uint16_t *num_16 = (uint16_t *) num_32 + count_16;
	if (0 == *num_16) {
		return 0;
	}
	inti remain_8 = remain_16 % 8;
	inti count_8 = remain_16 / 8;
	uint8_t *num_8 = (uint8_t *) num_16 + count_8;
	if (0 == *num_8) {
		return 0;
	}
	if (*num_8 & ((uint8_t) 1 << remain_8)) {
		return 1;
	} else {
		return 0;
	}
}
// End class BitArray


//// Batch based processing, 09/11/2018
class VertexCentricPLL {
private:
	struct ShortIndex {
		BitArray roots_indicator; // if evert root r in the short index
		smalli *roots_distances = nullptr; // distance to root r
		BitArray roots_candidates; // candidates for the current iteration
		smalli *candidates_dist = nullptr;
		BitArray roots_last; // added roots from the last iteration
		smalli *last_dist = nullptr;

		ShortIndex(inti size): roots_indicator(size), roots_candidates(size), roots_last(size)
		{
//			puts("ShortIndex Constructor...");fflush(stdout);//test
			roots_distances = (smalli *) malloc(size * sizeof(smalli));
			memset(roots_distances, (uint8_t) -1, size * sizeof(smalli));
			candidates_dist = (smalli *) malloc(size * sizeof(smalli));
			memset(candidates_dist, (uint8_t) -1, size * sizeof(smalli));
			last_dist = (smalli *) malloc(size * sizeof(smalli));
			memset(last_dist, (uint8_t) -1, size * sizeof(smalli));

		}
		ShortIndex(const ShortIndex &right) :
								roots_indicator(right.roots_indicator),
								roots_candidates(right.roots_candidates),
								roots_last(right.roots_last)
		{
//			puts("ShortIndex Copy...");fflush(stdout);//test
			inti size = right.roots_indicator.get_size();
			roots_distances = (smalli *) malloc(size * sizeof(smalli));
			memcpy(roots_distances, right.roots_distances, size * sizeof(smalli));
			candidates_dist = (smalli *) malloc(size * sizeof(smalli));
			memcpy(candidates_dist, right.candidates_dist, size * sizeof(smalli));
			last_dist = (smalli *) malloc(size * sizeof(smalli));
			memcpy(last_dist, right.last_dist, size * sizeof(smalli));
		}
		ShortIndex &operator=(const ShortIndex &right) = delete;
		ShortIndex(ShortIndex &&right) :
								roots_indicator(right.roots_indicator),
								roots_candidates(right.roots_candidates),
								roots_last(right.roots_last)
		{
//			puts("ShortIndex Move..."); fflush(stdout);//test
			roots_distances = right.roots_distances;
			right.roots_distances = nullptr;
			candidates_dist = right.candidates_dist;
			right.candidates_dist = nullptr;
			last_dist = right.last_dist;
			right.last_dist = nullptr;
		}
		ShortIndex &operator=(ShortIndex &&right) = delete;
		~ShortIndex()
		{
//			puts("ShortIndex Destructor...");fflush(stdout);//test
//			puts("roots_indicator...");fflush(stdout);//test
//			roots_indicator.~BitArray();
//			puts("roots_distances...");fflush(stdout);//test
			free(roots_distances);
//			puts("roots_candidates...");fflush(stdout);//test
//			roots_candidates.~BitArray();
//			puts("roots_last...");fflush(stdout);//test
//			roots_last.~BitArray();
		}
//		void unset_all(inti size)
//		{
//			roots_indicator.unset_all();
//			memset(roots_distances, (uint8_t) -1, size * sizeof(smalli));
//			roots_candidates.unset_all();
//			roots_last.unset_all();
//		}
		void set_index(inti loc, smalli d)
		{
			roots_indicator.set_bit(loc);
			roots_distances[loc] = d;
		}
		void set_candidate(inti loc, smalli d)
		{
			roots_candidates.set_bit(loc);
			candidates_dist[loc] = d;
		}
		void set_last(inti loc, smalli d)
		{
			roots_last.set_bit(loc);
			last_dist[loc] = d;
		}
	}; // struct ShortIndex

	vector<IndexType> L;
	void construct(const Graph &G, const vector<idi> &rank);
	void root_batch(
			const Graph &G,
			idi root_start,
			inti roots_size,
			inti bitarray_size,
			vector<IndexType> &L);
	void roots_labeling(
			const Graph &G,
			idi root_start,
			inti roots_size,
//			inti bit_array_size,
			vector<ShortIndex> &short_index,
			vector< vector<smalli> > &dist_matrix);
//			(vector<IndexType> &L);

	// Used only by BitArray::get_all_locs_set(), return the boundary for collecting all set locations
	inti get_bound(const idi &v, const idi &root_start, const idi &roots_size) const
	{
		inti bound = v - root_start;
		if (v < root_start || roots_size <= bound) {
			bound = roots_size;
		} else {
			++bound;
		}
		return bound;
	}

	bool is_root(const idi &v, const idi &root_start, const idi &roots_size) const
	{
		if (v < root_start || v - root_start >= roots_size) {
			return false;
		} else {
			return true;
		}
	}

	void switch_labels_to_old_id(const vector<idi> &rank2id);



public:
	VertexCentricPLL() = default;
	VertexCentricPLL(const Graph &G, const vector<idi> &rank2id);

	weighti query(
			idi u,
			idi v,
			const vector<IndexType> &index,
			const vector<ShortIndex> &short_index,
			idi root_start,
			inti roots_size);
//	weighti query(
//								idi u,
//								idi v,
//								const Graph &G,
//								const vector<IndexType> &index);
//	static vector<idi> make_rank(const Graph &G);
	void print();


}; // class VertexCentricPLL

VertexCentricPLL::VertexCentricPLL(const Graph &G, const vector<idi> &rank2id)
{
	construct(G, rank2id);
}

//// For index by vector, with batch and distance array, 09/09/2018
//void VertexCentricPLL::root_batch(
//						const Graph &G,
//						idi root_start, // start id of roots
//						inti roots_size, // how many roots in the batch
//						inti bit_array_size, // the fix length for bit array, must be divided by 64
//						vector<IndexType> &L)
//{
//	double time_can = 0;
//	double time_add = 0;
//
//
//	idi num_v = G.get_num_v();
//	vector<ShortIndex> short_index(num_v, bit_array_size);
//	vector<bool> is_active(num_v, false);
//
//	// Use a distance map to save time, 09/16/2018
//	vector< vector<smalli> > dist_matrix(roots_size);
//	for (idi r = 0; r < roots_size; ++r) {
//		vector<smalli> &dmr = dist_matrix[r];
//		dmr.resize(num_v, SMALLI_MAX);
//		idi root_id = r + root_start;
//		const IndexType &Lr = L[root_id];
//		idi size = Lr.get_size();
//		for (idi vi = 0; vi < size; ++vi) {
//			idi v = Lr.get_label_ith_v(vi);
//			dmr[v] = Lr.get_label_ith_d(vi);
//		}
//		dmr[r] = 0;
//	}
//	// End distance map
//
//	// Initialize roots' short_index
//	for (inti r_i = 0; r_i < roots_size; ++r_i) {
//		idi v = r_i + root_start;
//		short_index[v].set_index(r_i, 0);
//		short_index[v].roots_last.set_bit(r_i);
//		is_active[v] = true;
//	}
//	smalli iter = 1; // iterator, also the distance for current iteration
//	bool stop = false;
//	while (!stop) {
////		printf("iter: %u\n", iter);//test
//		WallTimer t_can("Candidating");
//		stop = true;
//		// Push to Candidate
//		for (idi head = 0; head < num_v; ++head) {
//			if (!is_active[head]) {
//				continue;
//			}
//			ShortIndex &head_si = short_index[head];
//			idi degree = G.ith_get_out_degree(head);
//			inti bound = get_bound(head, root_start, roots_size);
//			vector<inti> head_roots = head_si.roots_last.get_all_locs_set(bound);
//			for (idi e_i = 0; e_i < degree; ++e_i) {
//				idi tail = G.ith_get_edge(head, e_i);
//				ShortIndex &tail_si = short_index[tail];
//				for (const inti &hi : head_roots) {
//					idi x = hi + root_start;
//					if (tail < x) {
//						break;
//					}
//					if (
////						x < tail
////						&& !L[tail].is_v_in_label(x)
//						&& !tail_si.roots_candidates.is_bit_set(hi)){
//						tail_si.roots_candidates.set_bit(hi);
//					}
//				}
//			}
//			head_si.roots_last.unset_all();
//			is_active[head] = false;
//		}
//
//		t_can.print_runtime();
//		time_can += t_can.get_runtime();
//		WallTimer t_add("Adding");
//
//		// Traverse Candidate then add to short index
//		for (idi v = 0; v < num_v; ++v) {
//			ShortIndex &v_si = short_index[v];
//			inti bound = get_bound(v, root_start, roots_size);
//			vector<inti> candidates = v_si.roots_candidates.get_all_locs_set(bound);
//			if (0 == candidates.size()) {
//				continue;
//			}
//			for (const inti &r : candidates) {
////				weighti d = query(
////								v,
////								r + root_start,
////								L,
////								short_index,
////								root_start,
////								roots_size);
//				// The new query based on the distance matrix 09/16/2018
//				// Query based on the old Labels.
//				weighti d = WEIGHTI_MAX;
//				const IndexType &Lv = L[v];
//				idi size = Lv.get_size();
//				for (idi vi = 0; vi < size; ++vi) {
//					idi v_l = Lv.get_label_ith_v(vi);
//					if (SMALLI_MAX == dist_matrix[r][v_l]) {
//						continue;
//					}
//					weighti q_d = Lv.get_label_ith_d(vi) + dist_matrix[r][v_l];
//					if (q_d < d) {
//						d = q_d;
//					}
//				}
//				// Query based on short index in this batch.
//				const ShortIndex &SIv = short_index[v];
//				inti bound = get_bound(v, root_start, roots_size);
//				vector<inti> roots_v = SIv.roots_indicator.get_all_locs_set(bound);
//				for (const inti &rv : roots_v) {
//					if (SMALLI_MAX == dist_matrix[r][rv]) {
//						continue;
//					}
//					weighti q_d = SIv.roots_distances[rv] + dist_matrix[r][rv];
//					if (q_d < d) {
//						d = q_d;
//					}
//				}
//				// End the new query
//				if (iter < d) {
//					v_si.set_index(r, iter);
//					v_si.roots_last.set_bit(r);
//					is_active[v] = true;
//
//					idi root_id = v - root_start;
//					if (root_start <= v && root_id < roots_size) {
//						dist_matrix[root_id][r] = iter;
//					}
//
//
//					if (stop) {
//						stop = false;
//					}
//				}
//			}
//			v_si.roots_candidates.unset_all();
//		}
//		++iter;
//
//		t_add.print_runtime();
//		time_add += t_add.get_runtime();
//	}
//	// add short_index to L
//	double time_update = 0;
//	WallTimer t_update("Updating");
//	for (idi v = 0; v < num_v; ++v) {
//		const ShortIndex &v_si = short_index[v];
//		IndexType &Lv = L[v];
//		inti bound = get_bound(v, root_start, roots_size);
//		vector<inti> indicators = v_si.roots_indicator.get_all_locs_set(bound);
//		if (0 == indicators.size()) {
//			continue;
//		}
//		for (const inti &i : indicators) {
//			idi u = i + root_start;
//			Lv.add_label_seq(u, v_si.roots_distances[i]);
//		}
//	}
//	t_update.print_runtime();
//	time_update += t_update.get_runtime();
//
//	double total_time = time_can + time_add + time_update;
//	printf("Candidating time: %f (%f%%)\n", time_can, time_can / total_time * 100);
//	printf("Adding time: %f (%f%%)\n", time_add, time_add / total_time * 100);
//	printf("Updating time: %f (%f%%)\n", time_update, time_update / total_time * 100);
//}
/////////////////////////////////////////////////////////////////////////////


// At first, label the roots
// 9/17/2018
void VertexCentricPLL::roots_labeling(
						const Graph &G,
						idi root_start,
						inti roots_size,
//						inti bit_array_size,
						vector<ShortIndex> &short_index,
						vector< vector<smalli> > &dist_matrix)
//						(vector<IndexType> &L)
{
//	idi num_v = G.get_num_v();
	idi root_bound = root_start + roots_size;
//	vector<ShortIndex> short_index(roots_size, bit_array_size);
	vector< vector<bool> > is_visited_matrix(roots_size, vector<bool>(roots_size, false));
	vector< vector<bool> > is_active_matrix(roots_size, vector<bool>(roots_size, false));

//	vector< vector<smalli> > dist_matrix(roots_size);

//	for (inti r = 0; r < roots_size; ++r) {
//		vector<smalli> &dmr = dist_matrix[r];
//		dmr.resize(roots_size, SMALLI_MAX);
//		idi root_id = r + root_start;
//		const IndexType &Lr = L[root_id];
//		idi size = Lr.get_size();
//		for (idi vi = 0; vi < size; ++vi) {
//			idi v = Lr.get_label_ith_v(vi);
//			dmr[v] = Lr.get_label_ith_d(vi);
//		}
//		dmr[r] = 0;
//	}
	// Initialize flag arrays
	for (inti r_i = 0; r_i < roots_size; ++r_i) {
//		idi root_id = r_i + root_start;
//		short_index[root_id].set_index(r_i, 0);
//		short_index[root_id].roots_last.set_bit(r_i);
		is_visited_matrix[r_i][r_i] = true;
		is_active_matrix[r_i][r_i] = true;
	}
	smalli iter = 1; // iterator, also the distance for current iteration
	bool stop = false;
	vector<bool> stop_root(roots_size, false);
	while (!stop) {
		stop = true;
		for (inti root_i = 0; root_i < roots_size; ++root_i) {
			// BFS for from every root
			if (stop_root[root_i]) {
				continue;
			}
			// Candidating
			vector<bool> &is_active_r = is_active_matrix[root_i];
			vector<bool> &is_visited_r = is_visited_matrix[root_i];
			vector<bool> has_candidate(roots_size, false);
			for (idi head_id = root_start; head_id < root_bound; ++head_id) {
				inti root_id = head_id - root_start;
				if (!is_active_r[root_id]) {
					continue;
				}
				is_active_r[root_id] = false;
				ShortIndex &head_si = short_index[head_id];
				idi degree = G.ith_get_out_degree(head_id);
				vector<inti> head_roots = head_si.roots_last.get_all_locs_set(root_id + 1);
				for (idi e_i = 0; e_i < degree; ++e_i) {
					idi tail = G.ith_get_edge(head_id, e_i);
					inti tail_r_id = tail - root_start;
					if (!is_root(tail, root_start, roots_size)
						|| is_visited_r[tail_r_id]) {
						continue;
					}
					ShortIndex &tail_si = short_index[tail];
					for (const inti &hi : head_roots) {
						idi x = hi + root_start;
						if (tail < x) {
							break;
						}
						if (
//								x < tail
//								&& !L[tail].is_v_in_label(x) &&
								!tail_si.roots_candidates.is_bit_set(hi)){
							tail_si.roots_candidates.set_bit(hi);
							if (!has_candidate[tail_r_id]) {
								has_candidate[tail_r_id] = true;
							}
						}
					}
				}
				head_si.roots_last.unset_all();

			}

			// Adding to short index
			// v : root_id
			// r : cand
			for (inti root_id = 0; root_id < roots_size; ++root_id) {
				if (!has_candidate[root_id]) {
					continue;
				}
				idi root_v_id = root_id + root_start;
				ShortIndex &v_si = short_index[root_v_id];
				vector<inti> candidates = v_si.roots_candidates.get_all_locs_set(root_id + 1);
				for (const inti &cand : candidates) {
					// The new query based on the distance matrix 09/16/2018
					// Query based on the old Labels.
					weighti d = WEIGHTI_MAX;
					const IndexType &Lv = L[root_v_id];
					idi size = Lv.get_size();
					for (idi vi = 0; vi < size; ++vi) {
						idi v_l = Lv.get_label_ith_v(vi);
						if (!is_root(v_l, root_start, roots_size)) {
							continue;
						}
//						inti v_l_root_id = v_l - root_start;
						if (SMALLI_MAX == dist_matrix[cand][v_l]) {
							continue;
						}
						weighti q_d = Lv.get_label_ith_d(vi) + dist_matrix[cand][v_l];
						if (q_d < d) {
							d = q_d;
						}
					}
					// Query based on short index in this batch.
//					const ShortIndex &SIv = short_index[root_id];
					vector<inti> roots_v = v_si.roots_indicator.get_all_locs_set(root_id + 1);
					for (const inti &rv : roots_v) {
						if (SMALLI_MAX == dist_matrix[cand][rv]) {
							continue;
						}
						weighti q_d = v_si.roots_distances[rv] + dist_matrix[cand][rv];
						if (q_d < d) {
							d = q_d;
						}
					}
					// End the new query
					if (iter < d) {
						v_si.set_index(cand, iter);
						v_si.roots_last.set_bit(cand);
						is_active_r[root_id] = true;
						dist_matrix[root_id][cand] = iter;

						is_active_r[root_id] = true;
						is_visited_r[root_id] = true;
						if (stop_root[root_i]) {
							stop_root[root_i] = false;

							if (stop) {
								stop = false;
							}
						}
					}
				}
				v_si.roots_candidates.unset_all();
			}
		}
		++iter;
	}
//	// add short_index to L
//	for (idi v = 0; v < roots_size; ++v) {
//		const ShortIndex &v_si = short_index[v];
//		IndexType &Lv = L[v + root_start];
//		vector<inti> indicators = v_si.roots_indicator.get_all_locs_set(v + 1);
//		if (0 == indicators.size()) {
//			continue;
//		}
//		for (const inti &i : indicators) {
//			idi u = i + root_start;
//			Lv.add_label_seq(u, v_si.roots_distances[i]);
//		}
//	}
}
// root_batch:
// The batch process will be based on core-peripheral idea,
// label the roots first, then push their labels forward
// In this way, the time complexity should be almost O(n+m) rather than O(n(n+m)).
// 9/17/2018
void VertexCentricPLL::root_batch(
						const Graph &G,
						idi root_start, // start id of roots
						inti roots_size, // how many roots in the batch
						inti bit_array_size, // the fix length for bit array, must be divided by 64
						vector<IndexType> &L)
{
	double time_can = 0;
	double time_add = 0;


	idi num_v = G.get_num_v();
	vector<ShortIndex> short_index(num_v, bit_array_size);
	vector<bool> is_active(num_v, false);
	vector<bool> is_visited(num_v, false);

	// Use a distance map to save time, 09/16/2018
	vector< vector<smalli> > dist_matrix(roots_size);
	for (idi r = 0; r < roots_size; ++r) {
		vector<smalli> &dmr = dist_matrix[r];
		dmr.resize(num_v, SMALLI_MAX);
		idi root_id = r + root_start;
		const IndexType &Lr = L[root_id];
		idi size = Lr.get_size();
		for (idi vi = 0; vi < size; ++vi) {
			idi v = Lr.get_label_ith_v(vi);
			dmr[v] = Lr.get_label_ith_d(vi);
		}
		dmr[r] = 0;
	}
	// End distance map

	// Initialize roots' short_index
	for (inti r_i = 0; r_i < roots_size; ++r_i) {
		idi v = r_i + root_start;
		short_index[v].set_index(r_i, 0);
		short_index[v].roots_last.set_bit(r_i);
		is_active[v] = true;
		is_visited[v] = true;
	}

	// Label the roots first
	roots_labeling(
				G,
				root_start,
				roots_size,
				short_index,
				dist_matrix);

//	smalli iter = 1; // iterator, also the distance for current iteration
	bool stop = false;
	while (!stop) {
//		printf("iter: %u\n", iter);//test
		WallTimer t_can("Candidating");
		stop = true;
		// Push to Candidate
		vector<bool> has_candidate(num_v, false);
		for (idi head = 0; head < num_v; ++head) {
			if (!is_active[head]) {
				continue;
			}
			ShortIndex &head_si = short_index[head];
			idi degree = G.ith_get_out_degree(head);
			inti bound = get_bound(head, root_start, roots_size);
			vector<inti> head_roots = head_si.roots_last.get_all_locs_set(bound);
			for (idi e_i = 0; e_i < degree; ++e_i) {
				idi tail = G.ith_get_edge(head, e_i);
				ShortIndex &tail_si = short_index[tail];
				for (const inti &hi : head_roots) {
					idi x = hi + root_start;
					if (tail < x) {
						break;
					}
					if (
//						x < tail
//						&& !L[tail].is_v_in_label(x) &&
						!tail_si.roots_candidates.is_bit_set(hi)){
						tail_si.roots_candidates.set_bit(hi);
					}
				}
			}
			head_si.roots_last.unset_all();
			is_active[head] = false;
		}

		t_can.print_runtime();
		time_can += t_can.get_runtime();
		WallTimer t_add("Adding");

		// Traverse Candidate then add to short index
		// v : root_id
		// r : cand
		for (idi v = 0; v < num_v; ++v) {
			ShortIndex &v_si = short_index[v];
			inti bound = get_bound(v, root_start, roots_size);
			vector<inti> candidates = v_si.roots_candidates.get_all_locs_set(bound);
			if (0 == candidates.size()) {
				continue;
			}
			for (const inti &cand : candidates) {
//				weighti d = query(
//								v,
//								r + root_start,
//								L,
//								short_index,
//								root_start,
//								roots_size);
				// The new query based on the distance matrix 09/16/2018
				// Query based on the old Labels.
				weighti d = WEIGHTI_MAX;
				const IndexType &Lv = L[v];
				idi size = Lv.get_size();
				for (idi vi = 0; vi < size; ++vi) {
					idi v_l = Lv.get_label_ith_v(vi);
					if (SMALLI_MAX == dist_matrix[cand][v_l]) {
						continue;
					}
					weighti q_d = Lv.get_label_ith_d(vi) + dist_matrix[cand][v_l];
					if (q_d < d) {
						d = q_d;
					}
				}
				// Query based on short index in this batch.
//				const ShortIndex &SIv = short_index[v];
//				inti bound = get_bound(v, root_start, roots_size);
				vector<inti> roots_v = v_si.roots_indicator.get_all_locs_set(bound);
				for (const inti &rv : roots_v) {
					if (SMALLI_MAX == dist_matrix[cand][rv]) {
						continue;
					}
					weighti q_d = v_si.roots_distances[rv] + dist_matrix[cand][rv];
					if (q_d < d) {
						d = q_d;
					}
				}
				// End the new query
				if (iter < d) {
					v_si.set_index(cand, iter);
					v_si.roots_last.set_bit(cand);
					is_active[v] = true;

					// for the new version, this is impossible as the roots are already labeled in advance.
//					idi root_id_v = v - root_start;
//					if (root_start <= v && root_id_v < roots_size) {
//						dist_matrix[root_id_v][cand] = iter;
//					}


					if (stop) {
						stop = false;
					}
				}
			}
			v_si.roots_candidates.unset_all();
		}
		++iter;

		t_add.print_runtime();
		time_add += t_add.get_runtime();
	}
	// add short_index to L
	double time_update = 0;
	WallTimer t_update("Updating");
	for (idi v = 0; v < num_v; ++v) {
		const ShortIndex &v_si = short_index[v];
		IndexType &Lv = L[v];
		inti bound = get_bound(v, root_start, roots_size);
		vector<inti> indicators = v_si.roots_indicator.get_all_locs_set(bound);
		if (0 == indicators.size()) {
			continue;
		}
		for (const inti &i : indicators) {
			idi u = i + root_start;
			Lv.add_label_seq(u, v_si.roots_distances[i]);
		}
	}
	t_update.print_runtime();
	time_update += t_update.get_runtime();

	double total_time = time_can + time_add + time_update;
	printf("Candidating time: %f (%f%%)\n", time_can, time_can / total_time * 100);
	printf("Adding time: %f (%f%%)\n", time_add, time_add / total_time * 100);
	printf("Updating time: %f (%f%%)\n", time_update, time_update / total_time * 100);
}

void VertexCentricPLL::construct(const Graph &G, const vector<idi> &rank2id)
{
	// Initialization to (v, 0) for every v
	idi num_v = G.get_num_v();
	L.resize(num_v);
	const inti bit_array_size = 64;
//	vector<ShortIndex> short_index(num_v, roots_size);
	idi remainer = num_v % bit_array_size;
	idi b_i_bound = num_v - remainer;
	for (idi b_i = 0; b_i < b_i_bound; b_i += bit_array_size) {
		printf("b_i: %llu\n", b_i);//test
		root_batch(
				G,
				b_i,
				bit_array_size,
				bit_array_size,
				L);
	}
	if (remainer != 0) {
		printf("b_i: %llu\n", b_i_bound);//test
		root_batch(
				G,
				b_i_bound,
				remainer,
				bit_array_size,
				L);
	}

	switch_labels_to_old_id(rank2id);
	print();//test
}

void VertexCentricPLL::switch_labels_to_old_id(const vector<idi> &rank2id)
{
	idi num_v = rank2id.size();
	vector<IndexType> new_L(num_v);
	for (idi r = 0; r < num_v; ++r) {
		idi v = rank2id[r];
		const IndexType &Lr = L[r];
		IndexType &Lv = new_L[v];
		idi size = Lr.get_size();
		for (idi li = 0; li < size; ++li) {
			idi l = Lr.get_label_ith_v(li);
			idi new_l = rank2id[l];
			Lv.add_label_seq(new_l, Lr.get_label_ith_d(li));
		}
	}
	L = new_L;
}

weighti VertexCentricPLL::query(
							idi u,
							idi v,
							const vector<IndexType> &index,
							const vector<ShortIndex> &short_index,
							idi root_start,
							inti roots_size)
{
	// Traverse the index
	const IndexType &Lu = index[u];
	const IndexType &Lv = index[v];
	weighti dist = WEIGHTI_MAX;
	idi lu_size = Lu.get_size();
	idi lv_size = Lv.get_size();
	idi iu = 0;
	idi iv = 0;
	while (iu < lu_size && iv < lv_size) {
		idi u_e = Lu.get_label_ith_v(iu);
		idi v_e = Lv.get_label_ith_v(iv);
		if (u_e == v_e) {
			weighti d = Lu.get_label_ith_d(iu) + Lv.get_label_ith_d(iv);
			if (d < dist) {
				dist = d;
			}
			++iu;
			++iv;
		} else if (u_e < v_e) {
			++iu;
		} else {
			++iv;
		}
	}

	// Traverse the short index.
	const ShortIndex &Su = short_index[u];
	const ShortIndex &Sv = short_index[v];
	inti bound = get_bound(u, root_start, roots_size);
	vector<inti> u_roots = Su.roots_indicator.get_all_locs_set(bound);
	bound = get_bound(v, root_start, roots_size);
	vector<inti> v_roots = Sv.roots_indicator.get_all_locs_set(bound);
	inti iu_size = u_roots.size();
	inti iv_size = v_roots.size();
	inti i_u = 0;
	inti i_v = 0;
	while (i_u < iu_size && i_v < iv_size) {
		idi u_e = u_roots[i_u];
		idi v_e = v_roots[i_v];
		if (u_e == v_e) {
			weighti d = Su.roots_distances[u_e] + Sv.roots_distances[v_e];
			if (d < dist) {
				dist = d;
			}
			++i_u;
			++i_v;
		} else if (u_e < v_e) {
			++i_u;
		} else {
			++i_v;
		}
	}
	return dist;
}

void VertexCentricPLL::print()
{
	for (idi v = 0; v < L.size(); ++v) {
		const IndexType &Lv = L[v];
		idi size = Lv.get_size();
		printf("Vertex %llu (Size %llu):", v, size);
		for (idi i = 0; i < size; ++i) {
			printf(" (%llu, %d)", Lv.get_label_ith_v(i), Lv.get_label_ith_d(i));
			fflush(stdout);
		}
		puts("");
	}
}

///////////////////////////////////////////////////////////////////////
//// For index by map
//void VertexCentricPLL::construct(const Graph &G, const vector<idi> &rank)
//{
//	L.resize(G.get_num_v());
//	// Initialization to (v, 0) for every v
//	idi num_v = G.get_num_v();
//	vector< vector< pair<idi, weighti> > > llast(num_v);
//	for (idi v = 0; v < num_v; ++v) {
//		L[v].add_label_seq(v, 0);
//		llast[v].push_back(make_pair(v, 0));
//	}
////	printf("iter: 0\n");//test
////	print();//test
//
//	weighti iter = 1;
////	weighti last_iter = iter - 1;
//	bool stop = false;
//	vector< unordered_map<idi, weighti> > C(num_v); // candidate set C
//
//
//	double time_can = 0; // test
//	double time_add = 0; // test
//
//	while (!stop) {
//
//		stop = true;
//		WallTimer t_can("Candidating");
//		for (idi v = 0; v < num_v; ++v) {
//			auto &llv = llast[v];
//			if (0 == llv.size()) {
//				continue;
//			}
//			idi degree = G.ith_get_out_degree(v);
//			for (idi e_i = 0; e_i < degree; ++e_i) {
//				idi u = G.ith_get_edge(v, e_i);
//				for (const auto &ll : llv) {
//					idi x = ll.first;
//					weighti dist = ll.second;
//					if (rank[x] < rank[u]
//						&& !L[u].is_v_in_label(x)) {
//						const auto &tmp_l = C[u].find(x);
//						if (tmp_l == C[u].end()) {
//							C[u][x] = dist + 1;
//						}
//					}
//				}
//			}
//			llv.clear();
//		}
//
//		time_can += t_can.get_runtime();
//		t_can.print_runtime();
//		WallTimer t_add("Adding");
//		for (idi v = 0; v < num_v; ++v) {
//			for (const auto &p : C[v]) {
//				weighti d = query(v, p.first, G, L);
//				if (p.second < d) { // dist < d
//					L[v].add_label_seq(p.first, p.second);
//					llast[v].push_back(make_pair(p.first, p.second));
//					if (true == stop) {
//						stop = false;
//					}
//				}
//			}
//			C[v].clear();
//		}
//		time_add += t_add.get_runtime();
//		t_add.print_runtime();
//		printf("iter: %d\n", iter);//test
////		print();//test
////		last_iter = iter;
//		++iter;
//	}
//
//	printf("Time_can: %f (%f%%)\n", time_can, time_can/(time_can + time_add) * 100);
//	printf("Time_add: %f (%f%%)\n", time_add, time_add/(time_can + time_add) * 100);//test
//}
//
//
//weighti VertexCentricPLL::query(
//							idi u,
//							idi v,
//							const Graph &G,
//							const vector<IndexType> &index)
//{
//	const IndexType &Lu = index[u];
//	const IndexType &Lv = index[v];
//	weighti dist = WEIGHTI_MAX;
//
//	auto iu = Lu.get_label_begin();
//	auto iv = Lv.get_label_begin();
//	auto iu_end = Lu.get_label_end();
//	auto iv_end = Lv.get_label_end();
//
//	while (iu != iu_end && iv != iv_end) {
//		if (iu->first == iv->first) {
//			weighti d = iu->second + iv->second;
//			if (d < dist) {
//				dist = d;
//			}
//			++iu;
//			++iv;
//		} else if (iu->first < iv->first) {
//			++iu;
//		} else {
//			++iv;
//		}
//	}
//
//	return dist;
//}
///////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
// For index by vector, with naive implementation 09/07/2018
//void VertexCentricPLL::construct(const Graph &G, const vector<idi> &rank)
//{
//	L.resize(G.get_num_v());
//	// Initialization to (v, 0) for every v
//	idi num_v = G.get_num_v();
//	for (idi v = 0; v < num_v; ++v) {
//		L[v].add_label_seq(v, 0);
//	}
////	printf("iter: 0\n");//test
////	print();//test
//
//	weighti iter = 1;
//	weighti last_iter = iter - 1;
//	bool stop = false;
//	vector< unordered_map<idi, weighti> > C(num_v); // candidate set C
//
//	double time_can = 0; // test
//	double time_add = 0; // test
//
//	while (!stop) {
//
//		stop = true;
//		WallTimer t_can("Candidating");
//		for (idi v = 0; v < num_v; ++v) {
//			IndexType &lv = L[v];
//			if (last_iter != lv.get_last_label_d()) {
//				continue;
//			}
//			idi degree = G.ith_get_out_degree(v);
//			for (idi e_i = 0; e_i < degree; ++e_i) {
//				idi u = G.ith_get_edge(v, e_i);
//				idi last = lv.get_size() - 1;
//				idi x = lv.get_label_ith_v(last);
//				weighti dist = lv.get_label_ith_d(last);
//				while (dist == last_iter) {
//					if (rank[x] < rank[u]
//						&& !L[u].is_v_in_label(x)) {
//						const auto &tmp_l = C[u].find(x);
//						if (tmp_l == C[u].end()) {
//							C[u][x] = dist + 1; // insert (x, dist + 1) to C[u]
//						}
//					}
//					if (last == 0) {
//						break;
//					} else {
//						--last;
//					}
//					x = lv.get_label_ith_v(last);
//					dist = lv.get_label_ith_d(last);
//				}
//			}
//		}
//
//		time_can += t_can.get_runtime();
//		t_can.print_runtime();
//		WallTimer t_add("Adding");
//		for (idi v = 0; v < num_v; ++v) {
//			for (const auto &p : C[v]) {
//				weighti d = query(v, p.first, G, L);
//				if (p.second < d) { // dist < d
//					L[v].add_label_seq(p.first, p.second);
//					if (true == stop) {
//						stop = false;
//					}
//				}
//			}
//			C[v].clear();
//		}
//		time_add += t_add.get_runtime();
//		t_add.print_runtime();
//		printf("iter: %d\n", iter);//test
////		print();//test
//		last_iter = iter;
//		++iter;
//	}
//
//	print();//test
//	printf("Time_can: %f (%f)\n", time_can, time_can/(time_can + time_add));
//	printf("Time_add: %f (%f)\n", time_add, time_add/(time_can + time_add));//test
//}
//
//weighti VertexCentricPLL::query(
//							idi u,
//							idi v,
//							const Graph &G,
//							const vector<IndexType> &index)
//{
//	const IndexType &Lu = index[u];
//	const IndexType &Lv = index[v];
//	weighti dist = WEIGHTI_MAX;
//	unordered_map<idi, weighti> markers;
//	idi label_size = Lu.get_size();
//	for (idi i = 0; i < label_size; ++i) {
//		markers[Lu.get_label_ith_v(i)] = Lu.get_label_ith_d(i);
//	}
//	label_size = Lv.get_size();
//	for (idi i = 0; i < label_size; ++i) {
//		const auto &tmp_l = markers.find(Lv.get_label_ith_v(i));
//		if (tmp_l == markers.end()) {
//			continue;
//		}
//		int d = tmp_l->second + Lv.get_label_ith_d(i);
//		if (d < dist) {
//			dist = d;
//		}
//	}
//	return dist;
//}
//
//void VertexCentricPLL::print()
//{
//	for (idi v = 0; v < L.size(); ++v) {
//		const IndexType &Lv = L[v];
//		idi size = Lv.get_size();
//		printf("Vertex %llu (Size %llu):", v, size);
////		for (auto l = Lv.get_label_begin(); l != Lv.get_label_end(); ++l) {
////			printf(" (%llu, %d)", l->first, l->second);
////			fflush(stdout);
////		}
//		for (idi i = 0; i < size; ++i) {
//			printf(" (%llu, %d)", Lv.get_label_ith_v(i), Lv.get_label_ith_d(i));
//			fflush(stdout);
//		}
//		puts("");
//	}
//}
////////////////////////////////////////////////////////////////

//// Rank according to degrees
//vector<idi> VertexCentricPLL::make_rank(const Graph &G)
//{
//	vector< pair<idi, idi> > degree2id;
//	idi num_v = G.get_num_v();
//	for (idi v = 0; v < num_v; ++v) {
//		degree2id.push_back(make_pair(G.ith_get_out_degree(v), v));
//	}
//	stable_sort(degree2id.rbegin(), degree2id.rend());
//	vector<idi> rank(num_v);
//	for (idi r = 0; r < num_v; ++r) {
//		rank[degree2id[r].second] = r + 1;
//	}
//	return rank;
//}

// End class VertexCentricPLL

} // namespace PADO


//		{
//			printf("v %llu:\n", v);
//			const ShortIndex &si = short_index[v];
//			vector<inti> indicator = si.roots_indicator.get_all_locs_set(roots_size);
//			printf("indicator:");
//			for (const auto &i : indicator) {
//				printf(" %u", i);
//			}
//			puts("");
//
//			printf("distances:");
//			for (inti i = 0; i < roots_size; ++i) {
//				printf(" %u", si.roots_distances[i]);
//			}
//			puts("");
//
//			vector<inti> cand = si.roots_candidates.get_all_locs_set(roots_size);
//			printf("candidates:");
//			for (const auto &i : cand) {
//				printf(" %u", i);
//			}
//			puts("");
//
//			vector<inti> last = si.roots_last.get_all_locs_set(roots_size);
//			printf("last:");
//			for (const auto &i : last) {
//				printf(" %u", i);
//			}
//			puts("");
//		}


#endif /* INCLUDES_PADO_H_ */