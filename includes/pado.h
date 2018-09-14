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
	uint64_t *bits;
	inti size;

public:
	BitArray(unsigned len);
	BitArray(unsigned len, uint64_t num) : BitArray(len)
	{
		*bits = num;
	}
	~BitArray()
	{
		free(bits);
		size = 0;
	}

	// Execute fun for every set bit in bits (bit array)
	template<typename T> void process_every_bit(T fun); // fun is an expected function
	vector<inti> get_all_locs_set(inti bound) const; // return the vector contains all locations which are set
	inti is_bit_set(inti loc);
	void set_bit(inti loc)
	{
		if (loc >= size) {
			fprintf(stderr, "Error: BitArray::set_bit: loc is larger than size.\n");
			return;
		}
//		if (loc < 64) {
//			*bits |= ((uint64_t) 1 << loc);
//			return;
//		}
		inti remain = loc % 64;
		inti count = loc / 64;
		*((uint64_t *) ((uint8_t *) bits + count * 8)) |= ((uint64_t) 1 << remain);
	}
	void unset_bit(inti loc)
	{
		if (loc >= size) {
			fprintf(stderr, "Error: BitArray::unset_bit: loc is larger than size.\n");
			return;
		}
//		if (loc < 64) {
//			*bits &= (~((uint64_t) 1 << loc));
//			return;
//		}
		inti remain = loc % 64;
		inti count = loc / 64;
		*((uint64_t *) ((uint8_t *) bits + count * 8)) &= (~((uint64_t) 1 << remain));
	}
//	bool is_any_set()
//	{
//		inti b_l = size / 64;
//		for (inti i = 0; i < b_l; ++i) {
//			if (bits[i]) {
//				return true;
//			}
//		}
//		return false;
//	}
	void unset_all()
	{
		memset(bits, 0, size * sizeof(uint64_t));
	}

};

BitArray::BitArray(inti len)
{
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
	if (bound < size) {
		count = bound / 64;
	}
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
					if (0 == *num_8) {
						continue;
					}
					inti offset = (i_8 + i_16 + i_32) * 8 + c * 64;
					for (inti i_1 = 0; i_1 < 8; ++i_1) {
						if (*num_8 & (1 << i_1)) {
//							fun(i_1 + offset);
							locs.push_back(i_1 + offset);
						}
					}
				}
			}
		}
	}
	return locs;
}

inti BitArray::is_bit_set(inti loc)
{
	if (loc >= size) {
		fprintf(stderr, "Error: BitArray::is_bit_set: loc is larger than size.\n");
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

class VertexCentricPLL {
private:
	vector<IndexType> L;
	void construct(const Graph &G, const vector<idi> &rank);
	void root_batch(
			const Graph &G,
			idi root_start,
			inti roots_size,
			vector<IndexType> &L);

	struct ShortIndex {
		BitArray roots_indicator; // if evert root r in the short index
		smalli *roots_distances; // distance to root r
		BitArray roots_candidates; // candidates for the current iteration
		BitArray roots_last; // added roots from the last iteration

		ShortIndex(inti size): roots_indicator(size), roots_candidates(size), roots_last(size)
		{
			roots_distances = (smalli *) malloc(size * sizeof(smalli));
			memset(roots_distances, (uint8_t) -1, size * sizeof(smalli));
		}
		~ShortIndex()
		{
			free(roots_distances);
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
	};

public:
	VertexCentricPLL() = default;
	VertexCentricPLL(const Graph &G, const vector<idi> &rank);

	weighti query(
			idi u,
			idi v,
			const Graph &G,
			const vector<IndexType> &index,
			const vector<ShortIndex> &short_index,
			idi root_start,
			inti roots_size);
//	static vector<idi> make_rank(const Graph &G);
	void print();


}; // class VertexCentricPLL

VertexCentricPLL::VertexCentricPLL(const Graph &G, const vector<idi> &rank)
{
	construct(G, rank);
}

// For index by vector, with batch and distance array, 09/09/2018
void VertexCentricPLL::root_batch(
						const Graph &G,
						idi root_start,
						inti roots_size,
						vector<IndexType> &L)
{
	idi num_v = G.get_num_v();
	vector<ShortIndex> short_index(num_v, roots_size);
	vector<bool> is_active(num_v, false);
	auto get_bound = [&] (idi v) -> inti {
		inti bound = v - root_start;
		if (bound < 0 || roots_size <= bound) {
			bound = roots_size;
		}
		return bound;
	};


	// Initialize roots' short_index
	for (inti r_i = 0; r_i < roots_size; ++r_i) {
		idi v = r_i + root_start;
		short_index[v].set_index(r_i, 0);
		short_index[v].roots_last.set_bit(r_i);
		is_active[v] = true;
	}
	smalli iter = 1; // iterator, also the distance for current iteration
	bool stop = false;
	while (!stop) {
		stop = true;
		// Push to Candidate
		for (idi head = 0; head < num_v; ++head) {
			if (!is_active[head]) {
				continue;
			}
			ShortIndex &head_si = short_index[head];
			idi degree = G.ith_get_out_degree(head);
			inti bound = get_bound(head);
			vector<inti> head_roots = head_si.roots_last.get_all_locs_set(bound);
			for (idi e_i = 0; e_i < degree; ++e_i) {
				idi tail = G.ith_get_edge(head, e_i);
				ShortIndex &tail_si = short_index[tail];
				for (const inti &hi : head_roots) {
					idi x = hi + root_start;
					if (x < tail
						&& !L[tail].is_v_in_label(x)
						&& tail_si.roots_candidates.is_bit_set(hi)){
						tail_si.roots_candidates.set_bit(hi);
					}
				}
			}
			head_si.roots_last.unset_all();
			is_active[head] = false;
		}

		// Traverse Candidate then add to index
		for (idi v = 0; v < num_v; ++v) {
			ShortIndex &v_si = short_index[v];
			inti bound = get_bound(v);
			vector<inti> candidates = v_si.roots_candidates.get_all_locs_set(bound);
			for (const inti &c : candidates) {
				weighti d = query(
								v,
								c + root_start,
								G,
								L,
								short_index,
								root_start,
								roots_size);
				if (iter < d) {
					v_si.set_index(c, iter);
					v_si.roots_last.set_bit(c);
					is_active[v] = true;
					if (stop) {
						stop = false;
					}
				}
			}
			v_si.roots_candidates.unset_all();
		}
		++iter;
	}
	// add short_index to L
	for (idi v = 0; v < num_v; ++v) {
		const ShortIndex &v_si = short_index[v];
		IndexType &Lv = L[v];
		inti bound = get_bound(v);
		vector<inti> indicators = v_si.roots_indicator.get_all_locs_set(bound);
		for (const inti &i : indicators) {
			idi u = i + root_start;
			Lv.add_label_seq(u, v_si.roots_distances[i]);
		}
	}
}

void VertexCentricPLL::construct(const Graph &G, const vector<idi> &rank)
{
	// Initialization to (v, 0) for every v
	idi num_v = G.get_num_v();
	const inti roots_size = 64;
//	vector<ShortIndex> short_index(num_v, roots_size);
	idi remainer = num_v % roots_size;
	idi b_i_bound = num_v - remainer;
	for (idi b_i = 0; b_i < b_i_bound; b_i += roots_size) {
		root_batch(
				G,
				b_i,
				roots_size,
				L);
	}
	if (remainer != 0) {
		root_batch(
				G,
				b_i_bound,
				roots_size,
				L);
	}

	///////////////////////////////////////////////////////////
	// Old version
	// Initialization to (v, 0) for every v
//	idi num_v = G.get_num_v();
//	L.resize(num_v);
//	for (idi v = 0; v < num_v; ++v) {
//		L[v].add_label_seq(v, 0);
//	}
////	printf("iter: 0\n");//test
////	print();//test
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
	// Old version
	/////////////////////////////////////////////////////////////

//	printf("Time_can: %f (%f)\n", time_can, time_can/(time_can + time_add));
//	printf("Time_add: %f (%f)\n", time_add, time_add/(time_can + time_add));//test
}

weighti VertexCentricPLL::query(
							idi u,
							idi v,
							const Graph &G,
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
	inti bound = u - root_start;
	if (bound < 0 || roots_size <= bound) {
		bound = roots_size;
	}
	vector<inti> u_roots = Su.roots_indicator.get_all_locs_set(bound);
	bound = v - root_start;
	if (bound < 0 || roots_size <= bound) {
		bound = roots_size;
	}
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
		printf("Vertex %llu:", v);
//		for (auto l = Lv.get_label_begin(); l != Lv.get_label_end(); ++l) {
//			printf(" (%llu, %d)", l->first, l->second);
//			fflush(stdout);
//		}
		for (idi i = 0; i < size; ++i) {
			printf(" (%llu, %d)", Lv.get_label_ith_v(i), Lv.get_label_ith_d(i));
			fflush(stdout);
		}
		puts("");
	}
}

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

//// For index by vector, with naive implementation 09/07/2018
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
//		printf("Vertex %llu:", v);
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

} // namespace PADO



#endif /* INCLUDES_PADO_H_ */
