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

namespace PADO {

class VertexCentricPLL {
private:
	vector<IndexType> L;
	void construct(const Graph &G, const vector<idi> &rank);
public:
	VertexCentricPLL() = default;
	VertexCentricPLL(const Graph &G, const vector<idi> &rank);

	weighti query(idi u, idi v, const Graph &G, const vector<IndexType> &index);
	static vector<idi> make_rank(const Graph &G);
	void print();


}; // class VertexCentricPLL

VertexCentricPLL::VertexCentricPLL(const Graph &G, const vector<idi> &rank)
{
	construct(G, rank);
}

// For index by vector, with batch and distance array, 09/09/2018
//void VertexCentricPLL::root_batch(const Graph &G, )

void VertexCentricPLL::construct(const Graph &G, const vector<idi> &rank)
{
	L.resize(G.get_num_v());

	// Initialization to (v, 0) for every v
	idi num_v = G.get_num_v();
	const idi roots_size = 64;
	idi remaining = num_v % roots_size;
	idi b_i_bound = num_v - remaining;
	for (idi b_i = 0; b_i < b_i_bound; b_i += roots_size) {
		idi r_i_start = b_i;
		idi r_i_bound = r_i_start + roots_size;
		for (idi r_i = r_i_start; r_i < r_i_bound; ++r_i) {
			L[r_i].add_label_seq(r_i, 0); // initialize the roots
		}
	}
//	for (idi v = 0; v < num_v; ++v) {
//		L[v].add_label_seq(v, 0);
//	}

//	printf("iter: 0\n");//test
//	print();//test

	weighti iter = 1;
	weighti last_iter = iter - 1;
	bool stop = false;
	vector< unordered_map<idi, weighti> > C(num_v); // candidate set C

	double time_can = 0; // test
	double time_add = 0; // test

	while (!stop) {

		stop = true;
		WallTimer t_can("Candidating");
		for (idi v = 0; v < num_v; ++v) {
			IndexType &lv = L[v];
			if (last_iter != lv.get_last_label_d()) {
				continue;
			}
			idi degree = G.ith_get_out_degree(v);
			for (idi e_i = 0; e_i < degree; ++e_i) {
				idi u = G.ith_get_edge(v, e_i);
				idi last = lv.get_size() - 1;
				idi x = lv.get_label_ith_v(last);
				weighti dist = lv.get_label_ith_d(last);
				while (dist == last_iter) {
					if (rank[x] < rank[u]
						&& !L[u].is_v_in_label(x)) {
						const auto &tmp_l = C[u].find(x);
						if (tmp_l == C[u].end()) {
							C[u][x] = dist + 1; // insert (x, dist + 1) to C[u]
						}
					}
					if (last == 0) {
						break;
					} else {
						--last;
					}
					x = lv.get_label_ith_v(last);
					dist = lv.get_label_ith_d(last);
				}
			}
		}

		time_can += t_can.get_runtime();
		t_can.print_runtime();
		WallTimer t_add("Adding");
		for (idi v = 0; v < num_v; ++v) {
			for (const auto &p : C[v]) {
				weighti d = query(v, p.first, G, L);
				if (p.second < d) { // dist < d
					L[v].add_label_seq(p.first, p.second);
					if (true == stop) {
						stop = false;
					}
				}
			}
			C[v].clear();
		}
		time_add += t_add.get_runtime();
		t_add.print_runtime();
		printf("iter: %d\n", iter);//test
//		print();//test
		last_iter = iter;
		++iter;
	}

	printf("Time_can: %f (%f)\n", time_can, time_can/(time_can + time_add));
	printf("Time_add: %f (%f)\n", time_add, time_add/(time_can + time_add));//test
}

weighti VertexCentricPLL::query(
							idi u,
							idi v,
							const Graph &G,
							const vector<IndexType> &index)
{
	const IndexType &Lu = index[u];
	const IndexType &Lv = index[v];
	weighti dist = WEIGHTI_MAX;
	unordered_map<idi, weighti> markers;
	idi label_size = Lu.get_size();
	for (idi i = 0; i < label_size; ++i) {
		markers[Lu.get_label_ith_v(i)] = Lu.get_label_ith_d(i);
	}
	label_size = Lv.get_size();
	for (idi i = 0; i < label_size; ++i) {
		const auto &tmp_l = markers.find(Lv.get_label_ith_v(i));
		if (tmp_l == markers.end()) {
			continue;
		}
		int d = tmp_l->second + Lv.get_label_ith_d(i);
		if (d < dist) {
			dist = d;
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

// Rank according to degrees
vector<idi> VertexCentricPLL::make_rank(const Graph &G)
{
	vector< pair<idi, idi> > degree2id;
	idi num_v = G.get_num_v();
	for (idi v = 0; v < num_v; ++v) {
		degree2id.push_back(make_pair(G.ith_get_out_degree(v), v));
	}
	stable_sort(degree2id.rbegin(), degree2id.rend());
	vector<idi> rank(num_v);
	for (idi r = 0; r < num_v; ++r) {
		rank[degree2id[r].second] = r + 1;
	}
	return rank;
}



} // namespace PADO



#endif /* INCLUDES_PADO_H_ */
