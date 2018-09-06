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
#include <limits.h>
#include "globals.h"
#include "graph.h"
#include "index.h"

using std::vector;
using std::unordered_map;

namespace PADO {

class VertexCentricPLL {
private:
	vector<IndexType> L;
	void construct(const Graph &G, const vector<idi> &rank);
public:
	VertexCentricPLL() = default;
	VertexCentricPLL(const Graph &G, const vector<idi> &rank);

	weighti query(idi u, idi v, const Graph &G, const vector<IndexType> &index);
	void print();


}; // class VertexCentricPLL

VertexCentricPLL::VertexCentricPLL(const Graph &G, const vector<idi> &rank)
{
	construct(G, rank);
}

void VertexCentricPLL::construct(const Graph &G, const vector<idi> &rank)
{
	L.resize(G.get_num_v());
	// Initialization to (v, 0) for every v
	idi num_v = G.get_num_v();
	for (idi v = 0; v < num_v; ++v) {
		L[v].add_label_seq(v, 0);
	}
	printf("iter: 0\n");//test
	print();//test


	weighti iter = 1;
	weighti last_iter = iter - 1;
	bool stop = false;
	vector< unordered_map<idi, weighti> > C(num_v); // candidate set C
	while (!stop) {
		stop = true;
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
				idi dist = lv.get_label_ith_d(last);
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
		}
		printf("iter: %d\n", iter);//test
		print();//test
		last_iter = iter;
		++iter;
	}
}

weighti VertexCentricPLL::query(idi u, idi v, const Graph &G, const vector<IndexType> &index)
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
		for (idi i = 0; i < size; ++i) {
			printf(" (%llu, %llu)", Lv.get_label_ith_v(i), Lv.get_label_ith_d(i));
			fflush(stdout);
		}
		puts("");
	}
}

} // namespace PADO



#endif /* INCLUDES_PADO_H_ */
