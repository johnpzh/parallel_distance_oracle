#include <iostream>

using namespace std;

bool test_redundant(ID v_a, ID, v_b)
{
	set<ID> set_va;
	set<ID> set_vb;
	
	set_va.insert(v_a);
	for (every neighbor vn of v_a) {
		set_va.insert(vn);
	}
	
	set_vb.insert(v_b);
	for (every neighbor vn of v_b) {
		set_vb.insert(vn);
	}
	
	return set_va == set_vb;
}

void eliminate_equivalent_vertices()
{
	Read the graph G and Make G is a undirected graph;
	A flag vector is_redundant(G.num_v_, false);
	
	num_threads = omp_get_num_threads();
	batch_size = (num_v_ + num_threads - 1) / num_threads;
	The starting locations vector<ID> start_locations(num_threads);
	for (i_t = 0; i_t < num_threads; ++i_t) {
		start_locations[i_t] = i_t * batch_size;
	} 
	
	#pragma omp parallel for
	for (i_t = 0; i_t < num_threads; ++i_t) {
		v_i_start = start_locations[i_t];
		v_i_bound = i_t != num_threads - 1 ? 
					start_locations[i_t + 1] :
					num_v_;
		for (v_i = v_i_start; v_i < v_i_bound; ++v_i) {
			if (is_redundant[v_i]) continue;
			a_set = adjacency_list[v_i];
			for (b_i = v_i + 1; b_i < v_i_bound; ++b_i) {
				if (is_redundant[b_v]) continue;
				b_set =adjacency_list[b_i];
				if (a_set == b_set) {
					is_redundant[b_i] = true;
				}
			}
		}				
	}
	
	for (v in [0, G.num_v_)) {
		if (is_redundant[v]) continue;
		for (v1 in [v + 1, G.num_v_)) {
			if (is_redundant[v1]) continue;
			if (test_equivalent(v, v1)) {
				is_redundant[v1] = true;
			}
		}
	}
	
	An edge list vector<pair<ID, ID>> edge_list;
	num_v_ = 0;
	num_e_ = 0;
	for (edge (h, e) reading from G) {
		if (is_redundant[h] || is_redundant[e]) {
			continue;
		}
		num_v_ = max(num_v_, max(h, e) + 1);
		edge_list.emplace_back(h, e);
	}
	num_e_ = edge_list.size();
	
	New graph G1;
	Write num_v_ and num_e_ into G1;
	Write edge_list into G1;
}

int main(int argc, char *argv[]) {
	eliminate_equivalent_vertices();
	
	return 0;
}