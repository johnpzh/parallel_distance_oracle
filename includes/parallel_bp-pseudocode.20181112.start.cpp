inline void VertexCentricPLL::bit_parallel_labeling(
			const Graph &G,
			vector<IndexType> &L,
			uint8_t *used_bp_roots) // CAS needs array
			//vector<bool> &used_bp_roots
{
	idi num_v = G.get_num_v();
	idi num_e = G.get_num_e();

	if (num_v <= BITPARALLEL_SIZE) {
		// Sequential version
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
	} else {
		// Parallel version
		vector< vector<smalli> > tmp_d(num_v, vector<smalli>(BITPARALLEL_SIZE, SMALLI_MAX));
		vector< vector<std::pair<uint64_t, uint64_t> > > tmp_s(num_v, vector<std::pair<uint64_t, uint64_t> >(BITPARALLEL_SIZE, make_pair(0, 0))); // first is S_r^{-1}, second is S_r^{0}
		vector<idi> que(num_v); // active queue
		// Select roots and their neighbors
#pragma omp parallel for
		for (inti r_i = 0; r_i < BITPARALLEL_SIZE; ++r_i) {
			used_bp_roots[r_i] = true;
			que[que_h++] = r_i;
			tmp_d[r_i][r_i] = 0;

			inti ns = 0;
			// Select neighbors
			idi i_bound = G.vertices[r_i] - 1;
			idi i_start = i_bound + G.out_degrees[r_i];
			for (idi i = i_start; i > i_bound; --i) {
				idi v = G.out_edges[i];
				if (!used_bp_roots[v]) {
					//used_bp_roots[v] = true;
					if (CAS(used_bp_roots + v, (uint8_t) 0, (uint8_t) 1)) {
						// Algo3:line4: for every v in S_r, (dist[v], S_r^{-1}[v], S_r^{0}[v]) <- (1, {v}, empty_set)
						que[que_h++] = v; // Need parallel enqueue
						tmp_d[v][r_i] = 1;
						tmp_s[v][r_i].first = 1ULL << ns;
						if (++ns == 64) break;
					}
				}
			}
		}

		// Process the queue
		smalli d = 0;
		while (que is not empty) {
#pragma omp parallel for
			for (every v in que) {
				// Traverse v's neighbors
				for (every neighbor tv of v) {
					for (inti r_i = 0; r_i < BITPARALLEL_SIZE; ++r_i) {
						if (d > tmp_d[tv][r_i]) {
							;
						}
						else if (d == tmp_d[tv][r_i]) {
							if (v < tv) { // ??? Why need v < tv !!! Because it's a undirected graph.
								tmp_s[v][r_i].second  |= tmp_s[tv][r_i].first;
								__sync_or_and_fetch(&tmp_s[tv][r_i].second, &tmp_s[v][r_i].first)
	//							tmp_s[tv][r_i].second |= tmp_s[v][r_i].first;
							}
						} else { // d < tmp_d[tv]
							if (CAS(&tmp_d[tv][r_i], SMALLI_MAX, d + 1)) { // tmp_d[tv] = td
	//							tmp_que[offsets_tmp_queue[tmp_que_i] + sizes_tmp_que[tmp_que_i]++] = tv;
								que[que_h++] = tv; // Need parallel enqueue
							}
	//						if (tmp_d[tv][r_i] == SMALLI_MAX) {
	//							que[que_h++] = tv; // Need parallel enqueue
	//							tmp_d[tv][r_i] = d + 1;
	//						}
							__sync_or_and_fetch(&tmp_s[tv][r_i].first, &tmp_s[v][r_i].first)
							__sync_or_and_fetch(&tmp_s[tv][r_i].second, &tmp_s[v][r_i].second)
	//						tmp_s[tv][r_i].first  |= tmp_s[v][r_i].first;
	//						tmp_s[tv][r_i].second |= tmp_s[v][r_i].second;
						}
					}
				}
			}

			++d;
		}

		// Record into Label L
		for (idi v = 0; v < num_v; ++v) {
			for (inti r_i = 0; r_i < BITPARALLEL_SIZE; ++r_i) {
				L[v].bp_dist[r_i] = tmp_d[v][r_i];
				L[v].bp_sets[r_i][0] = tmp_s[v][r_i].first; // S_r^{-1}
				L[v].bp_sets[r_i][1] = tmp_s[v][r_i].second & ~tmp_s[v][r_i].first; // Only need those r's neighbors who are not already in S_r^{-1}
			}
		}
	}
}
