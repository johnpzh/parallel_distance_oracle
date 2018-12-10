// Structure for the type of label
// As for weighted graph, the previous 3-level-index label structure is
// not helpful, because the distance is not the same in every iteration.
struct IndexType {
	vector<idi> vertices;
	vector<weighti> distances;
};

// Function: vertex v send distance messages to all its neighbors.
void sending_message(
		active vertex v,
		Graph G,
		distance table dist_table,
		candidate distance table cand_dist_table,
		has_candidates_queue)
{
	// Traverse all neighbors of vertex v
	for (vertex w = every neighbor of v) {
		weighti dist_v_w = weight(v, w); // weight of edge (v, w).
		// Check every source
		for (idi r = 0; r < num_v; ++r) {
			if (w < r) {
				// neighbors are ranked from low to high, r can only access to a lower rank vertex
				break;
			}
			weighti dist_r_v = dist_table[r][v]; // distance of (r, v)
			if (dist_r_v == INF) {
				// vertex r and vertex v has no path between them yet.
				continue;
			}
			weighti tmp_dist = dist_r_v + dist_v_w;
			if (tmp_dist < dist_table[r][w]) {
				// Mark r as a candidate of w
				cand_dist_table[w][r] = tmp_dist;
				has_candidates_queue.enqueue(w); // Here needs a bitmap to ensure that w is only added once.
			}
		}
	}
}

// Function: return false if shortest distance is covered by other path, return true if the shortest distance is obtained at first time.
bool distance_query(
		candidate c,
		vertex v,
		number of vertices num_v,
		distance dist_c_v
		)
{
	// Traverse all available hops of v, to see if they reach c
	for (idi hop_i = 0; hop_i < num_v, ++hop_i) {
		if (dist_table[v][hop_i] == INF) {
			continue;
		}
		if (dist_table[v][hop_i] + dist_table[c][hop_i] <= dist_c_v) {
			return false;
		}
	}

	return true;
}

void vertex_centric_labeling(
		Graph G,
		vector<IndexType> &L)
{
	An active queue is vector<idi> active_queue(num_v);
	An queue storing all vertices which have candidates is vector<idi> has_candidates_queue(num_v);
	The distance table is vector< vector<weighti> > dist_table(num_v, vector<weighti>(num_v, INF)); // The distance table is N by N, recording the shortest distance so far from every source to every target.
	The distance candidate table is vector< vector<weighti> > cand_dist_table(num_v, vector<weighti>(num_v, INF));

	/*
	First, use vertex-centric method, all vertices are sending messages
	to neighbors and updates their distances in the distance table. 
	The distance table is a temporary data structure for building the 
	index later. Here they do not add anything into the index.
	*/
	// Activate all vertices
	for (idi v = 0; v < num_v; ++v) {
		active_queue.enqueue(v);
		// Initialize the distance for every vertex itself
		dist_table[v][v] = 0;
	}

	// Active vertices sending messages.
	// Those vertices received messages are put into has_candidates_queue.
	while (!active_queue.empty()) {
		for (every vertex v in the active_queue) {
			// vertex v send distance messages to all its neighbors;
			// every neighbor get its candidates.
			sending_message(
					active vertex v,
					Graph G,
					distance table dist_table,
					candidate distance table cand_dist_table,
					has_candidates_queue);
		}
		active_queue.clear();
		// Traverse has_candidates_queue, check all candidates of every vertex
		for (every vertex v in the has_candidates_queue) {
			flag need_activate = false; // Flag: if v is a new active vertex in the next iteration
			// Traverse all candidates of v
			for (idi r = 0; r < num_v; ++r) {
				if (cand_dist_table[v][r] == INF) {
					continue;
				}
				// Distance check for pruning
				if (distance_query(
							candidate c,
							vertex v,
							number of vertices num_v,
							distance dist_c_v)) {
					// Needs to update distance table
					dist_table[r][v] = cand_dist_table[v][r];
					cand_dist_table[v][r] = INF; // Reset
					need_activate = true;
				}
			}
			if (need_activate) {
				active_queue.enqueue(v); // Here needs a bitmap to ensure v is added only once
			}
		}
		has_candidates_queue.clear();
	}

	/*
	Second, after the message-sending phase, all distances are
	available in the distance table. And according to it, we can build 
	the index.
	*/
	for (idi v = 0; v < num_v; ++v) {
		Insert (v, 0) into L[v];
		for (idi w = v + 1; w < num_v; ++w) {
			if (dist_table[v][w] != INF) {
				Insert (w, dist_table[v][w]) into L[w];
			}
		}
	}
}

//// Deprecated. It's just vertex-centric multi-source Dijkstra
//void vertex_centric_labeling_deprecated(
//		Graph G,
//		vector<IndexType> &L)
//{
//	An active queue is vector<idi> active_queue(num_v);
//	An temporary active queue for every iteration is vector<idi> tmp_active_queue(num_v);
//	The distance table is vector< vector<weighti> > dist_table(num_v, vector<weighti>(num_v, INF)); // The distance table is N by N, recording the shortest distance so far from every source to every target.
//
//	/*
//	First, use vertex-centric method, all vertices are sending messages
//	to neighbors and updates their distances in the distance table. 
//	The distance table is a temporary data structure. Here they do not 
//	add anything into the index.
//	*/
//	// Activate all vertices
//	for (idi v = 0; v < num_v; ++v) {
//		active_queue.enqueue(v);
//		// Initialize the distance for every vertex itself
//		dist_table[v][v] = 0;
//	}
//
//	// Active vertices sending messages
//	while (!active_queue.empty()) {
//		for (every vertex v in the active_queue) {
//			// Check every source
//			for (idi r = 0; r < num_v; ++r) {
//				weighti dist_r_v = dist_table[r][v];
//				if (dist_r_v == INF) {
//					// vertex r and vertex v has no path between them yet.
//					continue;
//				}
//				// Traverse all neighbors of vertex v
//				for (vertex w = every neighbor of v) {
//					if (w < r) {
//						// neighbors are ranked from low to high, r can only access to a lower rank vertex
//						break;
//					}
//					weighti tmp_dist = dist_r_v + weight(v, w);
//					if (tmp_dist < dist_table[r][w]) {
//						// Update distance from r to w
//						dist_table[r][w] = tmp_dist;
//						tmp_active_queue.enqueue(w);
//					}
//				}
//			}
//		}
//		active_queue.swap(tmp_active_queue);
//	}
//
//	/*
//	Second, after the message-sending phase, all distances are
//	availabel in the distance table. And according to it, we can build 
//	the index.
//	*/
//	for (idi v = 0; v < num_v; ++v) {
//		Insert (v, 0) into L[v];
//		for (idi w = v + 1; w < num_v; ++w) {
//			if (dist_table[v][w] != INF) {
//				Insert (w, dist_table[v][w]) into L[w];
//			}
//		}
//	}
//}
