// Structure for the type of label.
// For weighted graph, the previous 3-level-index label structure is
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
		labels table labels_table,
		candidate distance table cand_dist_table,
		has_candidates_queue)
{
	// Traverse all neighbors of vertex v
	for (vertex w = every neighbor of v) {
		weighti dist_v_w = weight(v, w); // weight of edge (v, w).
		// Check all labels of vertex v (TODO: should be only new inserted labels)
		for (idi r = 0; r < w; ++r) {
			// r should only access to a lower rank vertex.
			weighti dist_r_v = labels_table[v][r]; // distance of (r, v)
			if (dist_r_v == INF) {
				// Vertex v does not have the label r.
				continue;
			}
			weighti tmp_dist = dist_r_v + dist_v_w;
			if (tmp_dist < dist_table[r][w] && tmp_dist < cand_dist_table[w][r]) {
				// Mark r as a candidate of w
				cand_dist_table[w][r] = tmp_dist;
				dist_table[r][w] = tmp_dist;
				has_candidates_queue.enqueue(w); //
					// TODO: Here needs a bitmap to ensure that w is only added once.
			}
		}
	}
}

// Function: return a distance (less than INF) if shortest distance is covered by other path
// with the returned distance, return INF if the shortest distance is obtained at first time.
weighti distance_query(
		vertex v,
		candidate c,
		distance table dist_table,
		labels table labels_table,
		candidate distance table cand_dist_table,
		number of vertices num_v,
		distance tmp_dist_v_c)
{
	// Traverse all available hops of v, to see if they reach c
	for (idi hop_i = 0; hop_i < c, ++hop_i) {
		if (labels_table[v][hop_i] == INF || labels_table[c][hop_i] == INF) {
			continue;
		}
		weighti label_v_c = labels_table[v][hop_i] + labels_table[c][hop_i];
		if (label_v_c <= tmp_dist_v_c) {
			return label_v_c;
		}

		if (cand_dist_table[v][hop_i] == INF || cand_dist_table[c][hop_i] == INF) {
			continue;
		}
		weighti cand_dist_v_c = cand_dist_table[v][hop_i] + cand_dist_table[c][hop_i];
		if (cand_dist_v_c <= tmp_dist_v_c) {
			return cand_dist_v_c;
		}
	}

	return INF;
}

// Function:vertex s noticed that it received a distance between t but the distance is longer than what s can get,
// so s starts to send its shorter distance between s and t to all its neighbors. According to the distance sent,
// all active neighbors will update their distance table elements and reset their label table elements. The 
// process continue untill no active vertices.
void sending_back(
		vertex s,
		root r,
		distance table dist_table,
		labels table labels_table)
{
	A queue for active vertices of sending back messages vector<idi> active_queue;
		// Vertices in it are active and sends back messages to its neighbors.
	A temporary queue vector<idi> tmp_active_queue;

	active_queue.enqueue(s);
	while (!active_queue.empty()) {
		// Traverse active queue, get every vertex and its distance to the target
		for (every active vertex v in active_queue) {
			distance dist_r_v = dist_table[r][v];
			// Traverse all neighbors of vertex v
			for (every neighbor w of v) {
				if (w < r) {
					// Neighbors are ordered by ranks from low to high
					break;
				}
				distance tmp_dist_r_w = dist_r_v + weight(v, w);
				if (tmp_dist_r_w <= dist_table[r][w]) {
					dist_table[r][w] = tmp_dist_r_w; // Update distance table
					labels_table[w][r] = INF; // Reset label table
					tmp_active_queue.enqueue(w);
						// TODO: Here need a bitmap to make sure 
				}
			}
		}
		active_queue.clear();
		active_queue.swap(tmp_active_queue);
	}
}

// Function: some distances in the labels table are not necessary. Need to set them INF.
// The function should be called after the labels_table is finished.
void filter_out_labels_table_naive(
					distance table dist_table,
					labels table labels_table)
{
	for (idi v = 0; v < num_v; ++v) {
		for (idi r = 0; r < v; ++r) {
			if (INF == labels_table[v][r]) {
				continue;
			}
			weighti label_dist_v_r = labels_table[v][r];
			// Check the distance between v and r
			for (idi hop_i = 0; hop_i < r; ++hop_i) {
				if (INF == dist_table[hop_i][v] || INF == dist_table[hop_i][r]) {
					continue;
				}
				weighti dist_r_v = dist_table[hop_i][v] + dist_table[hop_i][r];
				if (dist_r_v <= label_dist_v_r) {
					labels_table[v][r] = INF;
				}
			}
		}
	}
}

// Function: vertex-centric labeling for weighted graphs;
// Activate all vertices at the beginning, build a distance table at first, 
// then build the index according to the distance table.
void vertex_centric_labeling(
		Graph G,
		vector<IndexType> &L)
{
	An active queue is vector<idi> active_queue(num_v);
	An queue storing all vertices which have candidates is vector<idi> has_candidates_queue(num_v);
	The distance table is vector< vector<weighti> > dist_table(num_v, vector<weighti>(num_v, INF)); 
		// The distance table is N by N, recording the shortest distance so far from every root to every vertex.
	The label table is vector< vector<weighti> > labels_table(num_v, vector<weighti>(num_v, INF));
		// The label table records label distance for every vertex.
		// TODO: need other data structure to record those last inserted distances in the last iteration, then an active vertex will only send its new inserted distances.
	The distance candidate table is vector< vector<weighti> > cand_dist_table(num_v, vector<weighti>(num_v, INF)); 
		// Temporary distance table, recording in the current iteration the traversing distancefrom a vertex to a root.
		// TODO: The candidate table probably could replaced by a queue and bitmap.

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
		labels_table[v][v] = 0;
	}

	// Active vertices sending messages.
	// Those vertices received messages are put into has_candidates_queue.
	while (!active_queue.empty()) {
		// First stage, sending distances.
		// Traverse all active vertex and send distances
		for (every vertex v in the active_queue) {
			// vertex v send distance messages to all its neighbors;
			// every neighbor gets its candidates.
			sending_message(
					active vertex v,
					Graph G,
					distance table dist_table,
					labels table labels_table,
					candidate distance table cand_dist_table,
					has_candidates_queue);
		}
		active_queue.clear();

		// Second stage, checking candidates.
		// Traverse has_candidates_queue, check all candidates of every vertex
		for (every vertex v in the has_candidates_queue) {
			flag need_activate = false; // Flag: if v should be a new active vertex in the next iteration
			// Traverse all candidates of v
			for (idi c = 0; c < num_v; ++c) {
				weighti tmp_dist_v_c = cand_dist_table[v][c];
				if (dist_v_r == INF) {
					continue;
				}
				cand_dist_table[v][c] = INF; // Reset
				// Distance check for pruning
				weighti query_dist_v_c;
				if (INF == (query_dist_v_c = distance_query(
							vertex v,
							candidate c,
							distance table dist_table,
							labels table labels_table,
							candidate distance table cand_dist_table,
							number of vertices num_v,
							distance tmp_dist_v_c))) {
					// Record the new distance in the label table
					labels_table[v][c] = tmp_dist_v_c;
					need_activate = true;
				} else {
					dist_table[c][v] = query_dist_v_c;
					// First correcting option:
					// v needs to send message back to its neighbor to change potential wrong distance from root c
					sending_back(
							vertex v,
							candidate c,
							distance table dist_table,
							labels table labels_table);
				}
			}
			if (need_activate) {
				active_queue.enqueue(v); // TODO: Here needs a bitmap to ensure v is added only once
			}
		}
		has_candidates_queue.clear();
	}

	// Second correcting option:
//	filter_out_labels_table_naive(
//					distance table dist_table,
//					labels table labels_table);

	/*
	Second, after the message-sending phase, all distances are
	available in the distance table. And according to it, we can build 
	the index.
	*/
	for (idi v = 0; v < num_v; ++v) {
		// To every vertex (with lower rank)
		for (idi r = 0; r < v; ++r) {
			if (labels_table[v][r] != INF) {
				Insert label (r, labels_table[v][r]) into L[v];
			}
		}
		Insert label (v, 0) into L[v];
	}
}


int main()
{
	Get graph data Graph G; // Smaller vertex ID has higher rank (vertex 0 is highest ranked)
	The empty index vector<IndexType> L;

	// labeling
	vertex_centric_labeling(
			Graph G,
			vector<IndexType> &L);
	return 0;
}
