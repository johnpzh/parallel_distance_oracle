// Structure for the type of label.
// For weighted graph, the previous 3-level-index label structure is
// not helpful, because the distance is not the same in every iteration.
struct IndexType {
	vector<idi> vertices;
	vector<weighti> distances;
};

// Structure of temporary data for every vertex: its candidates (distances) and last inserted distances
struct ShortIndex {
	// Use a queue to store candidates
	vector<idi> candidates_que;
	// Use a array to store distances of candidates; length of roots_size
	vector<weighti> candidates_dists; // record the distances to candidates. If candidates_dists[c] = INF, then c is NOT a candidate
		// The candidates_dists is also used for distance query.
	// Use a queue to store last inserted labels (IDs); distances are stored in vertices_dists.
	vector<idi> last_new_roots;

	// Use a queue to store temporary labels in this batch; use it so don't need to traverse vertices_dists.
	vector<idi> vertices_que; 
	// Use an array to store distances to roots; length of roots_size
	vector<weighti> vertices_dists; // labels_table

	// Usa a queue to record which roots have reached this vertex in this batch.
	// It is used for reset the dists_table
	vector<idi> reached_roots_que;
};

// Function: 1. Initialize dists_table, put in labels of roots;
// 2. Every root r puts (r, 0) into its short_index as the last inserted distances.
// 3. Put roots into active_queue;
void initialize_tables(
		The candidate distances data structure vector<ShortIndex> short_index,
		The distance table vector<weighti> dists_table,
		The active queue vector<idi> active_queue,
		The ID of beginning vertex of this batch idi roots_start,
		The number of vertices in this batch inti roots_size,
		The index vector<IndexType> L)
{
	// Distance Table dists_table
	{
		for (every root r_real_id) {
			// Traverse labels of r
			for (every label (v_id, dist) in L[r_real_id]) {
				dists_table[r_real_id - roots_start][v_id] = dist; // Note the ID transfer
			}
			dists_table[r_real_id - roots_start][r_real_id] = 0; // r to itself
		}
	}

	// Short Index short_index
	{
		for (every root r_real_id) {
			short_index[r_real_id].vertices_que.enqueue(r_real_id - roots_start);
			short_index[r_real_id].vertices_dists[r_real_id - roots_start] = 0;
			short_index[r_real_id].last_new_roots.enqueue(r_real_id - roots_start);
		}
	}

	// Active Queue active_queue
	{
		for (every root r_real_id) {
			active_queue.enqueue(r_real_id);
		}
	}
}
// Function: vertex v send distance messages to all its neighbors.
void send_messages(
		active vertex v,
		The beginning ID of roots roots_start,
		Graph G,
		distance table dists_table,
		the temporary data of candidates vector<ShortIndex> short_index,
		The queue has_candidates_queue,
		The flag array has_candidates)
{
	// Traverse all neighbors of vertex v
	for (vertex w = every neighbor of v) {
		if (w <= roots_start) {
			// Neighbors are sorted from low ranks to high ranks; w needs labels with higher ranks
			break;
		}
		weighti dist_v_w = weight(v, w); // weight of edge (v, w).
		// Check all last inserted labels of vertex v
		for (every label ID r in short_index[v].last_new_roots) {
			// r should only access to a lower rank vertex.
			if (w < r + roots_start) {
				continue;
			}
			weighti dist_r_v = short_index[v].vertices_dists[r]; // distance of (r, v)
			weighti tmp_dist_r_w = dist_r_v + dist_v_w;
			if (tmp_dist_r_w < dists_table[r][w] && tmp_dist_r_w < short_index[w].candidates_dists[r]) {
				// Mark r as a candidate of w
				if (INF == short_index[w].candidates_dists[r]) {
					short_index[w].candidates_que.enqueue(r);
				}
				short_index[w].candidates_dists[r] = tmp_dist_r_w;
				if (INF == dists_table[r][w]) {
					short_index[w].reached_roots_que.enqueue(r);
				}
				dists_table[r][w] = tmp_dist_r_w; // This is for filtering out future longer distance.
				if (!has_candidates[w]) {
					has_candidates[w] = true;
					has_candidates_queue.enqueue(w); //
				}
			}
		}
	}
	short_index[v].last_new_roots.clear();
}

// Function: return a distance (less than INF) if shortest distance is covered by other path
// with the returned distance, return INF if the shortest distance is obtained at first time.
weighti distance_query(
		vertex v,
		candidate c,
		The distance table dists_table,
		The candidate distances data structure vector<ShortIndex> short_index,
		The index L,
		The beginning ID of roots_start,
		distance tmp_dist_v_c)
{
	c_real_id = c + roots_start;
	// Traverse all available hops of v, to see if they reach c
	// 1. Labels in L[v]
	for (every label (r, dist_v_r) in L[v]) {
		if (c_real_id < r || INF == dists_table[c][r]) {
			continue;
		}
		weighti label_dist_v_c = dist_v_r + dists_table[c][r];
		if (label_dist_v_c <= tmp_dist_v_c) {
			return label_dist_v_c;
		}
	}
	// 2. Labels in short_index[v].vertices_que
	for (every label r in short_index[v].vertices_que) {
		if (c_real_id < r) {
			continue;
		}
		if (INF != short_index[c_real_id].vertices_dists[r]) {
			int label_dist_v_c = short_index[v].vertices_dists[r] + short_index[c_real_id].vertices_dists[r];
			if (label_dist_v_c <= tmp_dist_v_c) {
				return label_dist_v_c;
			}
		}
		if (INF != short_index[c_real_id].candidates_dists[r]) {
			int label_dist_v_c = short_index[v].vertices_dists[r] + short_index[c_real_id].candidates_dists[r];
			if (label_dist_v_c <= tmp_dist_v_c) {
				return label_dist_v_c;
			}
		}
	}
	// 3. Labels in short_index[v].candidates_que
	for (every label r in short_index[v].candidates_que) {
		if (c_real_id < r) {
			continue;
		}
		if (INF != short_index[c_real_id].vertices_dists[r]) {
			int label_dist_v_c = short_index[v].candidates_dists[r] + short_index[c_real_id].vertices_dists[r];
			if (label_dist_v_c <= tmp_dist_v_c) {
				return label_dist_v_c;
			}
		}
		if (INF != short_index[c_real_id].candidates_dists[r]) {
			int label_dist_v_c = short_index[v].candidates_dists[r] + short_index[c_real_id].candidates_dists[r];
			if (label_dist_v_c <= tmp_dist_v_c) {
				return label_dist_v_c;
			}
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
		distance table dists_table,
		The candidate distances data structure vector<ShortIndex> short_index)
{
	A queue for active vertices of sending back messages vector<idi> active_queue;
		// Vertices in it are active and sends back messages to its neighbors.
	A flag array vector<bool> is_active(num_v);
	A temporary queue vector<idi> tmp_active_queue;
	A flag array vector<bool> tmp_is_active(num_v);


	active_queue.enqueue(s);
	is_active[s] = true;
	while (!active_queue.empty()) {
		// Traverse active queue, get every vertex and its distance to the target
		for (every active vertex v in active_queue) {
			is_active[v] = false;
			distance dist_r_v = dists_table[r][v];
			// Traverse all neighbors of vertex v
			for (every neighbor w of v) {
				if (w < r) {
					// Neighbors are ordered by ranks from low to high
					break;
				}
				distance tmp_dist_r_w = dist_r_v + weight(v, w);
				if (tmp_dist_r_w <= dists_table[r][w]) {
					dists_table[r][w] = tmp_dist_r_w; // Update distance table
					short_index[w].vertices_dists[r] = INF; // Reset label table
					if (!tmp_is_active[w]) {
						tmp_is_active[w] = true;
						tmp_active_queue.enqueue(w);
					}
				}
			}
		}
		active_queue.clear();
		active_queue.swap(tmp_active_queue);
		is_active.swap(tmp_is_active);
	}
}

// Function: according to the short index, build the index L
void update_index(
		The candidate distances data structure vector<ShortIndex> short_index,
		The index vector<IndexType> L,
		The ID of beginning vertex of this batch idi roots_start)
{
	for (every vertex v) {
		// Traverse the vertices_que in short_index[v]
		for (every r in short_index[v].vertices_que) {
			r_real_id = r + roots_start;
			if (INF != short_index[v].vertices_dists[r]) {
				Insert label (r_real_id, short_index[v].vertices_dists[r]) into L[v];
				short_index[v].vertices_dists[r] = INF; // Reset vertices_dists
			}
		}
	}
}

// Function: reset distance table dists_table
void reset_tables(
		The candidate distances data structure vector<ShortIndex> short_index,
		The index L,
		The Distance Tabel vector< vector<weighti> > dists_table)
{
	// Reset dists_table according to L (old labels)
	for (every root r_real_id) {
		// Traverse labels of r
		for (every label (v_id, dist) in L[r_real_id]) {
			dists_table[r_real_id - roots_start][v_id] = INF; // Note the ID transfer
		}
		dists_table[r_real_id - roots_start][r_real_id] = INF; // r to itself
	}

	// Reset dists_table according to short_index[v].reached_roots_que
	for (every vertex v) {
		for (every r in short_index[v].reached_roots_que) {
			dists_table[r][v] = INF;
		}
		short_index[v].reached_roots_que.clear();
	}
}

// Function: some distances in the labels table are not necessary. Need to set them INF.
// The function should be called after the labels_table is finished.
void filter_out_labels_table_naive(
					distance table dists_table,
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
				if (INF == dists_table[hop_i][v] || INF == dists_table[hop_i][r]) {
					continue;
				}
				weighti dist_r_v = dists_table[hop_i][v] + dists_table[hop_i][r];
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
void vertex_centric_labeling_in_batches(
				The start ID of vertex roots_start; // The ID of the beginning vertex of this batch
				The number of roots in this batch roots_size; // The number of vertices of this batch
				Graph G,
				vector<IndexType> &L)
{
	An active queue is vector<idi> active_queue(num_v);
	A bitmap array vector<bool> is_active(num_v, false); // Flag array: is_active[v] is true means v is in active queue.
	An queue storing all vertices which have candidates is vector<idi> has_candidates_queue(num_v);
	A bitmap array vector<bool> has_candidates(num_v, false); // Flag array: has_candidates[v] is true means v is in has_candidates_queue.
	The the distance table is vector< vector<weighti> > dists_table(roots_size, vector<weighti>(num_v, INF)); 
		// The distance table is roots_sizes by N. 1. record the shortest distance so far from every root to every vertex;
		// 2. The distance buffer, recording label distances of every root. It needs to be initialized every batch by labels of roots.
//	The label table is vector< vector<weighti> > labels_table(num_v, vector<weighti>(roots_size, INF));
		// The label table records label distance for every vertex.
		// This is replaced by the vertices_dists in the short_index.
	The temporary data structure for storing candidates of vertices vector<ShortIndex> short_index(num_v);
		// Temporary distance table, recording in the current iteration the traversing distance from a vertex to a root.
		// The candidate table is replaced by the ShortIndex structure: every vertex has a queue and a distance array;
   		// 1. the queue records last inserted labels.
		// 2. the distance array acts like a bitmap but restores distances.


	/*
	First, use vertex-centric method, all vertices are sending messages
	to neighbors and updates their distances in the distance table. 
	The distance table is a temporary data structure for building the 
	index later. Here they do not add anything into the index.
	*/
	// Activate roots
	initialize_tables(
		The candidate distances data structure vector<ShortIndex> short_index,
		The distance table vector<weighti> dists_table,
		The active queue vector<idi> active_queue,
		The ID of beginning vertex of this batch idi roots_start,
		The number of vertices in this batch inti roots_size,
		The index vector<IndexType> L);

	// Active vertices sending messages.
	// Those vertices received messages are put into has_candidates_queue.
	while (!active_queue.empty()) {
		// First stage, sending distances.
		// Traverse all active vertex and send distances
		for (every vertex v in the active_queue) {
			// vertex v send distance messages to all its neighbors;
			// every neighbor gets its candidates.
			is_active[v] = false; // reset is_active
			send_messages(
					active vertex v,
					The beginning ID of roots roots_start,
					Graph G,
					distance table dists_table,
					the temporary data of candidates vector<ShortIndex> short_index,
					The queue has_candidates_queue,
					The flag array has_candidates);
		}
		active_queue.clear();

		// Second stage, checking candidates.
		// Traverse has_candidates_queue, check all candidates of every vertex
		for (every vertex v in the has_candidates_queue) {
			flag need_activate = false; // Flag: if v should be a new active vertex in the next iteration
			// Traverse all candidates of v
			for (every candidate c in short_index[v].candidates_que) {
				weighti tmp_dist_v_c = short_index[v].candidates_dists[c];
				// Distance check for pruning
				weighti query_dist_v_c;
				if (INF == (query_dist_v_c = distance_query(
							vertex v,
							candidate c,
							The distance table dists_table,
							The candidate distances data structure vector<ShortIndex> short_index,
							The index L,
							The beginning ID of roots_start,
							distance tmp_dist_v_c))) {
					if (INF == short_index[v].vertices_dists[c]) {
						short_index[v].vertices_que.enqueue(c);
					}
					// Record the new distance in the label table
					short_index[v].vertices_dists[c] = tmp_dist_v_c;
					short_index[v].last_new_roots.enqueue(c);
					need_activate = true;
				} else if (query_dist_v_c < tmp_dist_v_c){
					dists_table[c][v] = query_dist_v_c;
					// First correcting option:
					// v needs to send message back to its neighbor to change potential wrong distance from root c
					sending_back(
							vertex v,
							candidate c,
							distance table dists_table,
							labels table labels_table);
				}
				//short_index[v].candidates_dists[c] = INF; // DEPRECATED! // Reset candidates_dists after using in distance_query.
			}
			//short_index[v].candidates_que.clear(); // DEPRECATED!
			if (need_activate) {
				if (!is_active[v]) {
					is_active[v] = true;
					active_queue.enqueue(v); // Here needs a bitmap to ensure v is added only once
				}
			}
		}
		// Reset vertices' candidates_que and candidates_dists
		for (every vertex v in the has_candidates_queue) {
			has_candidates[v] = false; // reset has_candidates
			for (every candidate c in short_index[v].candidates_que) {
				short_index[v].candidates_dists[c] = INF; // Reset candidates_dists
			}
			short_index[v].candidates_que.clear(); // Clear candidates_que
		}
		has_candidates_queue.clear();
	}

	// Second correcting option:
//	filter_out_labels_table_naive(
//					distance table dists_table,
//					labels table labels_table);

	reset_tables(
		The candidate distances data structure vector<ShortIndex> short_index,
		The index L,
		The Distance Tabel vector< vector<weighti> > dists_table);
	/*
	Second, after the message-sending phase, all distances are
	available in the distance table. And according to it, we can build 
	the index.
	*/
	update_index(
		The candidate distances data structure vector<ShortIndex> short_index,
		The index vector<IndexType> L,
		The ID of beginning vertex of this batch idi roots_start);

}


int main()
{
	Get graph data Graph G; // Smaller vertex ID has higher rank (vertex 0 is highest ranked)
	The empty index vector<IndexType> L;
	The batch size BATCH_SIZE = 1024;
	The number of batches num_batches = num_v / BATCH_SIZE;
	The number of remaining vertices remainder = num_v % BATCH_SIZE;

	for (inti b_i = 0; b_i < num_batches; ++b_i) {
		vertex_centric_labeling_in_batches(
				The start ID of vertex b_i * BATCH_SIZE; // roots_start
				The number of roots in this batch BATCH_SIZE; // roots_size
				Graph G,
				vector<IndexType> &L);
	}
	if (0 != remainder) {
		vertex_centric_labeling_in_batches(
				The start ID of vertex num_batches * BATCH_SIZE; // roots_start
				The number of roots in this batch remainder; // roots_size
				Graph G,
				vector<IndexType> &L);
	}

	// labeling
//	vertex_centric_labeling(
//			Graph G,
//			vector<IndexType> &L);
	return 0;
}
