// Function for initializing at the begin of a batch
// For a batch, roots' labels;
// traverse roots' labels to initialize distance buffer;
void initialize(
			Root set roots,
			A distance buffer matrix dist_matrix,
			A queue active_queue,
			Label set L)
{
	// Initialize roots labels and distance matrix
	for (every root root_id in roots) {
		Insert (root_id, 0) into L[root_id];
		// Push it to active queue
		active_queue.enqueue(root_id);

		// Initialize distance matrix
		// Traverse r_id's all existing labels
		for (every label (v, dist) in L[root_id]) {
			dist_matrix[root_id][v] <- dist;
		}
	}
}

// Function that pushes v's labels to v's every neighbor
void push_labels(
			Active vertex v,
			Label set L,
			Queue candidate_queue,
			A flag array got_candidates)
{
	// Traverse v's every neighbor u
	for (every neighbor u of v) {
		for (every new inserted label (k, dist) of L[v] (inserted in the last interation)) {
			if (rank(k) is higher than rank(u) AND k is not in u.candidates) {
				Insert k into u.candidates;
				if (got_candidates[u] is false) {
					// Add u into candidate_queue
					got_candidates[u] <- true;
					candidate_queue.enqueue(u);
				}
			}
		}
	}
}

// Function for distance query;
// traverse vertex v's labels;
// return false if shorter distance exists already, return true if the cand_id can be added into v's label.
bool distance_query(
			Candidate cand_id,
			Vertex v,
			Label set L,
			Distance buffer dist_matrix,
			Iterator iter) // also the targeted distance
{
	for (every label (k, dist) in L[v]) {
		if (dist_matrix[cand_id][k] + dist <= iter) {
			return false;
		}
	}
	return true;
}

// Function processes every batch
void batch_process(
				Graph G,
				Root set roots,
				Label set L)
{
	// A queue of active vertices in the batch
	A queue containing all vertices who are active is active_queue; // as static vector<int>, the worklist for all active vertices;
	A flag array is is_active; // as static vector<bool>, is_active[v] is true means vertex v is in the active queue.
	// A queue of vertices who have candidate in the iteration
	A queue containing all vertices who have candidate labels is candidate_queue; // as static vector<int>, the worklist for all vertices who has candidate labels
	A flag array is got_candidates; // as static vector<bool>, got_candidates[v] is true means vertex v is in the queue candidate_queue
	A distance buffer matrix is dist_matrix, every element is initialized as INF; // as static vetor< vector<byte>>, to buffer distances from roots to other vertices

	// At the beginning of a batch, initialize the labels L and distance buffer dist_matrix;
	initialize(
			Root set roots,
			A distance buffer matrix dist_matrix,
			A queue active_queue,
			Label set L);

	byte iter = 0; // The iterator, also the distance for current iteration
	while (active_queue is not empty) { // active_queue is not empty
		++iter;
		// Traverse active vertices to push their labels as candidates
		for (every vertex v in active_queue) {
			// v_head = active_queue[i_queue]; // The active vertex v_head
			is_active[v] = false; // reset is_active
			// Push v_head's labels to v_head's every neighbor
			push_labels(
					Active vertex v,
					Label set L,
					Queue candidate_queue,
					A flag array got_candidates);
		}
		Set active_queue empty; // Set the active_queue empty

		// Traverse vertices in the candidate_queue to insert labels
		for (every vertex v in candidate_queue) {
			// A counter inserted_count = 0; //recording number of v_id's truly inserted candidates
			got_candidates[v] = false; // reset got_candidates
			A flag got_new = false; // if got_new is true, vertex v got new labels in this iteration
			// Traverse v's all candidates
			for (every candidate cand_id in v.candidates) {
				// Only insert cand_id into v's label if its distance to v is shorter than existing distance
				if ( distance_query(
								Candidate cand_id,
								Vertex v,
								Label set L,
								Distance buffer dist_matrix,
								Iterator iter) ) {
					if (got_new is false) {
						got_new <- true;
					}
					Insert label (cand_id, iter) into L[v];
				}
			}
			if (got_new is true AND is_active[v] is false) {
				// Insert v into the active queue
				is_active[v] <- true;
				active_queue.enqueue(v);
			}
		}
		Set candidate_queue empty;
	}
}

int main(Graph G)
{
	The label set L;
	BATCH_SIZE = 1024;
	The number of batches n_batch = N / BATCH_SIZE;

	for (every batch) {
		Add all root vertices into roots;
		batch_process(
				Graph G,
				Root set roots,
				Label set L);
	}
}