// Function for distance query;
// traverse vertex v's labels;
// return false if shorter distance exists already, return true if the cand_id can be added into v's label.
function distance_query(
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
function batch_process(
				Graph G,
				Root set roots,
				Label set L)
{
	// A queue of active vertices in the batch
	A queue containing all vertices who are active is active_queue; // as static vector<int>, the worklist for all active vertices;
	// A queue of vertices who have candidate in the iteration
	A queue containing all vertices who have candidate labels is candidate_queue; // as static vector<int>, the worklist for all vertices who has candidate labels
	A distance buffer matrix is dist_matrix, every element is initialized as INF; // as static vetor< vector<byte>>, to buffer distances from roots to other vertices

	// At the beginning of a batch, initialize the labels L and distance buffer dist_matrix;
	// Initialize roots labels and distance matrix
	for (every root root_id in roots) {
		Insert (root_id, 0) into L[root_id];
		// Push it to active queue
		Insert root_id toto active_queue;

		// Initialize distance matrix
		// Traverse r_id's all existing labels
		for (every label (v, dist) in L[root_id]) {
			dist_matrix[root_id][v] <- dist;
		}
	}

	The iterator iter = 0; // The iterator, also the distance for current iteration
	while (active_queue is not empty) { // active_queue is not empty
		iter <- iter + 1;
		// Traverse active vertices to push their labels as candidates
		for (every vertex v in active_queue) {
			// Push v_head's labels to v_head's every neighbor
			// Traverse v's every neighbor u
			for (every neighbor u of v) {
				for (every new inserted label (k, dist) of L[v] (inserted in the last iteration)) {
					if (rank(k) is higher than rank(u) AND k is not in u.candidates) {
						Insert k into u.candidates;
						Insert u into candidate_queue;
					}
				}
			}
		}

		// Traverse vertices in the candidate_queue to insert labels
		for (every vertex v in candidate_queue) {
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
			if (got_new is true) {
				// Insert v into the active queue
				Insert v into active_queue;
			}
		}
	}
}

function labeling(Graph G)
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