// Structure for the type of label
struct IndexType {
	struct Batch {
		int batch_id; // Batch ID
		int start_index; // Index to the array distances where the batch starts
	};

	struct DistanceIndexType {
		int start_index; // Index to the array vertices where the same-ditance vertices start
		int size; // Number of the same-distance vertices
		byte dist; // The real distance
	};

	vector<Batch> batches; // Batch info
	vector<DistanceIndexType> distances; // Distance info
	vector<byte> vertices; // Vertices in the label, preresented as temperory ID
};

// Structure for the type of temporary label
struct ShortIndex {
	bitset indicator; // Global indicator, indicator[r] is set means root r once selected as candidate already
	bitset candidates; // Candidates one iteration, candidates[r] is set means root r is candidate in this iteration
}

// Function processes every batch
void batch_process(
				Graph G,
				Batch ID b_id,
				Start ID of roots roots_start,
				The number of roots ROOTS_SIZE,
				The index L)
{
	A distance matrix dist_matrix; // as static vetor< vector<byte>>, to buffer distances from roots to other vertices
	An array short_index; // as static vector<ShortIndex>, for every vertex's temperary labels in a batch
	A flag array is_active; // as static vector<bool>, is_active[v] is true means vertex v is active
	A flag array got_labels; // as static vector<bool>, got_labels[v] is true means vertex v got new labels in this batch

	for (every vertex v) {
		// Unset v's short_index's bit set indicator; the bitset candidates is already unset
		short_index[v].indicator.reset();
		got_labels[v] = false; // No vertex has got new labels in this batch, yet.
	}
	// Initialize roots labels and distance matrix
	for (every root r_id which 0 <= r_id < ROOTS_SIZE) {
		// Initialize roots labels
		The global vertex ID of r_id is r_real_id = r_id + roots_start;
		short_index[r_id].indicator.set(r_real_id); // Set
		// Insert (r_real_id, 0) to r_id's label L[r_real_id]
		// Insert new Batch's batch_id and start_index
		L[r_real_id].batches.push_back(Batch(b_id, L[r_real_id].distances.size()));
		// Insert new DistanceIndexType's start_index, size, and dist
		L[r_real_id].distances.push_back(DistanceIndexType(L[r_real_id].vertices.size(), 1, 0));
		// Insert label vertices temporary ID
		L[r_real_id].vertices.push_back(r_id);
		is_active[r_real_id] = true;
		got_labels[r_real_id] = true;

		// Initialize distance matrix
		Set all elements in dist_matrix[r_id] as INF;
		// Traverse r_id's all existing labels
		for (every batch b_i in L[r_real_id].batches) {
			// Temporary ID's offset to real ID
			int id_offset = L[r_real_id].batches[b_i].batch_id * ROOTS_SIZE;
			// Index to the distances array
			int dist_i = L[r_real_id].batches[b_i].start_index;
			// Index to the label vertices array
			int start_index = L[r_real_id].distances[dist_i].start_index;
			int bound_index = start_index + L[r_real_id].distances[dist_i].size;
			// The label's distance
			byte dist = L[r_real_id].distances[dist_i].dist;
			// Traverse labels for the same distance
			for (every v_i which start_index <= v_i < bound_index) {
				Label v = L[r_real_id].vertices[v_i] + id_offset;
				// Record the label's distance into the distance buffer
				dist_matrix[r_id][v] = dist;
			}
		}
	}

	byte iter = 0; // The iterator, also the distance for current iteration

	Stop = false;
	while (Stop) {
		Stop = true;
		++iter;

		// Traverse active vertices to push their labels as candidates
		for (every active vertex v_head which roots_start <= v_head < N) {
			if (!is_active[v_head]) {
				continue;
			}
			is_active[v_head] = true;
			l_i_start = L[v_head].distances.rbegin()->start_index; // The distances array's last element has the start index of labels
			l_i_bound = t_i_start + L[v_head].distances.rbegin()->size;
			// Push v_head's last inserted labels to its all neighbors
			for (every neighbor v_tail of v_head) { // v_head's neighbors are decreasingly ordered by rank.
				if (v_tail < L[v_head].vertices[l_i_start] + roots_start) { // v_tail has higher rank than any v_head's labels
					break;
				}
				// Traverse v_head's last inserted labels
				for (every l_i which l_i_start <= l_i < l_i_bound) {
					label_root_id = L[v_head].vertices[l_i];
					if (v_tail < label_root_id + roots_start) { // v_head's last inserted labels are ordered by rank.
						// v_tail has higher rank than all remaining labels
						break;
					}
					if (short_index[v_tail].indicator[label_root_id] is set) {
						// The label is alreay selected before
						continue;
					}
					// Record the label as v_tail's candidates label
					short_index[v_tail].candidates.set(label_root_id);
				}
			}
		}

		// Check every vertex's candidates if need adding to labels
		for (every vertex v_id which roots_start <= v_id < N) {
			if (!short_index[v_id].candidates.any()) {
				// vertex v_id has no candidates
				continue;
			}
			A counter recording number of truly inserted candidates is inserted_count = 0;
			// Traverse v_id's all candidates
			for (every cand_root_id which 0 <= cand_root_id < ROOTS_SIZE) {
				if (!short_index[v_id].candidates[cand_root_id]) {
					// Root cand_root_id is not vertex v_id's candidate
					continue;
				}
				// Traverse vertex v_id's labels to check if v_id to cand_root_id has shorter path
				Distance d_query = INF;
				cand_real_id = cand_root_id + roots_start; // cand_root_id's real ID
				for (every batch b_i in L[v_id].batches) {
					// Temporary ID's offset to real ID
					int id_offset = L[v_id].batches[b_i].batch_id * ROOTS_SIZE;
					// Index to the distances array
					int dist_i = L[v_id].batches[b_i].start_index;
					// Index to the label vertices array
					int start_index = L[v_id].distances[dist_i].start_index;
					int bound_index = start_index + L[v_id].distances[dist_i].size;
					// The label's distance
					byte dist = L[v_id].distances[dist_i].dist;
					if (dist > iter) { // In a batch, the labels' distances are increasingly ordered.
						// If the half path distance is already greater than ther targeted distance, jump to next batch
						continue;
					}
					// Traverse labels for the same distance
					for (every v_i which start_index <= v_i < bound_index) {
						Label v = L[v_id].vertices[v_i] + id_offset;
						if (v > cand_real_id) {
							// Vertex cand_real_id cannot have labels whose ranks are lower than it.
							continue;
						}
						// Record the label's distance into the distance buffer
						Query distance d_tmp = dist + dist_matrix[cand_root_id][v];
						if (d_tmp < d_query) {
							d_query = d_tmp;
						}
					}
				}
				if (iter < d_query) {
					// Only insert the label with distance shorter than existing distance
					L[v_id].vertices.push_back(cand_root_id);
					++inserted_count; // Increase the counter recording number of truly inserted candidates
					is_active[v_id] = true;
					stop = false;
					// Update the distance buffer if necessary
					v_root_id = v_id - roots_start;
					if (v_id >= roots_start && v_root_id < ROOTS_SIZE) {
						// Only if vertex v_id is a root in this batch
						dist_matrix[v_root_id][cand_real_id] = iter;
					}
				}
			}
			// Update other arrays in L[v_id] if new labels inserted in this iteration
			if (inserted_count != 0) {
				if (!got_labels[v_id]) {
					// If vertex v_id has not inserted new labels in this batch, yet
					got_labels[v_id] = true;
					// Insert new Batch's batch_id and start_index
					L[v_id].batches.push_back(Batch(b_id, L[v_id].distances.size()));
				}
				// Insert new DistanceIndexType's start_index, size, and dist
				L[v_id].distances.push_back(DistanceIndexType(L[v_id].vertices.size() - inserted_count, inserted_count, iter));
			}
		}
	}
}

int main(Graph G)
{
	The number of vertices is N;
	The vertices ID from 0 to N - 1 in G have ranks from high to low.
	The index of all vertices is L // as vertex<IndexType>
	The number of roots in every batch ROOTS_SIZE = 256;
	The number of batches n_batch = N / ROOTS_SIZE;

	for (every batch) {
		batch_process(
				Graph G,
				Batch ID b_id,
				Start ID of roots roots_start,
				The number of roots ROOTS_SIZE,
				The index L);
	}
}