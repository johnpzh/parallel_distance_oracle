// Structure for the type of label
struct IndexType {
	struct Batch {
		int batch_id; // Batch ID
		int start_index; // Index to the array distances where the batch starts
		int size; // Number of distances element in this batch
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

const int BATCH_SIZE = 256;

// Function for initializing at the begin of a batch
// For a batch, initialize the temporary labels and real labels of roots;
// traverse roots' labels to initialize distance buffer;
// unset flag arrays is_active and got_labels
void initialize(
			An array of temporary labels short_index,
			A distance buffer matrix dist_matrix,
			A queue active_queue,
			A index end_active_queue,
			A flag array got_labels,
			Batch ID b_id,
			Start ID of roots roots_start,
			The number of roots roots_size,
			The index L,
			Number of vertices N)
{
	for (vertex v = 0; v < N; ++v) {
		// Unset v's short_index's bit set indicator; the bitset candidates is already unset
		short_index[v].indicator.reset();
		got_labels[v] = false; // No vertex has got new labels in this batch, yet.
	}
	// Initialize roots labels and distance matrix
	for (root ID r_id = 0; r_id < roots_size; ++r_id) {
		// Initialize roots labels
		The global vertex ID of r_id is r_real_id = r_id + roots_start;
		short_index[r_id].indicator.set(r_real_id); // Set r_real_id itself has been already selected in r_id's temporary labels
		// Insert (r_real_id, 0) to r_id's label L[r_real_id]
		// Insert new Batch's batch_id, start_index, and size
		L[r_real_id].batches.push_back(Batch(b_id, L[r_real_id].distances.size(), 1));
		// Insert new DistanceIndexType's start_index, size, and dist
		L[r_real_id].distances.push_back(DistanceIndexType(L[r_real_id].vertices.size(), 1, 0));
		// Insert label vertices temporary ID
		L[r_real_id].vertices.push_back(r_id);
		// Push it to active queue
		active_queue[end_active_queue++] = r_real_id;
		got_labels[r_real_id] = true;

		// Initialize distance matrix
		Set all elements in dist_matrix[r_id] as INF;
		// Traverse r_id's all existing labels
		for (every batch b_i in L[r_real_id].batches) {
			// Temporary ID's offset to real ID
			int id_offset = L[r_real_id].batches[b_i].batch_id * BATCH_SIZE;
			// Index to the distances array
			int dist_start_index = L[r_real_id].batches[b_i].start_index;
			int dist_bound_index = dist_start_index + L[r_real_id].batches[b_i].size;
			// Traverse the distances array
			for (index dist_i = dist_start_index; dist_i < dist_bound_index ++dist_i) {
				// Index to the label vertices array
				int v_start_index = L[r_real_id].distances[dist_i].start_index;
				int v_bound_index = v_start_index + L[r_real_id].distances[dist_i].size;
				// The label's distance
				byte dist = L[r_real_id].distances[dist_i].dist;
				// Traverse labels for the same distance
				for (index v_i = v_start_index; v_i < v_bound_index; ++v_i) {
					Label v = L[r_real_id].vertices[v_i] + id_offset;
					// Record the label's distance into the distance buffer
					dist_matrix[r_id][v] = dist;
				}
			}
		}
	}
}

// Function that pushes v_head's labels to v_head's every neighbor
void push_labels(
	The active vertex v_head,
	Start ID of roots roots_start,
	Graph G, // Traverse v_head's neighbors.
	Real label L,
	Temporary label short_index,
	The queue candidate_queue,
	The end index of the queue end_candidate_queue,
	A flag array got_candidates)
{
	// These 2 index are used for traversing v_head's last inserted labels
	l_i_start = L[v_head].distances.rbegin()->start_index; // The distances array's last element has the start index of labels
	l_i_bound = l_i_start + L[v_head].distances.rbegin()->size;
	// Traverse v_head's every neighbor v_tail
	for (every neighbor v_tail of v_head) { // v_head's neighbors are decreasingly ordered by rank.
		if (v_tail < L[v_head].vertices[l_i_start] + roots_start) { // v_tail has higher rank than any v_head's labels
			return;
		}
		// Traverse v_head's last inserted labels
		for (index l_i = l_i_start; l_i < l_i_bound; ++l_i) {
			label_root_id = L[v_head].vertices[l_i]; // label_root_id is a last inserted label
			if (v_tail < label_root_id + roots_start) { // v_head's last inserted labels are ordered by rank.
				// v_tail has higher rank than all remaining labels
				break;
			}
			if (short_index[v_tail].indicator[label_root_id] is set) {
				// The label is alreay selected before
				continue;
			}
			// Record vertex label_root_id as v_tail's candidates label
			short_index[v_tail].candidates.set(label_root_id);
			// Record vertex label_root_id as already selected label, so will not be selected again by v_tail in this batch
			short_index[v_tail].indicator.set(label_root_id);
			if (!got_candidates[v_tail]) {
				// If v_tail is not in candidate_queue, add it in
				got_candidates[v_tail] = true;
				candidate_queue[end_candidate_queue++] = v_tail;
			}
		}
	}
}

// Function for distance query;
// traverse vertex v_id's labels;
// return the distance between v_id and cand_root_id based on existing labels.
byte distance_query(
			The candidate temperary ID cand_root_id,
			The vertex ID v_id,
			Start ID of roots roots_start,
			The number of roots roots_size,
			Real labels L,
			Distance buffer dist_matrix,
			The iterator iter) // also the targeted distance
{

	Distance d_query = INF;
	cand_real_id = cand_root_id + roots_start; // cand_root_id's real ID
	// Traverse v_id's all existing labels
	for (every batch b_i in L[v_id].batches) {
		// Temporary ID's offset to real ID
		int id_offset = L[v_id].batches[b_i].batch_id * BATCH_SIZE;
		// Index to the distances array
		int dist_start_index = L[r_real_id].batches[b_i].start_index;
		int dist_bound_index = dist_start_index + L[r_real_id].batches[b_i].size;
		// Traverse the distances array
		for (index dist_i = dist_start_index; dist_i < dist_bound_index ++dist_i) {
			// Index to the label vertices array
			int v_start_index = L[r_real_id].distances[dist_i].start_index;
			int v_bound_index = v_start_index + L[r_real_id].distances[dist_i].size;
			// The label's distance
			byte dist = L[r_real_id].distances[dist_i].dist;
			if (dist > iter) { // In a batch, the labels' distances are increasingly ordered.
				// If the half path distance is already greater than ther targeted distance, jump to next batch
				break;
			}
			// Traverse labels for the same distance
			for (index v_i = v_start_index; v_i < v_bound_index; ++v_i) {
				Label v = L[r_real_id].vertices[v_i] + id_offset;
				if (v > cand_real_id) {
					// Vertex cand_real_id cannot have labels whose ranks are lower than it.
					continue;
				}
				// The query distance based on exsiting labels;
				Query distance d_tmp = dist + dist_matrix[cand_root_id][v];
				if (d_tmp < d_query) {
					d_query = d_tmp;
				}
			}
		}
	}
	return d_query;
}

// Function inserts candidate cand_root_id into vertex v_id's labels;
// update the distance buffer dist_matrix;
// but it only update the v_id's labels' vertices array;
void insert_label_only(
			The candidate temperary ID cand_root_id,
			The vertex ID v_id,
			Start ID of roots roots_start,
			The number of roots roots_size,
			Real labels L,
			Distance buffer dist_matrix,
			The iterator iter)
{
	L[v_id].vertices.push_back(cand_root_id);
	// Update the distance buffer if necessary
	Real ID v_root_id = v_id - roots_start;
	if (v_id >= roots_start && v_root_id < roots_size) {
		// Only if vertex v_id is a root in this batch, update distance buffer dist_matrix
		dist_matrix[v_root_id][cand_root_id + roots_start] = iter;
	}
}

// Function updates those index arrays in v_id's label only if v_id has been inserted new labels
void update_label_indices(
			The vertex ID v_id,
			The counder inserted_count;
			Real labels L,
			Flag array got_labels
			Batch id b_id,
			The iterator iter)
{
	if (got_labels[v_id]) {
		// Increase the batches' last element's size because a new distance element need to be added
		++(L[v_id].batches.rbegin()->size);
	} else {
		// If vertex v_id has not yet inserted new labels in this batch
		got_labels[v_id] = true;
		// Insert a new Batch with batch_id, start_index, and size because a new distance element need to be added
		L[v_id].batches.push_back(Batch(b_id, L[v_id].distances.size(), 1));
	}
	// Insert a new distance element with start_index, size, and dist
	L[v_id].distances.push_back(DistanceIndexType(L[v_id].vertices.size() - inserted_count, inserted_count, iter));
}

// Function processes every batch
void batch_process(
				Graph G,
				Batch ID b_id,
				Start ID of roots roots_start,
				The number of roots roots_size,
				The index L)
{
	// A queue of active vertices in the batch
	// The queue is initialized as N elements and use 1 index to maintain in order to reduce the cost of poping and pushing.
	A queue containing all vertices who are active is active_queue; // as static vector<int>, the worklist for all active vertices;
	A index recording the end of active_queue is end_active_queue = 0;
	// A queue of vertices who have candidate in the iteration
	// Similarly, The queue is initialized as N elements and use 1 index to maintain in order to reduce the cost of poping and pushing.
	A queue containing all vertices who have candidate labels is candidate_queue; // as static vector<int>, the worklist for all vertices who has candidate labels
	A index pointing to the end of candidate_queue is end_candidate_queue = 0;
	An array of temporary labels is short_index; // as static vector<ShortIndex>, for every vertex's temperary labels in a batch
	A distance buffer matrix is dist_matrix; // as static vetor< vector<byte>>, to buffer distances from roots to other vertices
	A flag array is got_labels; // as static vector<bool>, got_labels[v] is true means vertex v got new labels in this batch
	A flag array is got_candidates; // as static vector<bool>, got_candidates[v] is true means vertex v is in the queue candidate_queue
	A flag array is is_active; // as static vector<bool>, is_active[v] is true means vertex v is in the active queue.

	// At the beginning of a batch, initialize the labels L and distance buffer dist_matrix;
	initialize(
			An array of temporary labels short_index,
			A distance buffer matrix dist_matrix,
			A queue active_queue,
			A index end_active_queue,
			A flag array got_labels,
			Batch ID b_id,
			Start ID of roots roots_start,
			The number of roots roots_size,
			The index L,
			Number of vertices N);

	byte iter = 0; // The iterator, also the distance for current iteration
	while (end_active_queue != 0) { // active_queue is not empty
		++iter;
		// Traverse active vertices to push their labels as candidates
		for (index i_queue = 0; i_queue < end_active_queue; ++i_queue) {
			v_head = active_queue[i_queue]; // The active vertex v_head
			is_active[v_head] = false; // reset is_active
			// Push v_head's labels to v_head's every neighbor
			push_labels(
					The active vertex v_head,
					Start ID of roots roots_start,
					Graph G, // Traverse v_head's neighbors.
					Real label L,
					Temporary label short_index,
					The queue candidate_queue,
					The end index of the queue end_candidate_queue,
					A flag array got_candidates);
		}
		end_active_queue = 0; // Set the active_queue empty

		// Traverse vertices in the candidate_queue to insert labels
		for (index i_queue = 0; i_queue < end_candidate_queue; ++i_queue) {
			v_id = candidates_queue[i_queue]; // The vertex v_id has candidates
			A counter inserted_count = 0; //recording number of v_id's truly inserted candidates
			got_candidates[v_id] = false; // reset got_candidates
			// Traverse v_id's all candidates
			for (index cand_root_id = 0; cand_root_id < roots_size; ++cand_root_id) {
				// Test if cand_root_id is v_id's candidate
				if (!short_index[v_id].candidates[cand_root_id]) {
					// Root cand_root_id is not vertex v_id's candidate
					continue;
				}
				// Get the distance between v_id and cand_root_id based on existing labels
				Distance d_query = distance_query(
										The candidate temperary ID cand_root_id,
										The vertex ID v_id,
										Start ID of roots roots_start,
										Real labels L,
										Distance buffer dist_matrix,
										The iterator iter);
				// Only insert cand_root_id into v_id's label if its distance to v_id is shorter than existing distance
				if (iter < d_query) {
					// Insert v_id into the active queue
					if (!is_active[v_id]) {
						is_active[v_id] = true;
						active_queue[end_active_queue++] = v_id;
					}
					// Update the counter of the number of inserted new labels
					++inserted_count;
					// The candidate cand_root_id needs to be added into v_id's label
					insert_label_only(
							The candidate temperary ID cand_root_id,
							The vertex ID v_id,
							Start ID of roots roots_start,
							The number of roots roots_size,
							Real labels L,
							Distance buffer dist_matrix,
							The iterator iter);
				}
			}
			if (inserted_count != 0) {
				// Update other arrays in L[v_id] if new labels were inserted in this iteration
				update_label_indices(
						The vertex ID v_id,
						The counder inserted_count;
						Real labels L,
						Flag array got_labels,
						Batch id b_id,
						The iterator iter);
			}
		}
		end_candidate_queue = 0; // Set the candidate_queue empty
	}
}

int main(Graph G)
{
	The number of vertices is N;
	The vertices ID from 0 to N - 1 in G have ranks from high to low.
	The index of all vertices is L // as vertex<IndexType>
	The number of roots in every batch roots_size = BATCH_SIZE;
	The number of batches n_batch = N / roots_size;

	for (every batch) {
		batch_process(
				Graph G,
				Batch ID b_id,
				Start ID of roots roots_start,
				The number of roots roots_size,
				The index L);
	}
}