++iter;

// Function to get the prefix sum of elements in offsets
inline idi prefix_sum_for_offsets(
					vector<idi> &offsets)
{
	// Get the offset as the prefix sum of out degrees
	idi offset_sum = 0;
	idi size = offsets.size();
	for (idi i = 0; i < size; ++i) {
		idi tmp = offsets[i];
		offsets[i] = offset_sum;
		offset_sum += tmp;
	}
	return offset_sum;
}

// inline void parepare_parallel_queue(
// 					const Graph &G,
// 					vector<idi> &offsets_tmp_candidate_queue,
// 					const vector<idi> &active_queue,
// 					idi end_active_queue)
// {
// 	// Get every thread's offset in the tmp_candidate_queue
// 	#pragma omp parallel for
// 	for (idi i_queue = 0; i_queue < end_active_queue; ++i_queue) {
// 		// Traverse all active vertices, get their out degrees.
// 		offsets_tmp_candidate_queue[i_queue] = G.out_degree[active_queue[i_queue]];
// 	}
// 	// Get the offset as the prefix sum of out degrees
// 	idi offset_sum = 0;
// 	for (idi i = 0; i < end_active_queue; ++i) {
// 		idi tmp = offsets_tmp_candidate_queue[i];
// 		offsets_tmp_candidate_queue[i] = offset_sum;
// 		offset_sum += tmp;
// 	}
// }

inline void ParaVertexCentricPLL::push_labels(
				idi v_head,
				idi roots_start,
				const Graph &G,
				const vector<IndexType> &L,
				vector<ShortIndex> &short_index,
				// vector<idi> &candidate_queue,
				// idi &end_candidate_queue,
				vector<idi> &tmp_candidate_queue,
				idi &size_tmp_candidate_queue,
				idi offset_tmp_candidate_queue,
				vector<bool> &got_candidates,
				vector<idi> &once_candidated_queue,
				idi &end_once_candidated_queue,
				vector<bool> &once_candidated,
				const vector<bool> &used_bp_roots,
				smalli iter)
{
	const IndexType &Lv = L[v_head];
	// These 2 index are used for traversing v_head's last inserted labels
	idi l_i_start = Lv.distances.rbegin() -> start_index;
	idi l_i_bound = l_i_start + Lv.distances.rbegin() -> size;
	// Traverse v_head's every neighbor v_tail
	idi e_i_start = G.vertices[v_head];
	idi e_i_bound = e_i_start + G.out_degrees[v_head];
	for (idi e_i = e_i_start; e_i < e_i_bound; ++e_i) {
		idi v_tail = G.out_edges[e_i];

		if (used_bp_roots[v_head]) {
			continue;
		}

		if (v_tail < roots_start) { // v_tail has higher rank than any roots, then no roots can push new labels to it.
			return;
		}
		if (v_tail <= Lv.vertices[l_i_start] + roots_start) { // v_tail has higher rank than any v_head's labels
			return;
		}
		const IndexType &L_tail = L[v_tail];
		_mm_prefetch(&L_tail.bp_dist[0], _MM_HINT_T0);
		_mm_prefetch(&L_tail.bp_sets[0][0], _MM_HINT_T0);
		// Traverse v_head's last inserted labels
		for (idi l_i = l_i_start; l_i < l_i_bound; ++l_i) {
			idi label_root_id = Lv.vertices[l_i];
			idi label_real_id = label_root_id + roots_start;
			if (v_tail <= label_real_id) {
				// v_tail has higher rank than all remaining labels
				break;
			}
			ShortIndex &SI_v_tail = short_index[v_tail];
			if (SI_v_tail.indicator[label_root_id]) {
				// The label is already selected before
				continue;
			}

			// Bit Parallel Checking: if label_real_id to v_tail has shorter distance already
//			++total_check_count;
			const IndexType &L_label = L[label_real_id];
			bool no_need_add = false;

			_mm_prefetch(&L_label.bp_dist[0], _MM_HINT_T0);
			_mm_prefetch(&L_label.bp_sets[0][0], _MM_HINT_T0);
		    for (inti i = 0; i < BITPARALLEL_SIZE; ++i) {
		      inti td = L_label.bp_dist[i] + L_tail.bp_dist[i];
		      if (td - 2 <= iter) {
		        td +=
		            (L_label.bp_sets[i][0] & L_tail.bp_sets[i][0]) ? -2 :
		            ((L_label.bp_sets[i][0] & L_tail.bp_sets[i][1]) |
		             (L_label.bp_sets[i][1] & L_tail.bp_sets[i][0]))
		            ? -1 : 0;
		        if (td <= iter) {
		        	no_need_add = true;
//		        	++bp_hit_count;
		        	break;
		        }
		      }
		    }
		    if (no_need_add) {
		    	continue;
		    }

		    // Record label_root_id as once selected by v_tail
			SI_v_tail.indicator.set(label_root_id);
			// Record vertex label_root_id as v_tail's candidates label
			SI_v_tail.candidates.set(label_root_id);

			// Add into once_candidated_queue
			if (!once_candidated[v_tail]) {
				// If v_tail is not in the once_candidated_queue yet, add it in
				once_candidated[v_tail] = true;
				once_candidated_queue[end_once_candidated_queue++] = v_tail;
			}
			// Add into candidate_queue
			if (!got_candidates[v_tail]) {
				// If v_tail is not in candidate_queue, add it in (prevent duplicate)
				// got_candidates[v_tail] = true;
				// candidate_queue[end_candidate_queue++] = v_tail;
				if (CAS(got_candidates + v_tail, false, true)) {
					tmp_candidate_queue[offset_tmp_candidate_queue + size_tmp_candidate_queue++] = v_tail;
				}
			}
		}
	}
}

void prepare_offsets_in_queue(
						idi end_active_queue,
						vector<idi> sizes_tmp_candidate_queue,
						vector<idi> offsets_tmp_candidate_queue)
{
	idi offset_sum = 0;
	for (idi i = 0; i < end_active_queue; ++i) {
		idi tmp = sizes_tmp_candidate_queue[i];
		offsets_tmp_candidate_queue[i] = offset_sum;
		offset_sum += tmp;
	}
}

void collect_into_queue(
					vector<idi> &tmp_queue,
					// idi end_active_queue,
					// vector<idi> &sizes_tmp_candidate_queue,
					vector<idi> &offsets_tmp_queue, // the locations in tmp_queue for writing from tmp_queue
					vector<idi> &offsets_queue, // the locations in queue for writing into queue.
					idi num_elements, // total number of elements which need to be added from tmp_queue to queue
					vector<idi> &queue,
					idi &end_queue)
{
	idi i_bound = offsets_tmp_queue.size();
	#pragma omp parallel for
	for (idi i = 0; i < i_bound; ++i) {
		idi i_q_start = end_queue + offsets_queue[i];
		// idi i_q_bound = i_queue_start + sizes_tmp_queue[i];
		idi i_q_bound;
		if (i_bound - 1 != i) {
			i_q_bound = end_queue + offsets_queue[i + 1];
		} else {
			i_q_bound = end_queue + num_elements;
		}
		idi end_tmp = offsets_tmp_queue[i];
		for (idi i_q = i_queue_start; i_q < i_q_bound; ++i_q) {
			queue[i_q] = tmp_queue[end_tmp++];
		}
	}
	end_queue += num_elements;
}

void main()
{
	// every vertex's offset location in tmp_candidate_queue
	vector<idi> offsets_tmp_candidate_queue(end_active_queue);
	#pragma omp parallel for
	for (idi i_queue = 0; i_queue < end_active_queue; ++i_queue) {
		// Traverse all active vertices, get their out degrees.
		offsets_tmp_candidate_queue[i_queue] = G.out_degree[active_queue[i_queue]];
	}
	prefix_sum_for_offsets(offsets_tmp_candidate_queue);
	// parepare_parallel_queue(
	// 				G,
	// 				vector<idi> &offsets_tmp_candidate_queue)
	// The tmp_candidate_queue has the length of offset_sum,
	// every thread write to tmp_candidate_queue at its offset location
	vector<idi> tmp_candidate_queue(offset_sum);
	// A vector to store the true number of pushed neighbors of every active vertex.
	vector<idi> sizes_tmp_candidate_queue(end_active_queue);


	// Traverse active vertices to push their labels as candidates
	#pragma omp parallel for
	for (idi i_queue = 0; i_queue < end_active_queue; ++i_queue) {
		idi v_head = active_queue[i_queue];
		is_active[v_head] = false; // reset is_active

		push_labels(
				v_head,
				roots_start,
				G,
				L,
				short_index,
				// candidate_queue,
				// end_candidate_queue,
				tmp_candidate_queue,
				sizes_tmp_candidate_queue[i_queue];
				offsets_tmp_candidate_queue[i_queue];
				got_candidates,
				once_candidated_queue,
				end_once_candidated_queue,
				once_candidated,
				used_bp_roots,
				iter);
	}

	// According to sizes_tmp_candidate_queue, get the offset for inserting to the real queue
	idi total_new = prefix_sum_for_offsets(sizes_tmp_candidate_queue);
	// prepare_offsets_in_queue(
	// 					idi end_active_queue,
	// 					vector<idi> sizes_tmp_candidate_queue,
	// 					vector<idi> offsets_tmp_candidate_queue);
	// Collect all candidate vertices from tmp_candidate_queue into candidate_queue.
	collect_into_queue(
				tmp_candidate_queue,
				offsets_tmp_candidate_queue, // the locations in tmp_queue for writing from tmp_queue
				sizes_tmp_candidate_queue, // the locations in queue for writing into queue.
				total_new, // total number of elements which need to be added from tmp_queue to queue
				candidate_queue,
				end_candidate_queue);
	end_active_queue = 0;


}
