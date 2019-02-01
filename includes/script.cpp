#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>

using namespace std;

void fun()
{
	int buffer_size = 1024
	vector<uint8_t> dists_buffer(buffer_size);
	vector<uint32_t> ids_buffer(buffer_size);

	for (every batch B) {
		int capacity = MAX_BUFFER_SIZE;
		int size_buffer = 0;
		for (every distance D in B) {
			int v_i = D.start_index;
			int remain = D.size();
			while (0 != remain) {
				// This distance still has labels
				if (0 != capacity) {
					// Buffer still has free space
					if (capacity >= remain) {
						// All remain can be put into Buffer
						// ids_buffer
						memcpy(&ids_buffer[size_buffer], &vertices[v_i], remain * sizeof(idi));
						// dists_buffer
						int bound_i = size_buffer + remain;
						for (; size_buffer < bound_i; ++size_buffer) {
							dists_buffer[size_buffer] = D.dist;
						}
						v_i += remain;
						capacity -= remain;
						remain = 0;
					} else {
						// Only the capacity-size of remain can be put into Buffer
						// ids_buffer
						memcpy(&ids_buffer[size_buffer], &vertices[v_i], capacity * sizeof(idi));
						// dists_buffer
						int bound_i = size_buffer + capacity;
						for (; size_buffer < bound_i; ++size_buffer) {
							dists_buffer[size_buffer] = D.dist;
						}
						v_i += capacity;
						remain -= capacity;
						capacity = 0;
					}
				} else {
					process();
				}
			}
		}
		process(); // process remains in the buffer
	}
}

void process(
		vector<weighti> dists_buffer,
		vector<idi> ids_buffer,
		idi size_buffer,
		idi cand_root_id,
		__m512i cand_real_id_v,
		__m512i id_offset_v,
		__m512i iter_v,
		const vector<IndexType> &L,
		const vector< vector<weighti> > &dist_matrix)
{
	inti remainder_simd = size_buffer % NUM_P_INT;
	idi bound_v_i = size_buffer - remainder_simd;
	for (idi v_i = 0; v_i < bound_v_i; v_i += NUM_P_INT) {
		__m512i v_v = _mm512_loadu_epi32(&ids_buffer[v_i]);
		v_v = _mm512_add_epi32(v_v, id_offset_v);
		__mmask16 is_r_higher_ranked_m = _mm512_cmplt_epi32_mask(v_v, cand_real_id_v);
		if (!is_r_higher_ranked_m) {
			continue;
		}
		// Distance from v to r
		__m512i dist_v = _mm512_mask_loadu_epi32(INF_v, is_r_higher_ranked_m, &dists_buffer[v_i]);
		dist_v = _mm512_and_epi32(dist_v, LOWEST_BYTE_MASK);
		// Get distance from r to c
		__m512i dists_r_c_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, v_v, &dist_matrix[cand_root_id][0], sizeof(weighti));
		dists_r_c_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_r_c_v, LOWEST_BYTE_MASK);
		// Query distance from v to c
		__m512i d_tmp_v = _mm512_mask_add_epi32(INF_v, is_r_higher_ranked_m, dist_v, dists_r_c_v);
		__mmask16 is_query_shorter = _mm512_mask_cmple_epi32_mask(is_r_higher_ranked_m, d_tmp_v, iter_v);
		if (is_query_shorter) {
			distance_query_time += WallTimer::get_time_mark();
			return false;
		}
	}
	if (remainder_simd) {
		__mmask16 in_m = (__mmask16) ((uint16_t) 0xFFFF >> (NUM_P_INT - remainder_simd));
		__m512i v_v = _mm512_mask_loadu_epi32(UNDEF_i32_v, in_m, &ids_buffer[v_i]);
		v_v = _mm512_add_epi32(v_v, id_offset_v);
		__mmask16 is_r_higher_ranked_m = _mm512_mask_cmplt_epi32_mask(in_m, v_v, cand_real_id_v);
		if (!is_r_higher_ranked_m) {
			continue;
		}
		// Distance from v to r
		__m512i dist_v = _mm512_mask_loadu_epi32(INF_v, is_r_higher_ranked_m, &dists_buffer[v_i]);
		dist_v = _mm512_and_epi32(dist_v, LOWEST_BYTE_MASK);
		// Get distance from r to c
		__m512i dists_r_c_v = _mm512_mask_i32gather_epi32(INF_v, is_r_higher_ranked_m, v_v, &dist_matrix[cand_root_id][0], sizeof(weighti));
		dists_r_c_v = _mm512_mask_and_epi32(INF_v, is_r_higher_ranked_m, dists_r_c_v, LOWEST_BYTE_MASK);
		// Query distance from v to c
		__m512i d_tmp_v = _mm512_mask_add_epi32(INF_v, is_r_higher_ranked_m, dist_v, dists_r_c_v);
		__mmask16 is_query_shorter = _mm512_mask_cmple_epi32_mask(is_r_higher_ranked_m, d_tmp_v, iter_v);
		if (is_query_shorter) {
			distance_query_time += WallTimer::get_time_mark();
			return false;
		}
	}
}
