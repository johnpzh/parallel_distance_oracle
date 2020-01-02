/*
 * globals.h
 *
 *  Created on: Sep 4, 2018
 *      Author: Zhen Peng
 *    Modified: 02/24/2019: Add a type name weightiLarge and its INF value WEIGHTILARGE_MAX, which are particularly used for Weighted version PADO.
 */

#ifndef INCLUDES_GLOBALS_H_
#define INCLUDES_GLOBALS_H_


#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include <string.h>
#include <cmath>
#include <omp.h>

using std::string;
//using std::vector;

namespace PADO {
//typedef uint64_t idi; // unsinged long long
typedef uint32_t idi; // unsigned int
//typedef int weighti;
typedef uint8_t weighti;
//typedef int32_t weightiLarge;
typedef int16_t weightiLarge;
typedef uint8_t smalli;
typedef uint32_t inti;


//const int WEIGHTI_MAX = INT_MAX;
const uint8_t WEIGHTI_MAX = UCHAR_MAX;
//const int32_t WEIGHTILARGE_MAX = INT_MAX;
const int16_t WEIGHTILARGE_MAX = SHRT_MAX;
const uint8_t SMALLI_MAX = UCHAR_MAX;

// Parallel Number of Threads
inti NUM_THREADS = 4;

// Utility Functions
// Compare and Swap
template <typename V_T>
inline bool CAS(V_T *ptr, V_T old_val, V_T new_val)
//inline bool CAS(void *ptr, V_T old_val, V_T new_val)
{
//	return __atomic_compare_exchange(ptr, &old_val, &new_val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
	if (1 == sizeof(V_T)) {
		return __atomic_compare_exchange(reinterpret_cast<uint8_t *>(ptr), reinterpret_cast<uint8_t *>(&old_val),
										 reinterpret_cast<uint8_t *>(&new_val), false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
	} else if (2 == sizeof(V_T)) {
		return __atomic_compare_exchange(reinterpret_cast<uint16_t *>(ptr), reinterpret_cast<uint16_t *>(&old_val),
										 reinterpret_cast<uint16_t *>(&new_val), false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
	} else if (4 == sizeof(V_T)) {
		return __atomic_compare_exchange(reinterpret_cast<uint32_t *>(ptr), reinterpret_cast<uint32_t *>(&old_val),
										 reinterpret_cast<uint32_t *>(&new_val), false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
	} else if (8 == sizeof(V_T)) {
		return __atomic_compare_exchange(reinterpret_cast<uint64_t *>(ptr), reinterpret_cast<uint64_t *>(&old_val),
										 reinterpret_cast<uint64_t *>(&new_val), false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
	} else {
		printf("CAS cannot support the type.\n");
		exit(EXIT_FAILURE);
	}
//	if (1 == sizeof(V_T)) {
//		return __sync_bool_compare_and_swap((uint8_t *) ptr, *((uint8_t *) &old_val), *((uint8_t *) &new_val));
//	} else if (2 == sizeof(V_T)) {
//		return __sync_bool_compare_and_swap((uint16_t *) ptr, *((uint16_t *) &old_val), *((uint16_t *) &new_val));
//	} else if (4 == sizeof(V_T)) {
//		return __sync_bool_compare_and_swap((uint32_t *) ptr, *((uint32_t *) &old_val), *((uint32_t *) &new_val));
//	} else if (8 == sizeof(V_T)) {
//		return __sync_bool_compare_and_swap((uint64_t *) ptr, *((uint64_t *) &old_val), *((uint64_t *) &new_val));
//	} else {
//		printf("CAS cannot support the type.\n");
//		exit(EXIT_FAILURE);
//	}
}


// Class for Timer
class WallTimer {
private:
	double start = 0.0;
	string item;
	void construct()
	{
		timeval t;
		gettimeofday(&t, NULL);
		start = t.tv_sec + t.tv_usec * 0.000001;
	}
public:
	WallTimer()
	{
		construct();
	}
	explicit WallTimer(const char *n) : item(n)
	{
		construct();
	}
	double get_runtime()
	{
		timeval t;
		gettimeofday(&t, NULL);
		double now = t.tv_sec + t.tv_usec * 0.000001;
		return now - start;
	}
	void print_runtime()
	{
		double runtime = get_runtime();
		printf("%s: %f\n", item.c_str(), runtime); fflush(stdout);
	}
	static double get_time_mark()
	{
		timeval t;
		gettimeofday(&t, NULL);
		return t.tv_sec + t.tv_usec * 0.000001;
	}
};
// End Class WallTimer

// Parallel Prefix Sum
inline idi prefix_sum_for_offsets(
        std::vector<idi> &offsets)
{
    idi size_offsets = offsets.size();
    if (1 == size_offsets) {
        idi tmp = offsets[0];
        offsets[0] = 0;
        return tmp;
    } else if (size_offsets < 2048) {
        idi offset_sum = 0;
        idi size = size_offsets;
        for (idi i = 0; i < size; ++i) {
            idi tmp = offsets[i];
            offsets[i] = offset_sum;
            offset_sum += tmp;
        }
        return offset_sum;
    } else {
        // Parallel Prefix Sum, based on Guy E. Blelloch's Prefix Sums and Their Applications
        idi last_element = offsets[size_offsets - 1];
        //	idi size = 1 << ((idi) log2(size_offsets - 1) + 1);
        idi size = 1 << ((idi) log2(size_offsets));
        //	std::vector<idi> nodes(size, 0);
        idi tmp_element = offsets[size - 1];
        //#pragma omp parallel for
        //	for (idi i = 0; i < size_offsets; ++i) {
        //		nodes[i] = offsets[i];
        //	}

        // Up-Sweep (Reduce) Phase
        idi log2size = log2(size);
        for (idi d = 0; d < log2size; ++d) {
            idi by = 1 << (d + 1);
#pragma omp parallel for
            for (idi k = 0; k < size; k += by) {
                offsets[k + (1 << (d + 1)) - 1] += offsets[k + (1 << d) - 1];
            }
        }

        // Down-Sweep Phase
        offsets[size - 1] = 0;
        for (idi d = log2(size) - 1; d != (idi) -1; --d) {
            idi by = 1 << (d + 1);
#pragma omp parallel for
            for (idi k = 0; k < size; k += by) {
                idi t = offsets[k + (1 << d) - 1];
                offsets[k + (1 << d) - 1] = offsets[k + (1 << (d + 1)) - 1];
                offsets[k + (1 << (d + 1)) - 1] += t;
            }
        }

        //#pragma omp parallel for
        //	for (idi i = 0; i < size_offsets; ++i) {
        //		offsets[i] = nodes[i];
        //	}
        if (size != size_offsets) {
            idi tmp_sum = offsets[size - 1] + tmp_element;
            for (idi i = size; i < size_offsets; ++i) {
                idi t = offsets[i];
                offsets[i] = tmp_sum;
                tmp_sum += t;
            }
        }

        return offsets[size_offsets - 1] + last_element;
    }
}

// Parallelly collect elements of tmp_queue into the queue.
template<typename T>
inline void collect_into_queue(
//					std::vector<idi> &tmp_queue,
        std::vector <T> &tmp_queue,
        std::vector <idi> &offsets_tmp_queue, // the locations in tmp_queue for writing from tmp_queue
        std::vector <idi> &offsets_queue, // the locations in queue for writing into queue.
        idi num_elements, // total number of elements which need to be added from tmp_queue to queue
//					std::vector<idi> &queue,
        std::vector <T> &queue,
        idi &end_queue)
{
    if (0 == num_elements) {
        return;
    }
    idi i_bound = offsets_tmp_queue.size();
#pragma omp parallel for
    for (idi i = 0; i < i_bound; ++i) {
        idi i_q_start = end_queue + offsets_queue[i];
        idi i_q_bound;
        if (i_bound - 1 != i) {
            i_q_bound = end_queue + offsets_queue[i + 1];
        } else {
            i_q_bound = end_queue + num_elements;
        }
        if (i_q_start == i_q_bound) {
// If the group has no elements to be added, then continue to the next group
            continue;
        }
        idi end_tmp = offsets_tmp_queue[i];
        for (idi i_q = i_q_start; i_q < i_q_bound; ++i_q) {
            queue[i_q] = tmp_queue[end_tmp++];
        }
    }
    end_queue += num_elements;
}

template<typename T, typename Int>
inline void TS_enqueue(
        std::vector<T> &queue,
        Int &end_queue,
        const T &e)
{
    volatile Int old_i = end_queue;
    volatile Int new_i = old_i + 1;
    while (!CAS(&end_queue, old_i, new_i)) {
        old_i = end_queue;
        new_i = old_i + 1;
    }
    queue[old_i] = e;
}

} // namespace PADO



#endif /* INCLUDES_GLOBALS_H_ */
