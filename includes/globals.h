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
//#include <vector>
#include <string.h>

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
//inline bool CAS(V_T *ptr, V_T old_val, V_T new_val)
inline bool CAS(void *ptr, V_T old_val, V_T new_val)
{
	if (1 == sizeof(V_T)) {
		return __sync_bool_compare_and_swap((uint8_t *) ptr, *((uint8_t *) &old_val), *((uint8_t *) &new_val));
	} else if (2 == sizeof(V_T)) {
		return __sync_bool_compare_and_swap((uint16_t *) ptr, *((uint16_t *) &old_val), *((uint16_t *) &new_val));
	} else if (4 == sizeof(V_T)) {
		return __sync_bool_compare_and_swap((uint32_t *) ptr, *((uint32_t *) &old_val), *((uint32_t *) &new_val));
	} else if (8 == sizeof(V_T)) {
		return __sync_bool_compare_and_swap((uint64_t *) ptr, *((uint64_t *) &old_val), *((uint64_t *) &new_val));
	} else {
		printf("CAS cannot support the type.\n");
		exit(EXIT_FAILURE);
	}
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



} // namespace PADO



#endif /* INCLUDES_GLOBALS_H_ */
