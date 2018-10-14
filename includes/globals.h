/*
 * globals.h
 *
 *  Created on: Sep 4, 2018
 *      Author: Zhen Peng
 */

#ifndef INCLUDES_GLOBALS_H_
#define INCLUDES_GLOBALS_H_


#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include <string>
#include <string.h>

using std::string;

namespace PADO {
//typedef uint64_t idi; // unsinged long long
typedef uint32_t idi; // unsigned int
//typedef int weighti;
typedef uint8_t weighti;
typedef uint8_t smalli;
typedef uint32_t inti;
//const int WEIGHTI_MAX = INT_MAX;
const uint8_t WEIGHTI_MAX = UCHAR_MAX;
const uint8_t SMALLI_MAX = UCHAR_MAX;


// Compare and Swap
template <typename V_T>
//inline bool CAS(V_T *ptr, V_T old_val, V_T new_val)
inline bool CAS(void *ptr, V_T old_val, V_T new_val)
{
	if (1 == sizeof(V_T)) {
		return __sync_bool_compare_and_swap((uint8_t *) ptr, *((uint8_t *) &old_val), *((uint8_t *) &new_val));
	} else if (4 == sizeof(V_T)) {
		return __sync_bool_compare_and_swap((uint32_t *) ptr, *((uint32_t *) &old_val), *((uint32_t *) &new_val));
//		return __sync_bool_compare_and_swap((int *) ptr, *((int *) &old_val), *((int *) &new_val));
//		return __sync_bool_compare_and_swap(reinterpret_cast<uint32_t *>(ptr), *(reinterpret_cast<uint32_t *>(&old_val)), *(reinterpret_cast<uint32_t *>(&new_val)));
//		return __sync_bool_compare_and_swap(((uint32_t *) &((uint32_t) *ptr)), *((uint32_t *) &old_val), *((uint32_t *) &new_val));
	} else if (8 == sizeof(V_T)) {
		return __sync_bool_compare_and_swap((uint64_t *) ptr, *((uint64_t *) &old_val), *((uint64_t *) &new_val));
	} else {
		printf("CAS cannot support the type.\n");
		exit(EXIT_FAILURE);
	}
}


class WallTimer {
private:
	double start = 0.0;
	string item;
	void construct();
public:
	WallTimer();
	WallTimer(const char *n);
	double get_runtime();
	void print_runtime();
	static double get_time_mark()
	{
		timeval t;
		gettimeofday(&t, NULL);
		return t.tv_sec + t.tv_usec * 0.000001;
	}
};

WallTimer::WallTimer()
{
	construct();
}

WallTimer::WallTimer(const char *n) : item(n)
{
	construct();
}

void WallTimer::construct()
{
	timeval t;
	gettimeofday(&t, NULL);
	start = t.tv_sec + t.tv_usec * 0.000001;
}

double WallTimer::get_runtime()
{
	timeval t;
	gettimeofday(&t, NULL);
	double now = t.tv_sec + t.tv_usec * 0.000001;
	return now - start;
}

void WallTimer::print_runtime()
{
	double runtime = get_runtime();
	printf("%s: %f\n", item.c_str(), runtime); fflush(stdout);
}
// End Class WallTimer



} // namespace PADO



#endif /* INCLUDES_GLOBALS_H_ */
