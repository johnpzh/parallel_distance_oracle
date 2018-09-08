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

using std::string;

namespace PADO {
typedef uint64_t idi; // unsinged long long
typedef int weighti;
//const int IDI_MAX = ULLONG_MAX; // Maxiumum of unsigned long long, not very portable (such as on Mac)
const int WEIGHTI_MAX = INT_MAX;

// Compare and Swap
template <typename V_T>
inline bool CAS(V_T *ptr, V_T old_val, V_T new_val)
{
	if (1 == sizeof(V_T)) {
		return __sync_bool_compare_and_swap((char *)ptr, *((char *) &old_val), *((char *) &new_val));
	} else if (4 == sizeof(V_T)) {
		return __sync_bool_compare_and_swap((int *)ptr, *((int *) &old_val), *((int *) &new_val));
	} else if (8 == sizeof(V_T) && 8 == sizeof(long)) {
		return __sync_bool_compare_and_swap((long *)ptr, *((long *) &old_val), *((long *) &new_val));
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
	printf("%s: %f\n", item.c_str(), runtime);
}

} // namespace PADO



#endif /* INCLUDES_GLOBALS_H_ */
