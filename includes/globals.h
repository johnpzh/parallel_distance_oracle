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
#include <vector>
#include <string.h>
#include <locale.h>
#include <papi.h>

using std::string;
using std::vector;

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
	WallTimer(const char *n) : item(n)
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

//// For PAPI, cache miss rate and instruction counts
//// PAPI test results
//class L2CacheMissRate {
//public:
//	void measure_start()
//	{
//		int retval;
//		if ((retval = PAPI_start_counters(events, num_events)) < PAPI_OK) {
//			test_fail(__FILE__, __LINE__, "measure_start", retval);
//		}
//	}
//	void measure_stop()
//	{
//		int retval;
////		long long counts[2];
//		uint64_t counts[num_events];
//		if ((retval = PAPI_stop_counters((long long *) counts, num_events)) < PAPI_OK) {
//			test_fail(__FILE__, __LINE__, "measure_stop", retval);
//		} else {
//			for (int i = 0; i < num_events; ++i) {
//				values[i] += counts[i];
//			}
//		}
//	}
//	void print(unsigned metrics = (unsigned) -1)
//	{
//		setlocale(LC_NUMERIC, "");
//		if (metrics == (unsigned) -1) {
//			printf("L3_cache_access: %'lu cache_misses: %'lu miss_rate: %.2f%%\n", values[0], values[1], values[1] * 100.0 / values[0]);
//			printf("Total_instructions_executed: %'lu\n", values[2]);
//		} else {
//			printf("%u %.2f\n", metrics, 1.0 * values[1]/values[0]);
//			printf("%u %'lu\n", metrics, values[2]);
//		}
//	}
//
//private:
//	static const int num_events = 3;
//	int events[num_events] = {PAPI_L3_TCA, PAPI_L3_TCM, PAPI_TOT_INS};
//	vector<uint64_t> values = vector<uint64_t>(num_events, 0);
//
//	void test_fail(const char *file, int line, const char *call, int retval)
//	{
//		printf("%s\tFAILED\nLine # %d\n", file, line);
//		if ( retval == PAPI_ESYS ) {
//			char buf[128];
//			memset( buf, '\0', sizeof(buf) );
//			sprintf(buf, "System error in %s:", call );
//			perror(buf);
//		}
//		else if ( retval > 0 ) {
//			printf("Error calculating: %s\n", call );
//		}
//		else {
//			printf("Error in %s: %s\n", call, PAPI_strerror(retval) );
//		}
//		printf("\n");
//		exit(1);
//	}
//};
// End class L2CacheMissRate

// For PAPI, cache miss rate
// PAPI test results
class L2CacheMissRate {
public:
	void measure_start()
	{
		int retval;
		if ((retval = PAPI_start_counters(events, num_events)) < PAPI_OK) {
			test_fail(__FILE__, __LINE__, "measure_start", retval);
		}
	}
	void measure_stop()
	{
		int retval;
		uint64_t counts[num_events];
		if ((retval = PAPI_stop_counters((long long *) counts, num_events)) < PAPI_OK) {
			test_fail(__FILE__, __LINE__, "measure_stop", retval);
		} else {
			for (int i = 0; i < num_events; ++i) {
				values[i] += counts[i];
			}
		}
	}
	void print(unsigned metrics = (unsigned) -1)
	{
		setlocale(LC_NUMERIC, "");
		if (metrics == (unsigned) -1) {
			printf("L3_cache_access: %'lu cache_misses: %'lu miss_rate: %.2f%%\n", values[0], values[1], values[1] * 100.0 / values[0]);
		} else {
			printf("%u %.2f\n", metrics, 1.0 * values[1]/values[0]);
		}
	}

private:
	static const int num_events = 2;
	int events[num_events] = {PAPI_L3_TCA, PAPI_L3_TCM};
	vector<uint64_t> values = vector<uint64_t>(num_events, 0);

	void test_fail(const char *file, int line, const char *call, int retval)
	{
		printf("%s\tFAILED\nLine # %d\n", file, line);
		if ( retval == PAPI_ESYS ) {
			char buf[128];
			memset( buf, '\0', sizeof(buf) );
			sprintf(buf, "System error in %s:", call );
			perror(buf);
		}
		else if ( retval > 0 ) {
			printf("Error calculating: %s\n", call );
		}
		else {
			printf("Error in %s: %s\n", call, PAPI_strerror(retval) );
		}
		printf("\n");
		exit(1);
	}
};

// For PAPI, total instructions executed
class TotalInstructsExe {
public:
	void measure_start()
	{
		int retval;
		if ((retval = PAPI_start_counters(events, num_events)) < PAPI_OK) {
			test_fail(__FILE__, __LINE__, "measure_start", retval);
		}
	}
	void measure_stop()
	{
		int retval;
		uint64_t counts[num_events];
		if ((retval = PAPI_stop_counters((long long *) counts, num_events)) < PAPI_OK) {
			test_fail(__FILE__, __LINE__, "measure_stop", retval);
		} else {
			value += counts[0];
		}
	}
	void print(unsigned metrics = (unsigned) -1)
	{
		setlocale(LC_NUMERIC, "");
		if (metrics == (unsigned) -1) {
			printf("total_instructions_executed: %'lu\n", value);
		} else {
			printf("%u %lu\n", metrics, value);
		}
	}

private:
	static const int num_events = 1;
	int events[num_events] = {PAPI_TOT_INS};
	uint64_t value = 0;

	void test_fail(const char *file, int line, const char *call, int retval)
	{
		printf("%s\tFAILED\nLine # %d\n", file, line);
		if ( retval == PAPI_ESYS ) {
			char buf[128];
			memset( buf, '\0', sizeof(buf) );
			sprintf(buf, "System error in %s:", call );
			perror(buf);
		}
		else if ( retval > 0 ) {
			printf("Error calculating: %s\n", call );
		}
		else {
			printf("Error in %s: %s\n", call, PAPI_strerror(retval) );
		}
		printf("\n");
		exit(1);
	}
};
// End class TotalInstructsExe

} // namespace PADO



#endif /* INCLUDES_GLOBALS_H_ */
