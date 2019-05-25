//
// Created by Zhen Peng on 5/25/19.
//

#ifndef PADO_PAPI_PADO_H
#define PADO_PAPI_PADO_H

#include <papi.h>
#include <locale.h>

namespace PADO {

// For PAPI, cache miss rate and instruction counts
// PAPI test results
// I named it L2 historically but actually it measures L3 cache.
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
//		long long counts[2];
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
            printf("Total_instructions_executed: %'lu\n", values[2]);
        } else {
            printf("%u %.2f\n", metrics, 1.0 * values[1]/values[0]);
            printf("%u %'lu\n", metrics, values[2]);
        }
    }

private:
    static const int num_events = 3;
    int events[num_events] = {PAPI_L3_TCA, PAPI_L3_TCM, PAPI_TOT_INS};
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
// End class L2CacheMissRate

//// For PAPI, cache miss rate
//// PAPI test results
//// I named it L2 historically but actually it measures L3 cache.
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
//		} else {
//			printf("%u %.2f\n", metrics, 1.0 * values[1]/values[0]);
//		}
//	}
//
//private:
//	static const int num_events = 2;
//	int events[num_events] = {PAPI_L3_TCA, PAPI_L3_TCM};
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
//
//// For PAPI, total instructions executed
//class TotalInstructsExe {
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
//		uint64_t counts[num_events];
//		if ((retval = PAPI_stop_counters((long long *) counts, num_events)) < PAPI_OK) {
//			test_fail(__FILE__, __LINE__, "measure_stop", retval);
//		} else {
//			value += counts[0];
//		}
//	}
//	void print(unsigned metrics = (unsigned) -1)
//	{
//		setlocale(LC_NUMERIC, "");
//		if (metrics == (unsigned) -1) {
//			printf("total_instructions_executed: %'lu\n", value);
//		} else {
//			printf("%u %lu\n", metrics, value);
//		}
//	}
//
//private:
//	static const int num_events = 1;
//	int events[num_events] = {PAPI_TOT_INS};
//	uint64_t value = 0;
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
//// End class TotalInstructsExe

} // End namespace PADO

#endif //PADO_PAPI_PADO_H
