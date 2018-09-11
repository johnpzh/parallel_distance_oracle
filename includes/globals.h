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
typedef uint8_t smalli;
typedef uint32_t inti;

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
// End Class WallTimer

class BitArray {
private:
	uint64_t *bits;
	unsigned size;

public:
	BitArray(unsigned len);
	BitArray(unsigned len, uint64_t num) : BitArray(len)
	{
		*bits = num;
	}
	~BitArray()
	{
		free(bits);
		size = 0;
	}

	// Execute fun for every set bit in bits (bit array)
	template<typename T> void process_every_bit(T fun); // fun is an expected function
	void set_bit(inti loc)
	{
		if (loc > size) {
			fprintf(stderr, "Error: BitArray::set_bit: loc is larger than size.\n");
			return;
		}
//		if (loc < 64) {
//			*bits |= ((uint64_t) 1 << loc);
//			return;
//		}
		inti remain = loc % 64;
		inti count = loc / 64;
		*((uint64_t *) ((uint8_t *) bits + count * 8)) |= ((uint64_t) 1 << remain);
	}
	void unset_bit(inti loc)
	{
		if (loc > size) {
			fprintf(stderr, "Error: BitArray::unset_bit: loc is larger than size.\n");
			return;
		}
//		if (loc < 64) {
//			*bits &= (~((uint64_t) 1 << loc));
//			return;
//		}
		inti remain = loc % 64;
		inti count = loc / 64;
		*((uint64_t *) ((uint8_t *) bits + count * 8)) &= (~((uint64_t) 1 << remain));
	}
};

BitArray::BitArray(unsigned len)
{
	if (len == 0 || len % 64 != 0) {
		fprintf(stderr, "Error: BitArray: the length should be divisible by 64\n");
		return;
	}
	size = len;
	unsigned b_l = len / 64;
	bits = (uint64_t *) calloc(b_l, sizeof(uint64_t));
	if (NULL == bits) {
		fprintf(stderr, "Error: BitArray: no enough memory for %u bytes.\n", b_l * 8);
		return;
	}
}

template<typename T>
void BitArray::process_every_bit(T fun)
{
	inti count = size / 64;
	for (inti c = 0; c < count; ++c) {
		uint64_t *num_64 = bits + c;
		for (inti i_32 = 0; i_32 < 8; i_32 += 4) {
			uint32_t *num_32 = (uint32_t *) ((uint8_t *) num_64 + i_32);
			for (inti i_16 = 0; i_16 < 4; i_16 += 2) {
				uint16_t *num_16 = (uint16_t *) ((uint8_t *) num_32 + i_16);
				for (inti i_8 = 0; i_8 < 2; i_8 += 1) {
					uint8_t *num_8 = (uint8_t *) ((uint8_t *) num_16 + i_8);
					printf("num_8: %u\n", *num_8);//test
					if (0 == *num_8) {
						continue;
					}
					inti offset = (i_8 + i_16 + i_32) * 8 + c * 64;
					for (inti i_1 = 0; i_1 < 8; ++i_1) {
						if (*num_8 & (1 << i_1)) {
							fun(i_1 + offset);
						}
					}
				}
			}
		}
	}
}




// End class BitArray

} // namespace PADO



#endif /* INCLUDES_GLOBALS_H_ */
