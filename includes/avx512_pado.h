//
// Created by Zhen Peng on 5/25/19.
//

#ifndef PADO_AVX512_PADO_H
#define PADO_AVX512_PADO_H

#include <immintrin.h>

namespace PADO {
// AVX-512 constant variables
const inti NUM_P_INT = 16;
const __m512i INF_v = _mm512_set1_epi32(WEIGHTI_MAX);
const __m512i UNDEF_i32_v = _mm512_undefined_epi32();
const __m512i LOWEST_BYTE_MASK = _mm512_set1_epi32(0xFF);
const __m512i LOW_TWO_BYTE_MASK = _mm512_set1_epi32(0xFFFF);
const __m128i INF_v_128i = _mm_set1_epi8(-1);
const __m256i INF_v_256i = _mm256_set1_epi16(-1);
const __m512i INF_LARGE_v = _mm512_set1_epi32(WEIGHTILARGE_MAX);
//const inti NUM_P_BP_LABEL = 8;
//const inti REMAINDER_BP = BITPARALLEL_SIZE % NUM_P_BP_LABEL;
//const inti BOUND_BP_I = BITPARALLEL_SIZE - REMAINDER_BP;
//const __mmask8 IN_EPI64_M = (__mmask8) ((uint8_t) 0xFF >> (NUM_P_BP_LABEL - REMAINDER_BP));
//const __mmask16 IN_128I_M = (__mmask16) ((uint16_t) 0xFFFF >> (NUM_P_INT - REMAINDER_BP));
const __m512i INF_v_epi64 = _mm512_set1_epi64(WEIGHTI_MAX);
const __m512i ZERO_epi64_v = _mm512_set1_epi64(0);
const __m512i MINUS_2_epi64_v = _mm512_set1_epi64(-2);
const __m512i MINUS_1_epi64_v = _mm512_set1_epi64(-1);

} // End namespace PADO

#endif //PADO_AVX512_PADO_H
