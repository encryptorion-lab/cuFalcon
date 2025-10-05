#ifndef RNG_CUH__
#define RNG_CUH__
#include <stdio.h>
#include <stdint.h>
#include "fpr.cuh"
#include "api.cuh"

/*
 * Structure for a PRNG. This includes a large buffer so that values
 * get generated in advance. The 'state' is used to keep the current
 * PRNG algorithm state (contents depend on the selected algorithm).
 *
 * The unions with 'dummy_u64' are there to ensure proper alignment for
 * 64-bit direct access.
 */
typedef struct {
	union {
		uint8_t d[512]; /* MUST be 512, exactly */
		uint64_t dummy_u64;
	} buf;
	size_t ptr;
	union {
		uint8_t d[256];
		uint64_t dummy_u64;
	} state;
	int type;
} prng_s;

// For GPU 
typedef struct {
	union {
		uint8_t *d;
		// cudaMalloc((void**) &d, 512 * sizeof(uint8_t));
		uint64_t dummy_u64;
	} buf;
	size_t ptr;
	union {
		uint8_t *d;
		// cudaMalloc((void**) &d, BATCH * 512 * sizeof(uint8_t));
		uint64_t dummy_u64;
	} state;
	int type;
} d_prng_s;


/* ==================================================================== */
/*
 * SHAKE256 implementation (shake.c).
 *
 * API is defined to be easily replaced with the fips202.h API defined
 * as part of PQClean.
 */

typedef struct {
	union {
		uint64_t A[25];
		uint8_t dbuf[200];
	} st;
	uint64_t dptr;
} inner_shake256_context_s;

// For GPU
typedef struct {
	union {
		uint64_t *A;
		uint8_t *dbuf;
	} st;
	uint64_t dptr;
} d_inner_shake256_context_s;

__device__ void prng_init_s(prng_s *p, inner_shake256_context_s *src);
__device__ void prng_refill_s(prng_s *p);

// #define inner_shake256_init      Zf(i_shake256_init)
// #define inner_shake256_inject    Zf(i_shake256_inject)
// #define inner_shake256_flip      Zf(i_shake256_flip)
// #define inner_shake256_extract   Zf(i_shake256_extract)

// void Zf(i_shake256_init)(
// 	inner_shake256_context *sc);
// void Zf(i_shake256_inject)(
// 	inner_shake256_context *sc, const uint8_t *in, size_t len);
// void Zf(i_shake256_flip)(
// 	inner_shake256_context *sc);
// void Zf(i_shake256_extract)(
// 	inner_shake256_context *sc, uint8_t *out, size_t len);
#endif