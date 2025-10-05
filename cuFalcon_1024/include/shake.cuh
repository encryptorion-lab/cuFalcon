#ifndef SHAKE_CUH__
#define SHAKE_CUH__

//#include <stdint.h>
//#include <stdio.h>
#include "api.cuh"

typedef struct {
    union {
        uint64_t A[BATCH*25];
        // uint8_t dbuf[BATCH*200];
        // uint64_t *A;
        // uint8_t *dbuf;        
    } st;
    uint64_t dptr[BATCH];
    // uint64_t *dptr;
} inner_shake256_context;


#define R64(a,b,c) (((a) << b) ^ ((a) >> c)) /* works on the GPU also for 
b = 64 or c = 64 */
#define NROUNDS 24
#define ROL(a, offset) (((a) << (offset)) ^ ((a) >> (64 - (offset))))
#define SHAKE256_RATE 136


/*
 * Round constants.
 */
static __device__ uint64_t RC[] = {
    0x0000000000000001, 0x0000000000008082,
    0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B,
    0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080,
    0x0000000080000001, 0x8000000080008008
};


static __constant__ uint64_t rc[5][NROUNDS] = {
    { 0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL },
    { 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL },
    { 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL },
    { 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL },
    { 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL } };

/* Rho-Offsets. Note that for each entry pair their respective sum is 64.
Only the first entry of each pair is a rho-offset. The second part is
used in the R64 macros. */
static __constant__  int ro[25][2] = {
    /*y=0*/         /*y=1*/         /*y=2*/         /*y=3*/         /*y=4*/
    /*x=0*/{ 0,64 }, /*x=1*/{ 44,20 }, /*x=2*/{ 43,21 }, /*x=3*/{ 21,43 }, /*x=4*/{ 14,50 },
    /*x=1*/{ 1,63 }, /*x=2*/{ 6,58 }, /*x=3*/{ 25,39 }, /*x=4*/{ 8,56 }, /*x=0*/{ 18,46 },
    /*x=2*/{ 62, 2 }, /*x=3*/{ 55, 9 }, /*x=4*/{ 39,25 }, /*x=0*/{ 41,23 }, /*x=1*/{ 2,62 },
    /*x=3*/{ 28,36 }, /*x=4*/{ 20,44 }, /*x=0*/{ 3,61 }, /*x=1*/{ 45,19 }, /*x=2*/{ 61, 3 },
    /*x=4*/{ 27,37 }, /*x=0*/{ 36,28 }, /*x=1*/{ 10,54 }, /*x=2*/{ 15,49 }, /*x=3*/{ 56, 8 } };

static __constant__  int a[25] = {
    0,  6, 12, 18, 24,
    1,  7, 13, 19, 20,
    2,  8, 14, 15, 21,
    3,  9, 10, 16, 22,
    4,  5, 11, 17, 23 };

static __constant__  int b[25] = {
    0,  1,  2,  3, 4,
    1,  2,  3,  4, 0,
    2,  3,  4,  0, 1,
    3,  4,  0,  1, 2,
    4,  0,  1,  2, 3 };

static __constant__  int c[25][3] = {
    { 0, 1, 2 },{ 1, 2, 3 },{ 2, 3, 4 },{ 3, 4, 0 },{ 4, 0, 1 },
    { 5, 6, 7 },{ 6, 7, 8 },{ 7, 8, 9 },{ 8, 9, 5 },{ 9, 5, 6 },
    { 10,11,12 },{ 11,12,13 },{ 12,13,14 },{ 13,14,10 },{ 14,10,11 },
    { 15,16,17 },{ 16,17,18 },{ 17,18,19 },{ 18,19,15 },{ 19,15,16 },
    { 20,21,22 },{ 21,22,23 },{ 22,23,24 },{ 23,24,20 },{ 24,20,21 } };

static __constant__  int d[25] = {
    0,  1,  2,  3,  4,
    10, 11, 12, 13, 14,
    20, 21, 22, 23, 24,
    5,  6,  7,  8,  9,
    15, 16, 17, 18, 19 };

__device__ void process_block(uint64_t *A);
__device__ void shake256_gpu(uint64_t *in) ;
__global__ void hash_to_point_vartime_par(uint64_t *scA, uint64_t  *scdptr, uint16_t *x);

__global__ void i_shake256_inject_gpu_kernel(uint64_t *d_scA, uint64_t *d_scdptr, const uint8_t *d_in, uint32_t len, size_t d_mem_pool_pitch);
__global__ void i_shake256_inject_gpu_kernel_2(uint64_t *d_scA, uint64_t *d_scdptr, const uint8_t *d_in, uint32_t len, size_t d_mem_pool_pitch);
__global__ void i_shake256_inject_gpu_kernel_666(uint64_t *d_scA, uint64_t *d_d_scdptr, const uint8_t *d_in, uint32_t *d_d_len, size_t d_mem_pool_pitch);
__global__ void i_shake256_flip_gpu_kernel(uint64_t *scA, uint64_t  *scdptr, size_t d_mem_pool_pitch);
__global__ void hash_to_point_vartime_par_kernel(uint64_t *d_scA, uint64_t *d_scdptr, uint16_t *d_x, size_t d_mem_pool_pitch);
__global__ void i_shake256_init_gpu_kernel(uint64_t *scA, uint64_t *d_scdptr, size_t d_mem_pool_pitch);

#endif