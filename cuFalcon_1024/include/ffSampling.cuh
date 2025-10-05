
#include "fpr.cuh"
#include "api.cuh"
#include "rng.cuh"


typedef struct
{
    fpr *t0;
    fpr *g00;
    fpr *g11;
    unsigned logn;
    unsigned char is_z0, is_z1;
} STACK;

typedef struct
{
    fpr *t0;
    fpr *t1;
    fpr *z0;
    fpr *z1;
    fpr *tmp;
    fpr *tree;
    unsigned logn;
    unsigned char is_z0, is_z1;
} STACK2;

typedef struct {
    prng_s p;
    fpr sigma_min;
} sampler_context_s;

typedef struct {
    prng_s p;
    fpr sigma_min;
} d_sampler_context_s;

__global__ void ffSampling_fft_dyntree(fpr *d_t0, fpr *d_g00, fpr *d_g11, unsigned orig_logn, unsigned logn, uint64_t *d_scA, uint64_t *scdptr, size_t d_mem_pool_pitch);

__device__ void poly_LDL_fft(const fpr *g00, fpr *g01, fpr *g11, unsigned logn);

__device__ void poly_split_fft(fpr *f0, fpr *f1, const fpr *f, unsigned logn);

__device__ int sampler(sampler_context_s *spc, fpr mu, fpr isigma);

__device__  void poly_sub(fpr *a, const fpr *b, unsigned logn);

__device__  void poly_mul_fft(fpr *a, const fpr *b, unsigned logn);

__device__ void poly_merge_fft(fpr *f, const fpr *f0, const fpr *f1, unsigned logn);

__device__  void poly_add(fpr *a, const fpr *b, unsigned logn);