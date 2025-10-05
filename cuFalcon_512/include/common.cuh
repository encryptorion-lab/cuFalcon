#include "fpr.cuh"

__device__ static const uint32_t l2bound[] = {
        0,    /* unused */
        101498,
        208714,
        428865,
        892039,
        1852696,
        3842630,
        7959734,
        16468416,
        34034726,
        70265242
};

__global__ void check1(int16_t* s1tmp, fpr *t0, uint16_t *hm, uint32_t *sqn, size_t d_mem_pool_pitch);

__global__ void check2(int16_t* s2tmp, fpr *t1, size_t d_mem_pool_pitch);

__global__ void is_short_half_gpu(uint32_t *d_sqn, const int16_t *d_s2, uint32_t *d_s, size_t d_mem_pool_pitch);

__global__ void comp_encode_gpu(uint8_t *buf, size_t max_out_len, const int16_t *x, unsigned logn, uint32_t *len, size_t d_mem_pool_pitch);

__global__ void byte_copy(uint8_t *out, uint8_t *in, uint32_t outlen, uint32_t inlen, size_t d_mem_pool_pitch);

__global__ void byte_copy_2(uint8_t *d_out, uint8_t *d_in, uint32_t outlen, uint32_t *inlen, size_t d_mem_pool_pitch);

__global__ void write_smlen_gpu(uint8_t *d_sm, uint32_t *d_sig_len, size_t d_mem_pool_pitch);

__global__ void modq_decode_gpu(uint16_t *x, unsigned logn, uint8_t *in, size_t max_in_len, size_t d_mem_pool_pitch);

__global__ void mq_NTT_tomonty(uint16_t *d_a, unsigned logn, size_t d_mem_pool_pitch);

__global__ void msg_len_gpu(uint8_t *d_sm, uint32_t *d_msg_len, uint32_t *d_smlen, uint32_t *d_sig_len, size_t d_mem_pool_pitch);

__global__ void comp_decode_gpu(int16_t *d_x, unsigned logn, uint8_t *d_in, uint32_t *d_in_len, uint32_t *d_msg_len, size_t d_mem_pool_pitch);

__global__ void comb_all_kernels(uint16_t *d_a, int16_t *d_s2, uint16_t *d_g, uint16_t *d_h, size_t d_mem_pool_pitch);

__global__ void is_short_gpu(int16_t *d_s1, int16_t *d_s2, uint32_t *d_s, size_t d_mem_pool_pitch);

__global__ void byte_cmp(uint8_t *d_m, uint8_t *d_m1, size_t d_mem_pool_pitch);
