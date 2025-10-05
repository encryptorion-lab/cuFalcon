/*
 * This file is part of cuFalcon.
 *
 * Copyright (c) 2025 Wenqian Li, et al.
 *
 * Licensed under the GNU General Public License v3.0 (GPLv3)
 * See the LICENSE file in the project root for license details.
 */

#include "../include/fpr.cuh"
#include "../include/fft.cuh"

__global__ void sign_dyn(fpr *d_b0, fpr *d_b1, fpr *d_b2, fpr *d_b3, fpr *d_b4, fpr *d_b5, uint16_t *d_hm, size_t d_mem_pool_pitch)
{
    fpr *b0 = d_b0 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b1 = d_b1 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b2 = d_b2 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b3 = d_b3 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b4 = d_b4 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b5 = d_b5 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    uint16_t *hm = d_hm + blockIdx.x * d_mem_pool_pitch / sizeof(uint16_t);

    fpr a_re, a_im, b_re, b_im;
    fpr tmp1;

    for(int i=threadIdx.x;i<1024;i+=blockDim.x){
        b5[i] = b0[i];
        b4[i] = b1[i];
    }
    __syncthreads();

    a_re = b0[threadIdx.x];
    a_im = b0[threadIdx.x + 512];
    b0[threadIdx.x] = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b0[threadIdx.x + 512] = fpr_zero;

    a_re = b0[threadIdx.x + 256];
    a_im = b0[threadIdx.x + 512 + 256];
    b0[threadIdx.x + 256] = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b0[threadIdx.x + 512 + 256] = fpr_zero;

    a_re = b1[threadIdx.x];
    a_im = b1[threadIdx.x + 512];
    tmp1 = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b0[threadIdx.x] = fpr_add(b0[threadIdx.x], tmp1);

    a_re = b1[threadIdx.x + 256];
    a_im = b1[threadIdx.x + 256 + 512];
    tmp1 = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b0[threadIdx.x + 256] = fpr_add(b0[threadIdx.x + 256], tmp1);
    __syncthreads();

    a_re = b1[threadIdx.x];
    a_im = b1[threadIdx.x + 512];
    b_re = b3[threadIdx.x];
    b_im = fpr_neg(b3[threadIdx.x + 512]);
    FPC_MUL(b1[threadIdx.x], b1[threadIdx.x + 512], a_re, a_im, b_re, b_im);

    a_re = b1[threadIdx.x + 256];
    a_im = b1[threadIdx.x + 256 + 512];
    b_re = b3[threadIdx.x + 256];
    b_im = fpr_neg(b3[threadIdx.x + 256 + 512]);
    FPC_MUL(b1[threadIdx.x + 256], b1[threadIdx.x + 256 + 512], a_re, a_im, b_re, b_im);

    a_re = b5[threadIdx.x];
    a_im = b5[threadIdx.x + 512];
    b_re = b2[threadIdx.x];
    b_im = fpr_neg(b2[threadIdx.x + 512]);
    FPC_MUL(b5[threadIdx.x], b5[threadIdx.x + 512], a_re, a_im, b_re, b_im);

    a_re = b5[threadIdx.x + 256];
    a_im = b5[threadIdx.x + 256 + 512];
    b_re = b2[threadIdx.x + 256];
    b_im = fpr_neg(b2[threadIdx.x + 256 + 512]);
    FPC_MUL(b5[threadIdx.x + 256], b5[threadIdx.x + 256 + 512], a_re, a_im, b_re, b_im);


    b1[threadIdx.x] = fpr_add(b1[threadIdx.x], b5[threadIdx.x]);
    b1[threadIdx.x + 512] = fpr_add(b1[threadIdx.x + 512], b5[threadIdx.x + 512]);

    b1[threadIdx.x + 256] = fpr_add(b1[threadIdx.x + 256], b5[threadIdx.x + 256]);
    b1[threadIdx.x + 256 + 512] = fpr_add(b1[threadIdx.x + 256 + 512], b5[threadIdx.x + 256 + 512]);

    a_re = b2[threadIdx.x];
    a_im = b2[threadIdx.x + 512];
    b2[threadIdx.x] = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b2[threadIdx.x + 512] = fpr_zero;

    a_re = b2[threadIdx.x + 256];
    a_im = b2[threadIdx.x + 256 + 512];
    b2[threadIdx.x + 256] = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b2[threadIdx.x + 256 + 512] = fpr_zero;

    a_re = b3[threadIdx.x];
    a_im = b3[threadIdx.x + 512];
    tmp1 = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b2[threadIdx.x] = fpr_add(b2[threadIdx.x], tmp1);

    a_re = b3[threadIdx.x + 256];
    a_im = b3[threadIdx.x + 256 + 512];
    tmp1 = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b2[threadIdx.x + 256] = fpr_add(b2[threadIdx.x + 256], tmp1);

    fpr reg0,reg1,reg2,reg3;
    __shared__ fpr s_f[Falcon_N+Falcon_N/2];


    size_t idx1 = threadIdx.x;
    size_t idx2 = threadIdx.x + threadIdx.x;

    reg0 = fpr_of(hm[idx1]);
    reg2 = fpr_of(hm[idx1 + 256]);
    reg1 = fpr_of(hm[idx1 + 512]);
    reg3 = fpr_of(hm[idx1 + 512 + 256]);

    fft1024(reg0, reg1, reg2, reg3, s_f);

    FPC_ADD(b5[idx2], b5[idx2 + 512],
            reg0, reg1, reg2, reg3);
    FPC_SUB(b5[idx2 + 1], b5[idx2 + 1 + 512],
            reg0, reg1, reg2, reg3);
    __syncthreads();

    for(int i=threadIdx.x;i<512;i+=blockDim.x){
        reg0 = b5[i];
        reg1 = b5[i + 512];
        reg2 = b4[i];
        reg3 = b4[i + 512];
        FPC_MUL(reg0, reg1, reg0, reg1, reg2, reg3);
        b4[i] = fpr_mul(reg0, fpr_n(fpr_inverse_of_q));
        b4[i + 512] = fpr_mul(reg1, fpr_n(fpr_inverse_of_q));
    }

    for(int i=threadIdx.x;i<512;i+=blockDim.x){
        reg0 = b5[i];
        reg1 = b5[i + 512];
        reg2 = b3[i];
        reg3 = b3[i + 512];
        FPC_MUL(reg0, reg1, reg0, reg1, reg2, reg3);
        b3[i] = fpr_mul(reg0, fpr_inverse_of_q);
        b3[i + 512] = fpr_mul(reg1, fpr_inverse_of_q);
    }
    __syncthreads();

}

__global__ void fft_polymul(fpr *d_b0, fpr *d_b1, fpr *d_b2, fpr *d_b3, fpr *d_b4, fpr *d_b5, fpr *d_b6, fpr *d_b7, size_t d_mem_pool_pitch)
{
    fpr *b0 = d_b0 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b1 = d_b1 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b2 = d_b2 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b3 = d_b3 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b4 = d_b4 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b5 = d_b5 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b6 = d_b6 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b7 = d_b7 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);

    fpr a_re, a_im, b_re, b_im, tmp1, tmp2;

    a_re = b4[threadIdx.x];
    a_im = b4[threadIdx.x + 512];
    b_re = b0[threadIdx.x];
    b_im = b0[threadIdx.x + 512];

    FPC_MUL(tmp1, tmp2, a_re, a_im, b_re, b_im);

    a_re = b5[threadIdx.x];
    a_im = b5[threadIdx.x + 512];
    b_re = b2[threadIdx.x];
    b_im = b2[threadIdx.x + 512];
    FPC_MUL(b6[threadIdx.x], b6[threadIdx.x + 512], a_re, a_im, b_re, b_im);

    b6[threadIdx.x] = fpr_add(b6[threadIdx.x], tmp1);
    b6[threadIdx.x + 512] = fpr_add(b6[threadIdx.x + 512], tmp2);

    a_re = b4[threadIdx.x];
    a_im = b4[threadIdx.x + 512];
    b_re = b1[threadIdx.x];
    b_im = b1[threadIdx.x + 512];
    FPC_MUL(tmp1, tmp2, a_re, a_im, b_re, b_im);

    a_re = b5[threadIdx.x];
    a_im = b5[threadIdx.x + 512];
    b_re = b3[threadIdx.x];
    b_im = b3[threadIdx.x + 512];
    FPC_MUL(b5[threadIdx.x], b5[threadIdx.x + 512], a_re, a_im, b_re, b_im);

    b5[threadIdx.x] = fpr_add(b5[threadIdx.x], tmp1);
    b5[threadIdx.x + 512] = fpr_add(b5[threadIdx.x + 512], tmp2);

    b4[threadIdx.x] = b6[threadIdx.x];
    b4[threadIdx.x + 512] = b6[threadIdx.x + 512];

}

__global__ void ifft_t(fpr *d_b4, fpr *d_b5, size_t d_mem_pool_pitch){


    fpr *b4 = d_b4 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *b5 = d_b5 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;

    fpr reg0,reg1,reg2,reg3;
    __shared__ fpr s_f[Falcon_N+Falcon_N/2];

    size_t butt_idx = threadIdx.x + threadIdx.x;
    reg0 = b4[butt_idx];
    reg1 = b4[butt_idx + 512];
    reg2 = b4[butt_idx + 1];
    reg3 = b4[butt_idx + 1 + 512];
    ifft1024(reg0,reg1,reg2,reg3,s_f);
    __syncthreads();
    for (int i = threadIdx.x; i < Falcon_N; i +=blockDim.x) {
        b4[i] = fpr_mul(s_f[i], fpr_p2_tab[10]);
    }
    __syncthreads();

    reg0 = b5[butt_idx];
    reg1 = b5[butt_idx + 512];
    reg2 = b5[butt_idx + 1];
    reg3 = b5[butt_idx + 1 + 512];
    ifft1024(reg0,reg1,reg2,reg3,s_f);
    __syncthreads();
    for (int i = threadIdx.x; i < Falcon_N; i +=blockDim.x) {
        b5[i] = fpr_mul(s_f[i], fpr_p2_tab[10]);
    }

}

__global__ void convert_B_fft(fpr *d_f0, fpr *d_f1, fpr *d_f2, fpr *d_f3, size_t d_mem_pool_pitch)
{

    fpr reg0,reg1,reg2,reg3;
    __shared__ fpr s_f[Falcon_N+Falcon_N/2];
    fpr *f0 = d_f0 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *f1 = d_f1 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *f2 = d_f2 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *f3 = d_f3 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;

    size_t idx1 = threadIdx.x;
    size_t idx2 = threadIdx.x + threadIdx.x;


    reg0 = f0[idx1];
    reg2 = f0[idx1 + 256];
    reg1 = f0[idx1 + 512];
    reg3 = f0[idx1 + 256 + 512];

    fft1024(reg0, reg1, reg2, reg3, s_f);

    FPC_ADD(f0[idx2], f0[idx2 + 512],
            reg0, reg1, reg2, reg3);
    FPC_SUB(f0[idx2 + 1], f0[idx2 + 1 + 512],
            reg0, reg1, reg2, reg3);
    __syncthreads();


    reg0 = f1[idx1];
    reg2 = f1[idx1 + 256];
    reg1 = f1[idx1 + 512];
    reg3 = f1[idx1 + 256 + 512];

    fft1024(reg0, reg1, reg2, reg3, s_f);

    FPC_ADD(s_f[2 * threadIdx.x], s_f[2 * threadIdx.x + 512],
            reg0, reg1, reg2, reg3);
    FPC_SUB(s_f[2 * threadIdx.x + 1], s_f[2 * threadIdx.x + 1 + 512],
            reg0, reg1, reg2, reg3); //FPC_SUB_NEG
    __syncthreads();
    f1[idx1] = fpr_neg(s_f[threadIdx.x]);
    f1[idx1 + 512] = fpr_neg(s_f[threadIdx.x + 512]);
    f1[idx1 + 256] = fpr_neg(s_f[threadIdx.x + 256]);
    f1[idx1 + 512 + 256] = fpr_neg(s_f[threadIdx.x + 512 + 256]);
    __syncthreads();


    reg0 = f2[idx1];
    reg2 = f2[idx1 + 256];
    reg1 = f2[idx1 + 512];
    reg3 = f2[idx1 + 256 + 512];

    fft1024(reg0, reg1, reg2, reg3, s_f);

    FPC_ADD(f2[idx2], f2[idx2 + 512],
            reg0, reg1, reg2, reg3);
    FPC_SUB(f2[idx2 + 1], f2[idx2 + 1 + 512],
            reg0, reg1, reg2, reg3);
    __syncthreads();


    reg0 = f3[idx1];
    reg2 = f3[idx1 + 256];
    reg1 = f3[idx1 + 512];
    reg3 = f3[idx1 + 256 + 512];

    fft1024(reg0, reg1, reg2, reg3, s_f);

    FPC_ADD(s_f[2 * threadIdx.x], s_f[2 * threadIdx.x + 512],
            reg0, reg1, reg2, reg3);
    FPC_SUB(s_f[2 * threadIdx.x + 1], s_f[2 * threadIdx.x + 1 + 512],
            reg0, reg1, reg2, reg3);
    __syncthreads();
    f3[idx1] = fpr_neg(s_f[threadIdx.x]);
    f3[idx1 + 256] = fpr_neg(s_f[threadIdx.x + 256]);
    f3[idx1 + 512] = fpr_neg(s_f[threadIdx.x + 512]);
    f3[idx1 + 512 + 256] = fpr_neg(s_f[threadIdx.x + 512 + 256]);

}


__global__ void recompute_B_fft(fpr *d_f0, fpr *d_f1, fpr *d_f2, fpr *d_f3, fpr *d_f4, fpr *d_f5,
                                const int8_t *d_G, const int8_t *d_f, const int8_t *d_g, const int8_t *d_F, size_t d_mem_pool_pitch)
{

    fpr reg0,reg1,reg2,reg3;
    __shared__ fpr s_f[Falcon_N+Falcon_N/2];
    fpr *f0 = d_f0 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *f1 = d_f1 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *f2 = d_f2 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *f3 = d_f3 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *f4 = d_f4 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *f5 = d_f5 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    const int8_t *G = d_G + blockIdx.x * d_mem_pool_pitch;
    const int8_t *f = d_f + blockIdx.x * d_mem_pool_pitch;
    const int8_t *g = d_g + blockIdx.x * d_mem_pool_pitch;
    const int8_t *F = d_F + blockIdx.x * d_mem_pool_pitch;

    f5[threadIdx.x] = f4[threadIdx.x];
    f5[256 + threadIdx.x] = f4[256 + threadIdx.x];
    f5[512 + threadIdx.x] = f4[512 + threadIdx.x];
    f5[768 + threadIdx.x] = f4[768 + threadIdx.x];

    f4[threadIdx.x] = f3[threadIdx.x];
    f4[256 + threadIdx.x] = f3[256 + threadIdx.x];
    f4[512 + threadIdx.x] = f3[512 + threadIdx.x];
    f4[768 + threadIdx.x] = f3[768 + threadIdx.x];

    size_t idx1 = threadIdx.x;
    size_t idx2 = threadIdx.x + threadIdx.x;


    reg0.v = (double)g[idx1];
    reg2.v = (double)g[idx1 + 256];
    reg1.v = (double)g[idx1 + 512];
    reg3.v = (double)g[idx1 + 512 + 256];

    fft1024(reg0, reg1, reg2, reg3, s_f);

    FPC_ADD(f0[idx2], f0[idx2 + 512],
            reg0, reg1, reg2, reg3);
    FPC_SUB(f0[idx2 + 1], f0[idx2 + 1 + 512],
            reg0, reg1, reg2, reg3);
    __syncthreads();

    reg0.v = (double)f[idx1];
    reg2.v = (double)f[idx1 + 256];
    reg1.v = (double)f[idx1 + 512];
    reg3.v = (double)f[idx1 + 512 + 256];

    fft1024(reg0, reg1, reg2, reg3, s_f);

    FPC_ADD(s_f[2 * threadIdx.x], s_f[2 * threadIdx.x + 512],
            reg0, reg1, reg2, reg3);
    FPC_SUB(s_f[2 * threadIdx.x + 1], s_f[2 * threadIdx.x + 1 + 512],
            reg0, reg1, reg2, reg3);
    __syncthreads();
    f1[idx1] = fpr_neg(s_f[threadIdx.x]);
    f1[idx1 + 256] = fpr_neg(s_f[threadIdx.x + 256]);
    f1[idx1 + 512] = fpr_neg(s_f[threadIdx.x + 512]);
    f1[idx1 + 512 + 256] = fpr_neg(s_f[threadIdx.x + 512 + 256]);
    __syncthreads();


    reg0.v = (double)G[idx1];
    reg2.v = (double)G[idx1 + 256];
    reg1.v = (double)G[idx1 + 512];
    reg3.v = (double)G[idx1 + 512 + 256];

    fft1024(reg0, reg1, reg2, reg3, s_f);

    FPC_ADD(f2[idx2], f2[idx2 + 512],
            reg0, reg1, reg2, reg3);
    FPC_SUB(f2[idx2 + 1], f2[idx2 + 1 + 512],
            reg0, reg1, reg2, reg3);
    __syncthreads();


    reg0.v = (double)F[idx1];
    reg2.v = (double)F[idx1 + 256];
    reg1.v = (double)F[idx1 + 512];
    reg3.v = (double)F[idx1 + 512 + 256];


    fft1024(reg0, reg1, reg2, reg3, s_f);

    FPC_ADD(s_f[2 * threadIdx.x], s_f[2 * threadIdx.x + 512],
            reg0, reg1, reg2, reg3);
    FPC_SUB(s_f[2 * threadIdx.x + 1], s_f[2 * threadIdx.x + 1 + 512],
            reg0, reg1, reg2, reg3);
    __syncthreads();
    f3[idx1] = fpr_neg(s_f[threadIdx.x]);
    f3[idx1 + 256] = fpr_neg(s_f[threadIdx.x + 256]);
    f3[idx1 + 512] = fpr_neg(s_f[threadIdx.x + 512]);
    f3[idx1 + 512 + 256] = fpr_neg(s_f[threadIdx.x + 512 + 256]);
}

__global__ void sign_balance(fpr *d_b0, const fpr *d_nntbb, uint16_t *d_hm, size_t d_mem_pool_pitch)
{
    fpr *b0 = d_b0 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b1 = b0 + Falcon_N;
    fpr *b2 = b1 + Falcon_N;
    fpr *b3 = b2 + Falcon_N;
    fpr *b4 = b3 + Falcon_N;
    fpr *b5 = b4 + Falcon_N;
    const fpr *d_nnt0 = d_nntbb;
    const fpr *d_nnt1 = d_nnt0 + Falcon_N;
    const fpr *d_nnt2 = d_nnt1 + Falcon_N;
    const fpr *d_nnt3 = d_nnt2 + Falcon_N;
    uint16_t *hm = d_hm + blockIdx.x * d_mem_pool_pitch / sizeof(uint16_t);

    fpr a_re, a_im, b_re, b_im;
    fpr tmp1;
    for(int i=threadIdx.x;i<1024;i+=blockDim.x){
        b5[i] = d_nnt0[i];
        b4[i] = d_nnt1[i];
    }
    __syncthreads();

    a_re = d_nnt0[threadIdx.x];
    a_im = d_nnt0[threadIdx.x + 512];
    b0[threadIdx.x] = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b0[threadIdx.x + 512] = fpr_zero;

    a_re = d_nnt0[threadIdx.x + 256];
    a_im = d_nnt0[threadIdx.x + 256 + 512];
    b0[threadIdx.x + 256] = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b0[threadIdx.x + 256 + 512] = fpr_zero;

    a_re = d_nnt1[threadIdx.x];
    a_im = d_nnt1[threadIdx.x + 512];
    tmp1 = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b0[threadIdx.x] = fpr_add(b0[threadIdx.x], tmp1);

    a_re = d_nnt1[threadIdx.x + 256];
    a_im = d_nnt1[threadIdx.x + 256 + 512];
    tmp1 = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b0[threadIdx.x + 256] = fpr_add(b0[threadIdx.x + 256], tmp1);
    __syncthreads();

    a_re = d_nnt1[threadIdx.x];
    a_im = d_nnt1[threadIdx.x + 512];
    b_re = d_nnt3[threadIdx.x];
    b_im = fpr_neg(d_nnt3[threadIdx.x + 512]);
    FPC_MUL(b1[threadIdx.x], b1[threadIdx.x + 512], a_re, a_im, b_re, b_im);

    a_re = d_nnt1[threadIdx.x + 256];
    a_im = d_nnt1[threadIdx.x + 256 + 512];
    b_re = d_nnt3[threadIdx.x + 256];
    b_im = fpr_neg(d_nnt3[threadIdx.x + 256 + 512]);
    FPC_MUL(b1[threadIdx.x + 256], b1[threadIdx.x + 256 + 512], a_re, a_im, b_re, b_im);

    a_re = b5[threadIdx.x];
    a_im = b5[threadIdx.x + 512];
    b_re = d_nnt2[threadIdx.x];
    b_im = fpr_neg(d_nnt2[threadIdx.x + 512]);
    FPC_MUL(b5[threadIdx.x], b5[threadIdx.x + 512], a_re, a_im, b_re, b_im);

    a_re = b5[threadIdx.x + 256];
    a_im = b5[threadIdx.x + 256 + 512];
    b_re = d_nnt2[threadIdx.x + 256];
    b_im = fpr_neg(d_nnt2[threadIdx.x + 256 + 512]);
    FPC_MUL(b5[threadIdx.x + 256], b5[threadIdx.x + 256 + 512], a_re, a_im, b_re, b_im);


    b1[threadIdx.x] = fpr_add(b1[threadIdx.x], b5[threadIdx.x]);
    b1[threadIdx.x + 512] = fpr_add(b1[threadIdx.x + 512], b5[threadIdx.x + 512]);

    b1[threadIdx.x + 256] = fpr_add(b1[threadIdx.x + 256], b5[threadIdx.x + 256]);
    b1[threadIdx.x + 256 + 512] = fpr_add(b1[threadIdx.x + 256 + 512], b5[threadIdx.x + 256 + 512]);

    a_re = d_nnt2[threadIdx.x];
    a_im = d_nnt2[threadIdx.x + 512];
    b2[threadIdx.x] = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b2[threadIdx.x + 512] = fpr_zero;

    a_re = d_nnt2[threadIdx.x + 256];
    a_im = d_nnt2[threadIdx.x + 256 + 512];
    b2[threadIdx.x + 256] = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b2[threadIdx.x + 256 + 512] = fpr_zero;

    a_re = d_nnt3[threadIdx.x];
    a_im = d_nnt3[threadIdx.x + 512];
    tmp1 = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b2[threadIdx.x] = fpr_add(b2[threadIdx.x], tmp1);

    a_re = d_nnt3[threadIdx.x + 256];
    a_im = d_nnt3[threadIdx.x + 256 + 512];
    tmp1 = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    b2[threadIdx.x + 256] = fpr_add(b2[threadIdx.x + 256], tmp1);

    fpr reg0,reg1,reg2,reg3;
    __shared__ fpr s_f[Falcon_N+Falcon_N/2];


    size_t idx1 = threadIdx.x;
    size_t idx2 = threadIdx.x + threadIdx.x;

    reg0 = fpr_of(hm[idx1]);
    reg2 = fpr_of(hm[idx1 + 256]);
    reg1 = fpr_of(hm[idx1 + 512]);
    reg3 = fpr_of(hm[idx1 + 512 + 256]);

    fft1024(reg0, reg1, reg2, reg3, s_f);

    FPC_ADD(b5[idx2], b5[idx2 + 512],
            reg0, reg1, reg2, reg3);
    FPC_SUB(b5[idx2 + 1], b5[idx2 + 1 + 512],
            reg0, reg1, reg2, reg3);
    __syncthreads();

    for(int i=threadIdx.x;i<512;i+=blockDim.x){
        reg0 = b5[i];
        reg1 = b5[i + 512];
        reg2 = b4[i];
        reg3 = b4[i + 512];
        FPC_MUL(reg0, reg1, reg0, reg1, reg2, reg3);
        b4[i] = fpr_mul(reg0, fpr_n(fpr_inverse_of_q));
        b4[i + 512] = fpr_mul(reg1, fpr_n(fpr_inverse_of_q));
    }

    for(int i=threadIdx.x;i<512;i+=blockDim.x){
        reg0 = b5[i];
        reg1 = b5[i + 512];
        reg2 = d_nnt3[i];
        reg3 = d_nnt3[i + 512];
        FPC_MUL(reg0, reg1, reg0, reg1, reg2, reg3);
        b3[i] = fpr_mul(reg0, fpr_inverse_of_q);
        b3[i + 512] = fpr_mul(reg1, fpr_inverse_of_q);
    }
    __syncthreads();

}

__global__ void arrange_kernel(fpr *d_f0, fpr *d_f1, fpr *d_f2, fpr *d_f3, fpr *d_f4, fpr *d_f5,
                               fpr *d_ntt, size_t d_mem_pool_pitch)
{
    fpr *f0 = d_f0 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *f1 = d_f1 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *f2 = d_f2 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *f3 = d_f3 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *f4 = d_f4 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *f5 = d_f5 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;

    const fpr *g_ntt = d_ntt;
    const fpr *f_ntt = g_ntt + Falcon_N ;
    const fpr *G_ntt = f_ntt + Falcon_N ;
    const fpr *F_ntt = G_ntt + Falcon_N ;

    f5[threadIdx.x] = f4[threadIdx.x];
    f5[256 + threadIdx.x] = f4[256 + threadIdx.x];
    f5[512 + threadIdx.x] = f4[512 + threadIdx.x];
    f5[768 + threadIdx.x] = f4[768 + threadIdx.x];

    f4[threadIdx.x] = f3[threadIdx.x];
    f4[256 + threadIdx.x] = f3[256 + threadIdx.x];
    f4[512 + threadIdx.x] = f3[512 + threadIdx.x];
    f4[768 + threadIdx.x] = f3[768 + threadIdx.x];

    f0[threadIdx.x] = g_ntt[threadIdx.x];
    f0[threadIdx.x + 256] = g_ntt[threadIdx.x + 256];
    f0[threadIdx.x + 512] = g_ntt[threadIdx.x + 512];
    f0[threadIdx.x + 768] = g_ntt[threadIdx.x + 768];

    f1[threadIdx.x] = f_ntt[threadIdx.x];
    f1[threadIdx.x + 256] = f_ntt[threadIdx.x + 256];
    f1[threadIdx.x + 512] = f_ntt[threadIdx.x + 512];
    f1[threadIdx.x + 768] = f_ntt[threadIdx.x + 768];

    f2[threadIdx.x] = G_ntt[threadIdx.x];
    f2[threadIdx.x + 256] = G_ntt[threadIdx.x + 256];
    f2[threadIdx.x + 512] = G_ntt[threadIdx.x + 512];
    f2[threadIdx.x + 768] = G_ntt[threadIdx.x + 768];

    f3[threadIdx.x] = F_ntt[threadIdx.x];
    f3[threadIdx.x + 256] = F_ntt[threadIdx.x + 256];
    f3[threadIdx.x + 512] = F_ntt[threadIdx.x + 512];
    f3[threadIdx.x + 768] = F_ntt[threadIdx.x + 768];
}


