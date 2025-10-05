/*
 * This file is part of cuFalcon.
 *
 * Copyright (c) 2025 Wenqian Li, et al.
 *
 * Licensed under the GNU General Public License v3.0 (GPLv3)
 * See the LICENSE file in the project root for license details.
 */

#include "../include/sign.cuh"
#include "../include/fft.cuh"
#include "../include/inner.cuh"


__global__ void trim_i8_decode_kernel(int8_t *d_x, int8_t *d_y, int8_t *d_z, unsigned logn, unsigned bits, uint8_t *d_buf, size_t max_in_len, size_t d_mem_pool_pitch)
{
    size_t n, in_len, u;
    uint32_t acc, mask1, mask2, count = 0;
    unsigned acc_len;
    uint8_t *buf = d_buf + blockIdx.x * d_mem_pool_pitch;
    int8_t *x = d_x + blockIdx.x * d_mem_pool_pitch;
    int8_t *y = d_y + blockIdx.x * d_mem_pool_pitch;
    int8_t *z = d_z + blockIdx.x * d_mem_pool_pitch;

    n = (size_t)1 << logn;
    in_len = ((n * bits) + 7) >> 3;
    if (in_len > max_in_len) {
        return;
    }

    u = 0;
    acc = 0;
    acc_len = 0;
    mask1 = ((uint32_t)1 << bits) - 1;
    mask2 = (uint32_t)1 << (bits - 1);
    while (u < n) {

        acc = (acc << 8) | buf[count];
        acc_len += 8;
        while (acc_len >= bits && u < n) {
            uint32_t w;

            acc_len -= bits;
            w = (acc >> acc_len) & mask1;
            w |= -(w & mask2);
            if (w == -mask2) {
                /*
                 * The -2^(bits-1) value is forbidden.
                 */
                return ;
            }
            x[u] = (int8_t)*(int32_t *)&w;
            u += 1;
        }
        count = count+1;
    }



    if ((acc & (((uint32_t)1 << acc_len) - 1)) != 0) {
        /*
         * Extra bits in the last byte must be zero.
         */
        return ;
    }

    max_in_len-= in_len;
    buf+= in_len;
    in_len = ((n * bits) + 7) >> 3;
    if (in_len > max_in_len) {
        return;
    }

    u = 0;  count = 0;
    acc = 0;
    acc_len = 0;
    while (u < n) {
        acc = (acc << 8) | buf[count];
        acc_len += 8;
        while (acc_len >= bits && u < n) {
            uint32_t w;

            acc_len -= bits;
            w = (acc >> acc_len) & mask1;
            w |= -(w & mask2);
            if (w == -mask2) {
                /*
                 * The -2^(bits-1) value is forbidden.
                 */
                return ;
            }
            y[u] = (int8_t)*(int32_t *)&w;
            u += 1;

        }
        count = count+1;
    }
    if ((acc & (((uint32_t)1 << acc_len) - 1)) != 0) {
        /*
         * Extra bits in the last byte must be zero.
         */
        return ;
    }


    bits = 8;   //max_FG_bits
    max_in_len-= in_len;
    buf+= in_len;
    in_len = ((n * bits) + 7) >> 3;
    if (in_len > max_in_len) {
        return;
    }

    u = 0; count = 0;
    acc = 0;
    acc_len = 0;
    mask1 = ((uint32_t)1 << bits) - 1;
    mask2 = (uint32_t)1 << (bits - 1);
    while (u < n) {
        acc = (acc << 8) | buf[count];
        acc_len += 8;
        while (acc_len >= bits && u < n) {
            uint32_t w;

            acc_len -= bits;
            w = (acc >> acc_len) & mask1;
            w |= -(w & mask2);
            if (w == -mask2) {
                /*
                 * The -2^(bits-1) value is forbidden.
                 */
                return ;
            }
            z[u] = (int8_t)*(int32_t *)&w;
            u += 1;

        }
        count = count+1;
    }
    if ((acc & (((uint32_t)1 << acc_len) - 1)) != 0) {
        /*
         * Extra bits in the last byte must be zero.
         */
        return ;
    }
    __syncthreads();
}


__global__ void complete_private_to_fpr_kernel(fpr *r, int8_t *G, const int8_t *f, const int8_t *g, const int8_t *F, size_t d_mem_pool_pitch)
{

    uint32_t tid = threadIdx.x;
    __shared__ uint16_t s_a[Falcon_N+Falcon_N/2];
    uint16_t reg0,reg1,reg2,reg3;
    int8_t *d_G = G + blockIdx.x * d_mem_pool_pitch;
    const int8_t *d_f = f + blockIdx.x * d_mem_pool_pitch;
    const int8_t *d_g = g + blockIdx.x * d_mem_pool_pitch;
    const int8_t *d_F = F + blockIdx.x * d_mem_pool_pitch;
    fpr *d_r = r + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    __syncthreads();

    reg0 = (uint16_t)mq_conv_small(d_g[tid]);
    reg1 = (uint16_t)mq_conv_small(d_g[tid + Falcon_N/2]);
    reg2 = (uint16_t)mq_conv_small(d_F[tid]);
    reg3 = (uint16_t)mq_conv_small(d_F[tid + Falcon_N/2]);

    d_r[tid].v = (double)d_g[tid];
    d_r[tid + Falcon_N/2].v = (double)d_g[tid + Falcon_N/2];
    d_r[tid + 3 * Falcon_N].v = (double)d_F[tid];
    d_r[tid + 3 * Falcon_N + Falcon_N/2].v = (double)d_F[tid + Falcon_N/2];

    mq_ntt(reg0,reg1,s_a);
    __syncthreads();
    mq_ntt(reg2,reg3,s_a);

    reg0 = (uint16_t)mq_montymul(reg0, R2);
    reg1 = (uint16_t)mq_montymul(reg1, R2);
    reg0 = (uint16_t)mq_montymul(reg0, reg2);
    reg1 = (uint16_t)mq_montymul(reg1, reg3);

    reg2 = (uint16_t)mq_conv_small(d_f[tid]);
    reg3 = (uint16_t)mq_conv_small(d_f[Falcon_N/2 + tid]);
    __syncthreads();

    d_r[tid + Falcon_N].v = (double)d_f[tid];
    d_r[tid + Falcon_N + Falcon_N/2].v = (double)d_f[Falcon_N/2 + tid];
    __syncthreads();

    mq_ntt(reg2,reg3,s_a);

    if (reg2 == 0) {
        return;
    }

    reg0 = (uint16_t)mq_div_12289(reg0, reg2);

    if (reg3 == 0) {
        return;
    }
    reg1 = (uint16_t)mq_div_12289(reg1, reg3);

    mq_intt(reg0,reg1,s_a);

    uint32_t  w;
    int32_t gi;

    w = (uint16_t)mq_montymul(reg0, 128);//reg0;
    w -= (Q & ~-((w - (Q >> 1)) >> 31));
    gi = *(int32_t *)&w;
    if (gi < -127 || gi > +127) {
        return;
    }
    d_G[tid] = (int8_t)gi;
    d_r[tid + 2 * Falcon_N].v = (double)gi;

    w = (uint16_t)mq_montymul(reg1, 128);//reg1;
    w -= (Q & ~-((w - (Q >> 1)) >> 31));
    gi = *(int32_t *)&w;
    if (gi < -127 || gi > +127) {
        return;
    }
    d_G[Falcon_N/2 + tid] = (int8_t)gi;
    d_r[tid + 2 * Falcon_N + Falcon_N/2].v = (double)gi;

}

