/*
 * This file is part of cuFalcon.
 *
 * Copyright (c) 2025 Wenqian Li, et al.
 *
 * Licensed under the GNU General Public License v3.0 (GPLv3)
 * See the LICENSE file in the project root for license details.
 */

#include "../include/common.cuh"
#include "../include/fft.cuh"
#include <cstdio>


__global__ void check1(int16_t* d_s1tmp, fpr *d_t0, uint16_t *d_hm, uint32_t *d_sqn, size_t d_mem_pool_pitch)
{
    uint32_t ng, u;
    int16_t* s1tmp= d_s1tmp + blockIdx.x * d_mem_pool_pitch / sizeof(int16_t);
    fpr *t0 = d_t0 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    uint16_t *hm= d_hm + blockIdx.x * d_mem_pool_pitch / sizeof(uint16_t);
    uint32_t *sqn = d_sqn + blockIdx.x * d_mem_pool_pitch / sizeof(uint32_t);

    *sqn = 0;
    ng = 0;
    for (u = 0; u < Falcon_N; u ++) {
        int32_t z;

        z = (int32_t)hm[u] - (int32_t)fpr_rint(t0[u]);
        *sqn += (uint32_t)(z * z);
        ng |= *sqn;
        s1tmp[u] = (int16_t)z;
    }
    *sqn |= -(ng >> 31);
}

__global__ void check2(int16_t* d_s2tmp, fpr *d_t1, size_t d_mem_pool_pitch)
{
    int16_t *s2tmp= d_s2tmp + blockIdx.x * d_mem_pool_pitch / sizeof(int16_t);
    fpr *t1 = d_t1 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);

    s2tmp[threadIdx.x] = (int16_t) (- fpr_rint(t1[threadIdx.x]));
}

__global__ void is_short_half_gpu(uint32_t *d_sqn, const int16_t *d_s2, uint32_t *d_s, size_t d_mem_pool_pitch)
{
    size_t n, u;
    uint32_t ng;
    uint32_t *sqn = d_sqn + blockIdx.x * d_mem_pool_pitch / sizeof(uint32_t);
    auto *s2 = d_s2 + blockIdx.x * d_mem_pool_pitch / sizeof(int16_t);
    uint32_t *s = d_s + blockIdx.x * d_mem_pool_pitch / sizeof(uint32_t);

    n = (size_t)1 << 10;
    ng = -(*sqn >> 31);
    for (u = 0; u < n; u ++) {
        int32_t z;

        z = s2[u];
        *sqn+= (uint32_t)(z * z);
        ng |= *sqn;
    }
    *sqn|= -(ng >> 31);

    if(*sqn <= l2bound[10])  //logn=10
    {
        *s=1;
    }
    if(!*s){};
}

__global__ void comp_encode_gpu(uint8_t *d_buf, size_t max_out_len,
                                const int16_t *d_x, unsigned logn, uint32_t *d_len, size_t d_mem_pool_pitch)
{
    size_t n, u, v;
    uint32_t acc;
    unsigned acc_len;

    uint8_t *buf = d_buf + blockIdx.x * d_mem_pool_pitch;
    auto *x = d_x + blockIdx.x * d_mem_pool_pitch / sizeof(int16_t);  // N
    uint32_t *len = d_len + blockIdx.x * d_mem_pool_pitch / sizeof(uint32_t);

    n = (size_t)1 << logn;

    /*
     * Make sure that all values are within the -2047..+2047 range.
     */
    for (u = threadIdx.x; u < n; u +=blockDim.x) {
        if (x[u] < -2047 || x[u] > +2047) {
            return ;
        }
    }
    acc = 0;
    acc_len = 0;
    v = 0;
    for (u = 0; u < n; u ++){
        int t;
        unsigned w;

        /*
         * Get sign and absolute value of next integer; push the
         * sign bit.
         */
        acc <<= 1;
        t = x[u];
        if (t < 0) {
            t = -t;
            acc |= 1;
        }
        w = (unsigned)t;

        /*
         * Push the low 7 bits of the absolute value.
         */
        acc <<= 7;
        acc |= w & 127u;
        w >>= 7;

        /*
         * We pushed exactly 8 bits.
         */
        acc_len += 8;

        /*
         * Push as many zeros as necessary, then a one. Since the
         * absolute value is at most 2047, w can only range up to
         * 15 at this point, thus we will add at most 16 bits
         * here. With the 8 bits above and possibly up to 7 bits
         * from previous iterations, we may go up to 31 bits, which
         * will fit in the accumulator, which is an uint32_t.
         */
        acc <<= (w + 1);
        acc |= 1;
        acc_len += w + 1;

        /*
         * Produce all full bytes.
         */
        while (acc_len >= 8) {
            acc_len -= 8;
            if (buf != NULL) {
                if (v >= max_out_len) {
                    return;
                }
                buf[v] = (uint8_t)(acc >> acc_len);
            }
            v ++;
        }
    }

    /*
     * Flush remaining bits (if any).
     */
    if (acc_len > 0) {
        if (buf != NULL) {
            if (v >= max_out_len) {
                return ;
            }
            buf[v] = (uint8_t)(acc << (8 - acc_len));
        }
        v ++;
    }

    *len = v+1;
}

__global__ void write_smlen_gpu(uint8_t *d_sm, uint32_t *d_sig_len, size_t d_mem_pool_pitch)
{
    uint8_t *sm = d_sm + blockIdx.x * d_mem_pool_pitch;
    uint32_t *sig_len = d_sig_len + blockIdx.x * d_mem_pool_pitch / sizeof(uint32_t);
    sm[0] = (unsigned char)(*sig_len>>8);
    sm[1] = (unsigned char)*sig_len;
}

__global__ void byte_copy(uint8_t *d_out, uint8_t *d_in, uint32_t outlen, uint32_t inlen, size_t d_mem_pool_pitch)
{
    uint8_t *out = d_out + blockIdx.x * d_mem_pool_pitch;
    uint8_t *in = d_in + blockIdx.x * d_mem_pool_pitch;

    out[threadIdx.x] = in[threadIdx.x];

}

__global__ void byte_copy_2(uint8_t *d_out, uint8_t *d_in, uint32_t outlen, uint32_t *d_inlen, size_t d_mem_pool_pitch)
{
    uint8_t *out = d_out + blockIdx.x * d_mem_pool_pitch;
    uint8_t *in = d_in + blockIdx.x * d_mem_pool_pitch;
    uint32_t *inlen = d_inlen + blockIdx.x * d_mem_pool_pitch / sizeof(uint32_t);


    for(int i=threadIdx.x; i<*inlen; i+=blockDim.x)
    	out[i] = in[i];
}


__global__ void modq_decode_gpu(uint16_t *d_x, unsigned logn, uint8_t *d_in, size_t max_in_len, size_t d_mem_pool_pitch)
{
    uint32_t n, in_len, u, i=0;
    uint32_t tid = threadIdx.x;
    uint32_t acc;
    int acc_len;

    uint16_t *x = d_x + blockIdx.x * d_mem_pool_pitch / sizeof(uint16_t);
    uint8_t *in = d_in + blockIdx.x * d_mem_pool_pitch;

    n = (size_t)1 << logn;
    in_len = ((n * 14) + 7) >> 3;
    if (in_len > max_in_len) {
        return ;
    }

    acc = 0;
    acc_len = 0;
    u = 0;
    while (u < 4) {
        // acc = (acc << 8) | (*in ++);
        acc = (acc << 8) | (in[tid*7 +i]);
        acc_len += 8;
        if (acc_len >= 14) {
            unsigned w;

            acc_len -= 14;
            w = (acc >> acc_len) & 0x3FFF;
            if (w >= 12289) {
                return ;
            }
            x[tid*4 + u] = (uint16_t)w;
            u ++;
        }
        i++;
    }
    if ((acc & (((uint32_t)1 << acc_len) - 1)) != 0) {
        return ;
    }

}

__global__ void mq_NTT_tomonty(uint16_t *d_a, unsigned logn, size_t d_mem_pool_pitch)
{
    uint32_t tid = threadIdx.x;
    __shared__ uint16_t s_a[Falcon_N];
    uint16_t reg0,reg1;
    uint16_t *a = d_a + blockIdx.x * d_mem_pool_pitch / sizeof(uint16_t);

    reg0 = a[tid];
    reg1 = a[tid+512];
    mq_ntt1024(reg0,reg1,s_a);
    __syncthreads();
    a[tid*2] = (uint16_t)mq_montymul(reg0, R2);
    a[tid*2+1] = (uint16_t)mq_montymul(reg1, R2);

}

__global__ void msg_len_gpu(uint8_t *d_sm, uint32_t *d_msg_len, uint32_t *d_smlen, uint32_t *d_sig_len, size_t d_mem_pool_pitch)
{
    uint8_t *sm = d_sm + blockIdx.x * d_mem_pool_pitch;
    uint32_t *msg_len = d_msg_len + blockIdx.x * d_mem_pool_pitch / sizeof(uint32_t);
    uint32_t *smlen = d_smlen + blockIdx.x * d_mem_pool_pitch / sizeof(uint32_t);
    uint32_t *sig_len = d_sig_len + blockIdx.x * d_mem_pool_pitch / sizeof(uint32_t);
    *sig_len = ((uint64_t)sm[0] << 8) | (uint64_t)sm[1];
    *msg_len = *smlen - 2 - NONCELEN - *sig_len;

}

__global__ void comp_decode_gpu(int16_t *d_x, unsigned logn, uint8_t *d_in, uint32_t *d_in_len, uint32_t *d_msg_len, size_t d_mem_pool_pitch)
{
    uint32_t u, v;
    uint8_t *buf;
    uint32_t acc;
    unsigned acc_len;

    int16_t *x = d_x + blockIdx.x * d_mem_pool_pitch / sizeof(uint16_t);
    uint8_t *in = d_in + blockIdx.x * d_mem_pool_pitch;
    uint32_t *in_len = d_in_len + blockIdx.x * d_mem_pool_pitch / sizeof(uint32_t);
    uint32_t *msg_len = d_msg_len + blockIdx.x * d_mem_pool_pitch / sizeof(uint32_t);

    uint32_t max_in_len = *in_len - 1;
    buf = in + 3 + NONCELEN + *msg_len;

    acc = 0;
    acc_len = 0;
    v = 0;
    for (u = 0; u < Falcon_N; u ++) {
        unsigned b, s, m;

        /*
         * Get next eight bits: sign and low seven bits of the
         * absolute value.
         */
        if (v >= max_in_len) {
            return;
        }
        acc = (acc << 8) | (uint32_t)buf[v ++];
        // printf("%u \n", v);
        b = acc >> acc_len;
        s = b & 128;
        m = b & 127;

        /*
         * Get next bits until a 1 is reached.
         */
        for (;;) {
            if (acc_len == 0) {
                if (v >= max_in_len) {
                    return;
                }
                acc = (acc << 8) | (uint32_t)buf[v ++];
                acc_len = 8;
            }
            acc_len --;
            if (((acc >> acc_len) & 1) != 0) {
                break;
            }
            m += 128;
            if (m > 2047) {
                return;
            }
        }

        /*
         * "-0" is forbidden.
         */
        if (s && m == 0) {
            return;
        }

        x[u] = (int16_t)(s ? -(int)m : (int)m);
    }

    /*
     * Unused bits in the last byte must be zero.
     */
    if ((acc & ((1u << acc_len) - 1u)) != 0) {
        return;
    }
}

__global__ void comb_all_kernels(uint16_t *d_a, int16_t *d_s2, uint16_t *d_g, uint16_t *d_h, size_t d_mem_pool_pitch)
{
    uint32_t w;
    uint32_t tid = threadIdx.x;
    __shared__ uint16_t s_a[Falcon_N];

    uint16_t *a = d_a + blockIdx.x * d_mem_pool_pitch / sizeof(uint16_t);
    int16_t *s2 = d_s2 + blockIdx.x * d_mem_pool_pitch / sizeof(int16_t);
    uint16_t *g = d_g + blockIdx.x * d_mem_pool_pitch / sizeof(uint16_t);
    uint16_t *h = d_h + blockIdx.x * d_mem_pool_pitch / sizeof(uint16_t);

    // reduce_s2
    w = (uint32_t)s2[tid];
    w += Q & -(w >> 31);
    a[tid] = (uint16_t)w;
    w = (uint32_t)s2[tid + Falcon_N/2];
    w += Q & -(w >> 31);
    a[tid + Falcon_N/2] = (uint16_t)w;
    __syncthreads();

    uint16_t reg0,reg1;
    uint32_t ni = 64;
    // NTT
    reg0 = a[tid];
    reg1 = a[tid + (Falcon_N>>1)];
    mq_ntt1024(reg0,reg1,s_a);
    reg0 = (uint16_t)mq_montymul(reg0, g[tid*2]);
    reg1 = (uint16_t)mq_montymul(reg1, g[tid*2+1]);
    __syncthreads();
    mq_intt1024(reg0,reg1,s_a);
    reg0 = (uint16_t)mq_montymul(reg0, ni);
    reg1 = (uint16_t)mq_montymul(reg1, ni);

    //mq_poly_sub
    reg0 = (uint16_t)mq_sub(reg0, h[tid]);
    reg1 = (uint16_t)mq_sub(reg1, h[Falcon_N/2 + tid]);
    // norm_s2
    w = (int32_t)reg0;
    w -= (int32_t)(Q & -(((Q >> 1) - (uint32_t)w) >> 31));
    a[tid] = (int16_t)w;
    w = (int32_t)reg1;
    w -= (int32_t)(Q & -(((Q >> 1) - (uint32_t)w) >> 31));
    a[Falcon_N/2 + tid] = (int16_t)w;
    __syncthreads();
}


__global__ void is_short_gpu(int16_t *d_s1, int16_t *d_s2, uint32_t *d_s, size_t d_mem_pool_pitch)
{
    /*
     * We use the l2-norm. Code below uses only 32-bit operations to
     * compute the square of the norm with saturation to 2^32-1 if
     * the value exceeds 2^31-1.
     */

    int16_t *s1 = d_s1 + blockIdx.x * d_mem_pool_pitch / sizeof(uint16_t);
    int16_t *s2 = d_s2 + blockIdx.x * d_mem_pool_pitch / sizeof(int16_t);
    uint32_t *s = d_s + blockIdx.x * d_mem_pool_pitch / sizeof(uint32_t);


    size_t u;
    uint32_t ng, tmp;

    tmp = 0;
    ng = 0;
    for (u = 0; u < Falcon_N; u ++) {
        int32_t z;

        z = s1[u];
        tmp += (uint32_t)(z * z);
        ng |= tmp;
        z = s2[u];
        tmp += (uint32_t)(z * z);
        ng |= tmp;
    }
    tmp |= -(ng >> 31);

    if(tmp <= l2bound[10])
    {
        *s=1;
    }

    if(!*s)
        printf("short detected %u\n", *s);

}

__global__ void byte_cmp(uint8_t *d_m, uint8_t *d_m1, size_t d_mem_pool_pitch)
{
    uint32_t tid = threadIdx.x;
    uint8_t *m = d_m + blockIdx.x * d_mem_pool_pitch;
    uint8_t *m1 = d_m1 + blockIdx.x * d_mem_pool_pitch;

    if(m[tid] != m1[tid]){
        printf("wrong signature at %u %u: %u %u \n", blockIdx.x, tid, m[tid], m1[tid]);

    }
    __syncthreads();
}




