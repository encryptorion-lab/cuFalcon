/*
 * This file is part of cuFalcon.
 *
 * Copyright (c) 2025 Wenqian Li, et al.
 *
 * Licensed under the GNU General Public License v3.0 (GPLv3)
 * See the LICENSE file in the project root for license details.
 */

#include "../include/shake.cuh"
#include "../include/fpr.cuh"
#include "../include/rng.cuh"
#include "../include/ffSampling.cuh"
#include "../include/fft.cuh"

#define MKN(logn)   ((size_t)1 << (logn))

typedef int (*samplerZ)(void *ctx, fpr mu, fpr sigma);

__device__ static inline unsigned ffLDL_treesize(unsigned logn)
{
    /*
     * For logn = 0 (polynomials are constant), the "tree" is a
     * single element. Otherwise, the tree node has size 2^logn, and
     * has two child trees for size logn-1 each. Thus, treesize s()
     * must fulfill these two relations:
     *
     *   s(0) = 1
     *   s(logn) = (2^logn) + 2*s(logn-1)
     */
    return (logn + 1) << logn;
}

__device__ void poly_LDLmv_fft(fpr *__restrict d11, fpr *__restrict l10,
                               const fpr *__restrict g00, const fpr *__restrict g01,
                               const fpr *__restrict g11, unsigned logn)
{
    size_t n, hn, u;

    n = (size_t)1 << logn;
    hn = n >> 1;
    for (u = threadIdx.x; u < hn; u += blockDim.x) {
        fpr g00_re, g00_im, g01_re, g01_im, g11_re, g11_im;
        fpr mu_re, mu_im;

        g00_re = g00[u];
        g00_im = g00[u + hn];
        g01_re = g01[u];
        g01_im = g01[u + hn];
        g11_re = g11[u];
        g11_im = g11[u + hn];
        FPC_DIV(mu_re, mu_im, g01_re, g01_im, g00_re, g00_im);
        FPC_MUL(g01_re, g01_im, mu_re, mu_im, g01_re, fpr_neg(g01_im));
        FPC_SUB(d11[u], d11[u + hn], g11_re, g11_im, g01_re, g01_im);
        l10[u] = mu_re;
        l10[u + hn] = fpr_neg(mu_im);
    }
}

__device__ static void ffLDL_fft_inner(fpr *__restrict tree,
                fpr *__restrict g0, fpr *__restrict g1, unsigned logn, fpr *__restrict tmp)
{
    size_t n, hn;

    n = MKN(logn);
    if (n == 1) {
        tree[0] = g0[0];
        return;
    }
    hn = n >> 1;

    /*
     * The LDL decomposition yields L (which is written in the tree)
     * and the diagonal of D. Since d00 = g0, we just write d11
     * into tmp.
     */
    poly_LDLmv_fft(tmp, tree, g0, g1, g0, logn);

    /*
     * Split d00 (currently in g0) and d11 (currently in tmp). We
     * reuse g0 and g1 as temporary storage spaces:
     *   d00 splits into g1, g1+hn
     *   d11 splits into g0, g0+hn
     */
    poly_split_fft_s_2(g1, g1 + hn, g0, logn);
    poly_split_fft_s_2(g0, g0 + hn, tmp, logn);

    /*
     * Each split result is the first row of a new auto-adjoint
     * quasicyclic matrix for the next recursive step.
     */
    ffLDL_fft_inner(tree + n,
                    g1, g1 + hn, logn - 1, tmp);
    ffLDL_fft_inner(tree + n + ffLDL_treesize(logn - 1),
                    g0, g0 + hn, logn - 1, tmp);
}

__device__  void
ffLDL_binary_normalize(fpr *tree, unsigned orig_logn, unsigned logn)
{
    /*
     * TODO: make an iterative version.
     */
    size_t n;

    n = MKN(logn);
    if (n == 1) {
        /*
         * We actually store in the tree leaf the inverse of
         * the value mandated by the specification: this
         * saves a division both here and in the sampler.
         */
        tree[0] = fpr_mul(fpr_sqrt(tree[0]), fpr_inv_sigma[orig_logn]);
    } else {
        ffLDL_binary_normalize(tree + n, orig_logn, logn - 1);
        ffLDL_binary_normalize(tree + n + ffLDL_treesize(logn - 1),
                               orig_logn, logn - 1);
    }
}

__global__ void sign_tree(fpr *d_t0, fpr *d_t1, fpr *d_b01, fpr *d_b11, const uint16_t *d_hm, int logn,size_t d_mem_pool_pitch)
{
    fpr reg0,reg1,reg2,reg3;
    fpr *t0 = d_t0 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *t1 = d_t1 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *b01 = d_b01 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    fpr *b11 = d_b11 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr) ;
    const uint16_t *hm = d_hm + blockIdx.x * d_mem_pool_pitch / sizeof(uint16_t) ;

    __shared__ fpr s_f[Falcon_N+Falcon_N/2];


    fpr ni = fpr_inverse_of_q;
    int n = MKN(logn);
    size_t idx1 = threadIdx.x;
    size_t idx2 = threadIdx.x + threadIdx.x;

    reg0 = fpr_of(hm[idx1]);
    reg2 = fpr_of(hm[idx1 + 128]);
    reg1 = fpr_of(hm[idx1 + 256]);
    reg3 = fpr_of(hm[idx1 + 256 + 128]);

    fft_512(reg0, reg1, reg2, reg3, s_f);

    FPC_ADD(t0[idx2], t0[idx2 + 256],
            reg0, reg1, reg2, reg3);
    FPC_SUB(t0[idx2 + 1], t0[idx2 + 1 + 256],
            reg0, reg1, reg2, reg3);
    __syncthreads();

    for(int i = threadIdx.x; i < n; i += blockDim.x){
        t1[i].v = t0[i].v;
    }
    __syncthreads();

    reg0 = t1[threadIdx.x];
    reg1 = t1[threadIdx.x + 256];
    reg2 = b01[threadIdx.x];
    reg3 = b01[threadIdx.x + 256];
    FPC_MUL(t1[threadIdx.x], t1[threadIdx.x + 256], reg0,reg1,reg2,reg3);
    __syncthreads();
    reg0 = t1[threadIdx.x + 128];
    reg1 = t1[threadIdx.x + 128 + 256];
    reg2 = b01[threadIdx.x + 128];
    reg3 = b01[threadIdx.x + 128 + 256];
    FPC_MUL(t1[threadIdx.x + 128], t1[threadIdx.x + 128 + 256], reg0,reg1,reg2,reg3);

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        t1[i] = fpr_mul(t1[i], fpr_neg(ni));
    }

    reg0 = t0[threadIdx.x];
    reg1 = t0[threadIdx.x + 256];
    reg2 = b11[threadIdx.x];
    reg3 = b11[threadIdx.x + 256];
    FPC_MUL(t0[threadIdx.x], t0[threadIdx.x + 256], reg0,reg1,reg2,reg3);

    reg0 = t0[threadIdx.x + 128];
    reg1 = t0[threadIdx.x + 128 + 256];
    reg2 = b11[threadIdx.x + 128];
    reg3 = b11[threadIdx.x + 128 + 256];
    FPC_MUL(t0[threadIdx.x + 128], t0[threadIdx.x + 128 + 256], reg0,reg1,reg2,reg3);

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        t0[i] = fpr_mul(t0[i], ni);
    }

}

__global__ void ffSampling_fft_tree(fpr *d_z0, fpr *d_z1, fpr *d_t0, fpr *d_t1, fpr *d_tree, fpr *d_tmp, unsigned orig_logn, unsigned logn, uint64_t *d_scA, uint64_t *d_scdptr, size_t d_mem_pool_pitch)
{
    size_t n, hn, i;
    STACK2 stack[LOGN + 1];	//orig_logn + 1
    unsigned stack_top = 0;

    stack[0].t0 = d_t0 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    stack[0].t1 = d_t1 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    stack[0].z0 = d_z0 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    stack[0].z1 = d_z1 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    stack[0].tmp = d_tmp + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    stack[0].tree = d_tree + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    stack[0].logn = logn;
    stack[0].is_z0 = 0;
    stack[0].is_z1 = 0;
    uint64_t *scdptr = d_scdptr + blockIdx.x * d_mem_pool_pitch / sizeof(uint64_t);
    uint64_t *scA = d_scA + blockIdx.x * d_mem_pool_pitch / sizeof(uint64_t);

    fpr *f;
    fpr *f0;

    fpr *tree0;
    fpr *tree1;

    __shared__ inner_shake256_context_s rng;
    __shared__ sampler_context_s samp_ctx;
    samp_ctx.sigma_min = fpr_sigma_min[logn];
    samp_ctx.p.ptr = 0;
    samp_ctx.p.type = 0;
    rng.dptr = *scdptr;
    for(i=threadIdx.x; i<25; i+=blockDim.x) {
        rng.st.A[i] = scA[i];
    }

    prng_init_s(&samp_ctx.p, &rng);

    while (1)
    {
        /*
         * Deepest level: the LDL tree leaf value is just g00 (the
         * array has length only 1 at this point); we normalize it
         * with regards to sigma, then use it for sampling.
         */

        if (stack[stack_top].logn == 2) {

            fpr x0, x1, y0, y1, w0, w1, w2, w3, sigma;
            fpr a_re, a_im, b_re, b_im, c_re, c_im;

            tree0 = stack[stack_top].tree + 4;
            tree1 = stack[stack_top].tree + 8;

            /*
             * We split t1 into w*, then do the recursive invocation,
             * with output in w*. We finally merge back into z1.
             */
            a_re = stack[stack_top].t1[0];
            a_im = stack[stack_top].t1[2];
            b_re = stack[stack_top].t1[1];
            b_im = stack[stack_top].t1[3];
            c_re = fpr_add(a_re, b_re);
            c_im = fpr_add(a_im, b_im);
            w0 = fpr_half(c_re);
            w1 = fpr_half(c_im);
            c_re = fpr_sub(a_re, b_re);
            c_im = fpr_sub(a_im, b_im);
            w2 = fpr_mul(fpr_add(c_re, c_im), fpr_invsqrt8);
            w3 = fpr_mul(fpr_sub(c_im, c_re), fpr_invsqrt8);

            x0 = w2;
            x1 = w3;
            sigma = tree1[3];
            w2 = fpr_of(sampler(&samp_ctx, x0, sigma));
            w3 = fpr_of(sampler(&samp_ctx, x1, sigma));
            a_re = fpr_sub(x0, w2);
            a_im = fpr_sub(x1, w3);
            b_re = tree1[0];
            b_im = tree1[1];
            c_re = fpr_sub(fpr_mul(a_re, b_re), fpr_mul(a_im, b_im));
            c_im = fpr_add(fpr_mul(a_re, b_im), fpr_mul(a_im, b_re));
            x0 = fpr_add(c_re, w0);
            x1 = fpr_add(c_im, w1);
            sigma = tree1[2];
            w0 = fpr_of(sampler(&samp_ctx, x0, sigma));
            w1 = fpr_of(sampler(&samp_ctx, x1, sigma));

            a_re = w0;
            a_im = w1;
            b_re = w2;
            b_im = w3;
            c_re = fpr_mul(fpr_sub(b_re, b_im), fpr_invsqrt2);
            c_im = fpr_mul(fpr_add(b_re, b_im), fpr_invsqrt2);
            stack[stack_top].z1[0] = w0 = fpr_add(a_re, c_re);
            stack[stack_top].z1[2] = w2 = fpr_add(a_im, c_im);
            stack[stack_top].z1[1] = w1 = fpr_sub(a_re, c_re);
            stack[stack_top].z1[3] = w3 = fpr_sub(a_im, c_im);

            /*
             * Compute tb0 = t0 + (t1 - z1) * L. Value tb0 ends up in w*.
             */
            w0 = fpr_sub(stack[stack_top].t1[0], w0);
            w1 = fpr_sub(stack[stack_top].t1[1], w1);
            w2 = fpr_sub(stack[stack_top].t1[2], w2);
            w3 = fpr_sub(stack[stack_top].t1[3], w3);

            a_re = w0;
            a_im = w2;
            b_re = stack[stack_top].tree[0];
            b_im = stack[stack_top].tree[2];
            w0 = fpr_sub(fpr_mul(a_re, b_re), fpr_mul(a_im, b_im));
            w2 = fpr_add(fpr_mul(a_re, b_im), fpr_mul(a_im, b_re));
            a_re = w1;
            a_im = w3;
            b_re = stack[stack_top].tree[1];
            b_im = stack[stack_top].tree[3];
            w1 = fpr_sub(fpr_mul(a_re, b_re), fpr_mul(a_im, b_im));
            w3 = fpr_add(fpr_mul(a_re, b_im), fpr_mul(a_im, b_re));

            w0 = fpr_add(w0, stack[stack_top].t0[0]);
            w1 = fpr_add(w1, stack[stack_top].t0[1]);
            w2 = fpr_add(w2, stack[stack_top].t0[2]);
            w3 = fpr_add(w3, stack[stack_top].t0[3]);

            /*
             * Second recursive invocation.
             */
            a_re = w0;
            a_im = w2;
            b_re = w1;
            b_im = w3;
            c_re = fpr_add(a_re, b_re);
            c_im = fpr_add(a_im, b_im);
            w0 = fpr_half(c_re);
            w1 = fpr_half(c_im);
            c_re = fpr_sub(a_re, b_re);
            c_im = fpr_sub(a_im, b_im);
            w2 = fpr_mul(fpr_add(c_re, c_im), fpr_invsqrt8);
            w3 = fpr_mul(fpr_sub(c_im, c_re), fpr_invsqrt8);

            x0 = w2;
            x1 = w3;
            sigma = tree0[3];
            w2 = y0 = fpr_of(sampler(&samp_ctx, x0, sigma));
            w3 = y1 = fpr_of(sampler(&samp_ctx, x1, sigma));
            a_re = fpr_sub(x0, y0);
            a_im = fpr_sub(x1, y1);
            b_re = tree0[0];
            b_im = tree0[1];
            c_re = fpr_sub(fpr_mul(a_re, b_re), fpr_mul(a_im, b_im));
            c_im = fpr_add(fpr_mul(a_re, b_im), fpr_mul(a_im, b_re));
            x0 = fpr_add(c_re, w0);
            x1 = fpr_add(c_im, w1);
            sigma = tree0[2];
            w0 = fpr_of(sampler(&samp_ctx, x0, sigma));
            w1 = fpr_of(sampler(&samp_ctx, x1, sigma));

            a_re = w0;
            a_im = w1;
            b_re = w2;
            b_im = w3;
            c_re = fpr_mul(fpr_sub(b_re, b_im), fpr_invsqrt2);
            c_im = fpr_mul(fpr_add(b_re, b_im), fpr_invsqrt2);
            stack[stack_top].z0[0] = fpr_add(a_re, c_re);
            stack[stack_top].z0[2] = fpr_add(a_im, c_im);
            stack[stack_top].z0[1] = fpr_sub(a_re, c_re);
            stack[stack_top].z0[3] = fpr_sub(a_im, c_im);
            if (!stack[--stack_top].is_z0)
            {
                poly_merge_fft_s_2(stack[stack_top].z1, stack[stack_top].tmp, stack[stack_top].tmp + hn, stack[stack_top].logn);
            }
            else
            {
                poly_merge_fft_s_2(stack[stack_top].z0, stack[stack_top].tmp, stack[stack_top].tmp + hn, stack[stack_top].logn);
            }
        }
        else
        {
            n = (size_t)1 << stack[stack_top].logn;
            hn = n >> 1;
            tree0 = stack[stack_top].tree + n;
            tree1 = stack[stack_top].tree + n + ffLDL_treesize(stack[stack_top].logn - 1);
            if (!stack[stack_top].is_z1) // 0
            {
                stack[stack_top].is_z1 = 1;
                poly_split_fft_s_2(stack[stack_top].z1, stack[stack_top].z1 + hn, stack[stack_top].t1, stack[stack_top].logn);
                stack[stack_top + 1].z0 = stack[stack_top].tmp;
                stack[stack_top + 1].z1 = stack[stack_top].tmp + hn;
                stack[stack_top + 1].tree = tree1;
                stack[stack_top + 1].t0 = stack[stack_top].z1;
                stack[stack_top + 1].t1 = stack[stack_top].z1 + hn;
                stack[stack_top + 1].tmp = stack[stack_top].tmp + n;
                stack[stack_top + 1].logn = stack[stack_top].logn - 1;
                stack[stack_top + 1].is_z0 = 0;
                stack[++stack_top].is_z1 = 0;

            }
            else if (!stack[stack_top].is_z0) // 0
            {
                f = stack[stack_top].tmp;
                f0 = stack[stack_top].t1;
                for(int i=threadIdx.x;i<n;i+=blockDim.x){
                    f[i] = f0[i];
                }
                poly_sub_s_2(stack[stack_top].tmp, stack[stack_top].z1, stack[stack_top].logn);
                poly_mul_fft_s_2(stack[stack_top].tmp, stack[stack_top].tree, stack[stack_top].logn);
                poly_add_s_2(stack[stack_top].tmp, stack[stack_top].t0, stack[stack_top].logn);

                stack[stack_top].is_z0 = 1;
                poly_split_fft_s_2(stack[stack_top].z0, stack[stack_top].z0 + hn, stack[stack_top].tmp, stack[stack_top].logn);
                stack[stack_top + 1].z0 = stack[stack_top].tmp;
                stack[stack_top + 1].z1 = stack[stack_top].tmp + hn;
                stack[stack_top + 1].tree = tree0;
                stack[stack_top + 1].t0 = stack[stack_top].z0;
                stack[stack_top + 1].t1 = stack[stack_top].z0 + hn;
                stack[stack_top + 1].tmp = stack[stack_top].tmp + n;
                stack[stack_top + 1].logn = stack[stack_top].logn - 1;
                stack[stack_top + 1].is_z1 = 0;
                stack[++stack_top].is_z0 = 0;

            }
            else
            {
                if (stack[stack_top].logn == orig_logn)
                {
                    return;
                }
                else
                {
                    if (!stack[--stack_top].is_z0)
                    {
                        poly_merge_fft_s_2(stack[stack_top].z1, stack[stack_top].tmp, stack[stack_top].tmp + n, stack[stack_top].logn);
                    }
                    else
                    {
                        poly_merge_fft_s_2(stack[stack_top].z0, stack[stack_top].tmp, stack[stack_top].tmp + n, stack[stack_top].logn);
                    }
                }
            }
        }
    }

}

__global__ void Get_lattice_point(fpr *d_t0, fpr *d_t1, fpr *d_tx, fpr *d_ty, fpr *d_b00, fpr *d_b01, fpr *d_b10, fpr *d_b11, unsigned logn, size_t d_mem_pool_pitch) {

    fpr *t0 = d_t0 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *t1 = d_t1 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *tx = d_tx + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *ty = d_ty + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b01 = d_b01 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b11 = d_b11 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b00 = d_b00 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);
    fpr *b10 = d_b10 + blockIdx.x * d_mem_pool_pitch / sizeof(fpr);

    fpr a_re, a_im, b_re, b_im, tmp1, tmp2;
    int n = MKN(logn);

    for (int i = threadIdx.x; i < n; i += blockDim.x) t0[i] = tx[i];
    for (int i = threadIdx.x; i < n; i += blockDim.x) t1[i] = ty[i];

    a_re = b00[threadIdx.x];
    a_im = b00[threadIdx.x + (Falcon_N>>1)];
    b_re = tx[threadIdx.x];
    b_im = tx[threadIdx.x + (Falcon_N>>1)];
    FPC_MUL(tmp1, tmp2, a_re, a_im, b_re, b_im);

    a_re = b10[threadIdx.x];
    a_im = b10[threadIdx.x + (Falcon_N>>1)];
    b_re = ty[threadIdx.x];
    b_im = ty[threadIdx.x + (Falcon_N>>1)];
    FPC_MUL(ty[threadIdx.x], ty[threadIdx.x + (Falcon_N>>1)], a_re, a_im, b_re, b_im);

    tx[threadIdx.x] = fpr_add(ty[threadIdx.x], tmp1);
    tx[threadIdx.x + (Falcon_N>>1)] = fpr_add(ty[threadIdx.x + (Falcon_N>>1)], tmp2);

    for (int i = threadIdx.x; i < n; i += blockDim.x) ty[i] = t0[i];

    a_re = b01[threadIdx.x];
    a_im = b01[threadIdx.x + (Falcon_N>>1)];
    b_re = ty[threadIdx.x];
    b_im = ty[threadIdx.x + (Falcon_N>>1)];
    FPC_MUL(ty[threadIdx.x],ty[threadIdx.x + (Falcon_N>>1)], a_re, a_im, b_re, b_im);

    for (int i = threadIdx.x; i < n; i += blockDim.x) t0[i] = tx[i];

    a_re = b11[threadIdx.x];
    a_im = b11[threadIdx.x + (Falcon_N>>1)];
    b_re = t1[threadIdx.x];
    b_im = t1[threadIdx.x + (Falcon_N>>1)];
    FPC_MUL(tmp1, tmp2, a_re, a_im, b_re, b_im);

    t1[threadIdx.x] = fpr_add(ty[threadIdx.x], tmp1);
    t1[threadIdx.x + (Falcon_N>>1)] = fpr_add(ty[threadIdx.x + (Falcon_N>>1)], tmp2);
}