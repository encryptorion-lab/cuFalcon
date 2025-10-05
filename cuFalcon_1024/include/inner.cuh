#pragma once

/*
 * Reduce a small signed integer modulo q. The source integer MUST
 * be between -q/2 and +q/2.
 */
__device__ static inline uint32_t mq_conv_small(int x)
{
    /*
     * If x < 0, the cast to uint32_t will set the high bit to 1.
     */
    uint32_t y;

    y = (uint32_t)x;
    y += Q & -(y >> 31);
    return y;
}


/*
 * Montgomery squaring (computes (x^2)/R).
 */
__device__ static inline uint32_t mq_montysqr(uint32_t x)
{
    return mq_montymul(x, x);
}

/*
 * Divide x by y modulo q = 12289.
 */
__device__ static inline uint32_t mq_div_12289(uint32_t x, uint32_t y)
{
    /*
     * We invert y by computing y^(q-2) mod q.
     *
     * We use the following addition chain for exponent e = 12287:
     *
     *   e0 = 1
     *   e1 = 2 * e0 = 2
     *   e2 = e1 + e0 = 3
     *   e3 = e2 + e1 = 5
     *   e4 = 2 * e3 = 10
     *   e5 = 2 * e4 = 20
     *   e6 = 2 * e5 = 40
     *   e7 = 2 * e6 = 80
     *   e8 = 2 * e7 = 160
     *   e9 = e8 + e2 = 163
     *   e10 = e9 + e8 = 323
     *   e11 = 2 * e10 = 646
     *   e12 = 2 * e11 = 1292
     *   e13 = e12 + e9 = 1455
     *   e14 = 2 * e13 = 2910
     *   e15 = 2 * e14 = 5820
     *   e16 = e15 + e10 = 6143
     *   e17 = 2 * e16 = 12286
     *   e18 = e17 + e0 = 12287
     *
     * Additions on exponents are converted to Montgomery
     * multiplications. We define all intermediate results as so
     * many local variables, and let the C compiler work out which
     * must be kept around.
     */
    uint32_t y0, y1, y2, y3, y4, y5, y6, y7, y8, y9;
    uint32_t y10, y11, y12, y13, y14, y15, y16, y17, y18;

    y0 = mq_montymul(y, R2);
    y1 = mq_montysqr(y0);
    y2 = mq_montymul(y1, y0);
    y3 = mq_montymul(y2, y1);
    y4 = mq_montysqr(y3);
    y5 = mq_montysqr(y4);
    y6 = mq_montysqr(y5);
    y7 = mq_montysqr(y6);
    y8 = mq_montysqr(y7);
    y9 = mq_montymul(y8, y2);
    y10 = mq_montymul(y9, y8);
    y11 = mq_montysqr(y10);
    y12 = mq_montysqr(y11);
    y13 = mq_montymul(y12, y9);
    y14 = mq_montysqr(y13);
    y15 = mq_montysqr(y14);
    y16 = mq_montymul(y15, y10);
    y17 = mq_montysqr(y16);
    y18 = mq_montymul(y17, y0);

    /*
     * Final multiplication with x, which is not in Montgomery
     * representation, computes the correct division result.
     */
    return mq_montymul(y18, x);
}

