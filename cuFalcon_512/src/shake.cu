/*
 * SHAKE implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2017-2019  Falcon Project
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   Thomas Pornin <thomas.pornin@nccgroup.com>
 */

#include <string.h>
#include "../include/shake.cuh"

/*
 * Process the provided state.
 */
__device__ void process_block(uint64_t *A)
{
	uint64_t t0, t1, t2, t3, t4;
	uint64_t tt0, tt1, tt2, tt3;
	uint64_t t, kt;
	uint64_t c0, c1, c2, c3, c4, bnn;
	int j;

	/*
	 * Invert some words (alternate internal representation, which
	 * saves some operations).
	 */
	A[ 1] = ~A[ 1];
	A[ 2] = ~A[ 2];
	A[ 8] = ~A[ 8];
	A[12] = ~A[12];
	A[17] = ~A[17];
	A[20] = ~A[20];

	/*
	 * Compute the 24 rounds. This loop is partially unrolled (each
	 * iteration computes two rounds).
	 */
	for (j = 0; j < 24; j += 2) {

		tt0 = A[ 1] ^ A[ 6];
		tt1 = A[11] ^ A[16];
		tt0 ^= A[21] ^ tt1;
		tt0 = (tt0 << 1) | (tt0 >> 63);
		tt2 = A[ 4] ^ A[ 9];
		tt3 = A[14] ^ A[19];
		tt0 ^= A[24];
		tt2 ^= tt3;
		t0 = tt0 ^ tt2;

		tt0 = A[ 2] ^ A[ 7];
		tt1 = A[12] ^ A[17];
		tt0 ^= A[22] ^ tt1;
		tt0 = (tt0 << 1) | (tt0 >> 63);
		tt2 = A[ 0] ^ A[ 5];
		tt3 = A[10] ^ A[15];
		tt0 ^= A[20];
		tt2 ^= tt3;
		t1 = tt0 ^ tt2;

		tt0 = A[ 3] ^ A[ 8];
		tt1 = A[13] ^ A[18];
		tt0 ^= A[23] ^ tt1;
		tt0 = (tt0 << 1) | (tt0 >> 63);
		tt2 = A[ 1] ^ A[ 6];
		tt3 = A[11] ^ A[16];
		tt0 ^= A[21];
		tt2 ^= tt3;
		t2 = tt0 ^ tt2;

		tt0 = A[ 4] ^ A[ 9];
		tt1 = A[14] ^ A[19];
		tt0 ^= A[24] ^ tt1;
		tt0 = (tt0 << 1) | (tt0 >> 63);
		tt2 = A[ 2] ^ A[ 7];
		tt3 = A[12] ^ A[17];
		tt0 ^= A[22];
		tt2 ^= tt3;
		t3 = tt0 ^ tt2;

		tt0 = A[ 0] ^ A[ 5];
		tt1 = A[10] ^ A[15];
		tt0 ^= A[20] ^ tt1;
		tt0 = (tt0 << 1) | (tt0 >> 63);
		tt2 = A[ 3] ^ A[ 8];
		tt3 = A[13] ^ A[18];
		tt0 ^= A[23];
		tt2 ^= tt3;
		t4 = tt0 ^ tt2;

		A[ 0] = A[ 0] ^ t0;
		A[ 5] = A[ 5] ^ t0;
		A[10] = A[10] ^ t0;
		A[15] = A[15] ^ t0;
		A[20] = A[20] ^ t0;
		A[ 1] = A[ 1] ^ t1;
		A[ 6] = A[ 6] ^ t1;
		A[11] = A[11] ^ t1;
		A[16] = A[16] ^ t1;
		A[21] = A[21] ^ t1;
		A[ 2] = A[ 2] ^ t2;
		A[ 7] = A[ 7] ^ t2;
		A[12] = A[12] ^ t2;
		A[17] = A[17] ^ t2;
		A[22] = A[22] ^ t2;
		A[ 3] = A[ 3] ^ t3;
		A[ 8] = A[ 8] ^ t3;
		A[13] = A[13] ^ t3;
		A[18] = A[18] ^ t3;
		A[23] = A[23] ^ t3;
		A[ 4] = A[ 4] ^ t4;
		A[ 9] = A[ 9] ^ t4;
		A[14] = A[14] ^ t4;
		A[19] = A[19] ^ t4;
		A[24] = A[24] ^ t4;
		A[ 5] = (A[ 5] << 36) | (A[ 5] >> (64 - 36));
		A[10] = (A[10] <<  3) | (A[10] >> (64 -  3));
		A[15] = (A[15] << 41) | (A[15] >> (64 - 41));
		A[20] = (A[20] << 18) | (A[20] >> (64 - 18));
		A[ 1] = (A[ 1] <<  1) | (A[ 1] >> (64 -  1));
		A[ 6] = (A[ 6] << 44) | (A[ 6] >> (64 - 44));
		A[11] = (A[11] << 10) | (A[11] >> (64 - 10));
		A[16] = (A[16] << 45) | (A[16] >> (64 - 45));
		A[21] = (A[21] <<  2) | (A[21] >> (64 - 2));
		A[ 2] = (A[ 2] << 62) | (A[ 2] >> (64 - 62));
		A[ 7] = (A[ 7] <<  6) | (A[ 7] >> (64 -  6));
		A[12] = (A[12] << 43) | (A[12] >> (64 - 43));
		A[17] = (A[17] << 15) | (A[17] >> (64 - 15));
		A[22] = (A[22] << 61) | (A[22] >> (64 - 61));
		A[ 3] = (A[ 3] << 28) | (A[ 3] >> (64 - 28));
		A[ 8] = (A[ 8] << 55) | (A[ 8] >> (64 - 55));
		A[13] = (A[13] << 25) | (A[13] >> (64 - 25));
		A[18] = (A[18] << 21) | (A[18] >> (64 - 21));
		A[23] = (A[23] << 56) | (A[23] >> (64 - 56));
		A[ 4] = (A[ 4] << 27) | (A[ 4] >> (64 - 27));
		A[ 9] = (A[ 9] << 20) | (A[ 9] >> (64 - 20));
		A[14] = (A[14] << 39) | (A[14] >> (64 - 39));
		A[19] = (A[19] <<  8) | (A[19] >> (64 -  8));
		A[24] = (A[24] << 14) | (A[24] >> (64 - 14));

		bnn = ~A[12];
		kt = A[ 6] | A[12];
		c0 = A[ 0] ^ kt;
		kt = bnn | A[18];
		c1 = A[ 6] ^ kt;
		kt = A[18] & A[24];
		c2 = A[12] ^ kt;
		kt = A[24] | A[ 0];
		c3 = A[18] ^ kt;
		kt = A[ 0] & A[ 6];
		c4 = A[24] ^ kt;
		A[ 0] = c0;
		A[ 6] = c1;
		A[12] = c2;
		A[18] = c3;
		A[24] = c4;
		bnn = ~A[22];
		kt = A[ 9] | A[10];
		c0 = A[ 3] ^ kt;
		kt = A[10] & A[16];
		c1 = A[ 9] ^ kt;
		kt = A[16] | bnn;
		c2 = A[10] ^ kt;
		kt = A[22] | A[ 3];
		c3 = A[16] ^ kt;
		kt = A[ 3] & A[ 9];
		c4 = A[22] ^ kt;
		A[ 3] = c0;
		A[ 9] = c1;
		A[10] = c2;
		A[16] = c3;
		A[22] = c4;
		bnn = ~A[19];
		kt = A[ 7] | A[13];
		c0 = A[ 1] ^ kt;
		kt = A[13] & A[19];
		c1 = A[ 7] ^ kt;
		kt = bnn & A[20];
		c2 = A[13] ^ kt;
		kt = A[20] | A[ 1];
		c3 = bnn ^ kt;
		kt = A[ 1] & A[ 7];
		c4 = A[20] ^ kt;
		A[ 1] = c0;
		A[ 7] = c1;
		A[13] = c2;
		A[19] = c3;
		A[20] = c4;
		bnn = ~A[17];
		kt = A[ 5] & A[11];
		c0 = A[ 4] ^ kt;
		kt = A[11] | A[17];
		c1 = A[ 5] ^ kt;
		kt = bnn | A[23];
		c2 = A[11] ^ kt;
		kt = A[23] & A[ 4];
		c3 = bnn ^ kt;
		kt = A[ 4] | A[ 5];
		c4 = A[23] ^ kt;
		A[ 4] = c0;
		A[ 5] = c1;
		A[11] = c2;
		A[17] = c3;
		A[23] = c4;
		bnn = ~A[ 8];
		kt = bnn & A[14];
		c0 = A[ 2] ^ kt;
		kt = A[14] | A[15];
		c1 = bnn ^ kt;
		kt = A[15] & A[21];
		c2 = A[14] ^ kt;
		kt = A[21] | A[ 2];
		c3 = A[15] ^ kt;
		kt = A[ 2] & A[ 8];
		c4 = A[21] ^ kt;
		A[ 2] = c0;
		A[ 8] = c1;
		A[14] = c2;
		A[15] = c3;
		A[21] = c4;
		A[ 0] = A[ 0] ^ RC[j + 0];

		tt0 = A[ 6] ^ A[ 9];
		tt1 = A[ 7] ^ A[ 5];
		tt0 ^= A[ 8] ^ tt1;
		tt0 = (tt0 << 1) | (tt0 >> 63);
		tt2 = A[24] ^ A[22];
		tt3 = A[20] ^ A[23];
		tt0 ^= A[21];
		tt2 ^= tt3;
		t0 = tt0 ^ tt2;

		tt0 = A[12] ^ A[10];
		tt1 = A[13] ^ A[11];
		tt0 ^= A[14] ^ tt1;
		tt0 = (tt0 << 1) | (tt0 >> 63);
		tt2 = A[ 0] ^ A[ 3];
		tt3 = A[ 1] ^ A[ 4];
		tt0 ^= A[ 2];
		tt2 ^= tt3;
		t1 = tt0 ^ tt2;

		tt0 = A[18] ^ A[16];
		tt1 = A[19] ^ A[17];
		tt0 ^= A[15] ^ tt1;
		tt0 = (tt0 << 1) | (tt0 >> 63);
		tt2 = A[ 6] ^ A[ 9];
		tt3 = A[ 7] ^ A[ 5];
		tt0 ^= A[ 8];
		tt2 ^= tt3;
		t2 = tt0 ^ tt2;

		tt0 = A[24] ^ A[22];
		tt1 = A[20] ^ A[23];
		tt0 ^= A[21] ^ tt1;
		tt0 = (tt0 << 1) | (tt0 >> 63);
		tt2 = A[12] ^ A[10];
		tt3 = A[13] ^ A[11];
		tt0 ^= A[14];
		tt2 ^= tt3;
		t3 = tt0 ^ tt2;

		tt0 = A[ 0] ^ A[ 3];
		tt1 = A[ 1] ^ A[ 4];
		tt0 ^= A[ 2] ^ tt1;
		tt0 = (tt0 << 1) | (tt0 >> 63);
		tt2 = A[18] ^ A[16];
		tt3 = A[19] ^ A[17];
		tt0 ^= A[15];
		tt2 ^= tt3;
		t4 = tt0 ^ tt2;

		A[ 0] = A[ 0] ^ t0;
		A[ 3] = A[ 3] ^ t0;
		A[ 1] = A[ 1] ^ t0;
		A[ 4] = A[ 4] ^ t0;
		A[ 2] = A[ 2] ^ t0;
		A[ 6] = A[ 6] ^ t1;
		A[ 9] = A[ 9] ^ t1;
		A[ 7] = A[ 7] ^ t1;
		A[ 5] = A[ 5] ^ t1;
		A[ 8] = A[ 8] ^ t1;
		A[12] = A[12] ^ t2;
		A[10] = A[10] ^ t2;
		A[13] = A[13] ^ t2;
		A[11] = A[11] ^ t2;
		A[14] = A[14] ^ t2;
		A[18] = A[18] ^ t3;
		A[16] = A[16] ^ t3;
		A[19] = A[19] ^ t3;
		A[17] = A[17] ^ t3;
		A[15] = A[15] ^ t3;
		A[24] = A[24] ^ t4;
		A[22] = A[22] ^ t4;
		A[20] = A[20] ^ t4;
		A[23] = A[23] ^ t4;
		A[21] = A[21] ^ t4;
		A[ 3] = (A[ 3] << 36) | (A[ 3] >> (64 - 36));
		A[ 1] = (A[ 1] <<  3) | (A[ 1] >> (64 -  3));
		A[ 4] = (A[ 4] << 41) | (A[ 4] >> (64 - 41));
		A[ 2] = (A[ 2] << 18) | (A[ 2] >> (64 - 18));
		A[ 6] = (A[ 6] <<  1) | (A[ 6] >> (64 -  1));
		A[ 9] = (A[ 9] << 44) | (A[ 9] >> (64 - 44));
		A[ 7] = (A[ 7] << 10) | (A[ 7] >> (64 - 10));
		A[ 5] = (A[ 5] << 45) | (A[ 5] >> (64 - 45));
		A[ 8] = (A[ 8] <<  2) | (A[ 8] >> (64 - 2));
		A[12] = (A[12] << 62) | (A[12] >> (64 - 62));
		A[10] = (A[10] <<  6) | (A[10] >> (64 -  6));
		A[13] = (A[13] << 43) | (A[13] >> (64 - 43));
		A[11] = (A[11] << 15) | (A[11] >> (64 - 15));
		A[14] = (A[14] << 61) | (A[14] >> (64 - 61));
		A[18] = (A[18] << 28) | (A[18] >> (64 - 28));
		A[16] = (A[16] << 55) | (A[16] >> (64 - 55));
		A[19] = (A[19] << 25) | (A[19] >> (64 - 25));
		A[17] = (A[17] << 21) | (A[17] >> (64 - 21));
		A[15] = (A[15] << 56) | (A[15] >> (64 - 56));
		A[24] = (A[24] << 27) | (A[24] >> (64 - 27));
		A[22] = (A[22] << 20) | (A[22] >> (64 - 20));
		A[20] = (A[20] << 39) | (A[20] >> (64 - 39));
		A[23] = (A[23] <<  8) | (A[23] >> (64 -  8));
		A[21] = (A[21] << 14) | (A[21] >> (64 - 14));

		bnn = ~A[13];
		kt = A[ 9] | A[13];
		c0 = A[ 0] ^ kt;
		kt = bnn | A[17];
		c1 = A[ 9] ^ kt;
		kt = A[17] & A[21];
		c2 = A[13] ^ kt;
		kt = A[21] | A[ 0];
		c3 = A[17] ^ kt;
		kt = A[ 0] & A[ 9];
		c4 = A[21] ^ kt;
		A[ 0] = c0;
		A[ 9] = c1;
		A[13] = c2;
		A[17] = c3;
		A[21] = c4;
		bnn = ~A[14];
		kt = A[22] | A[ 1];
		c0 = A[18] ^ kt;
		kt = A[ 1] & A[ 5];
		c1 = A[22] ^ kt;
		kt = A[ 5] | bnn;
		c2 = A[ 1] ^ kt;
		kt = A[14] | A[18];
		c3 = A[ 5] ^ kt;
		kt = A[18] & A[22];
		c4 = A[14] ^ kt;
		A[18] = c0;
		A[22] = c1;
		A[ 1] = c2;
		A[ 5] = c3;
		A[14] = c4;
		bnn = ~A[23];
		kt = A[10] | A[19];
		c0 = A[ 6] ^ kt;
		kt = A[19] & A[23];
		c1 = A[10] ^ kt;
		kt = bnn & A[ 2];
		c2 = A[19] ^ kt;
		kt = A[ 2] | A[ 6];
		c3 = bnn ^ kt;
		kt = A[ 6] & A[10];
		c4 = A[ 2] ^ kt;
		A[ 6] = c0;
		A[10] = c1;
		A[19] = c2;
		A[23] = c3;
		A[ 2] = c4;
		bnn = ~A[11];
		kt = A[ 3] & A[ 7];
		c0 = A[24] ^ kt;
		kt = A[ 7] | A[11];
		c1 = A[ 3] ^ kt;
		kt = bnn | A[15];
		c2 = A[ 7] ^ kt;
		kt = A[15] & A[24];
		c3 = bnn ^ kt;
		kt = A[24] | A[ 3];
		c4 = A[15] ^ kt;
		A[24] = c0;
		A[ 3] = c1;
		A[ 7] = c2;
		A[11] = c3;
		A[15] = c4;
		bnn = ~A[16];
		kt = bnn & A[20];
		c0 = A[12] ^ kt;
		kt = A[20] | A[ 4];
		c1 = bnn ^ kt;
		kt = A[ 4] & A[ 8];
		c2 = A[20] ^ kt;
		kt = A[ 8] | A[12];
		c3 = A[ 4] ^ kt;
		kt = A[12] & A[16];
		c4 = A[ 8] ^ kt;
		A[12] = c0;
		A[16] = c1;
		A[20] = c2;
		A[ 4] = c3;
		A[ 8] = c4;
		A[ 0] = A[ 0] ^ RC[j + 1];
		t = A[ 5];
		A[ 5] = A[18];
		A[18] = A[11];
		A[11] = A[10];
		A[10] = A[ 6];
		A[ 6] = A[22];
		A[22] = A[20];
		A[20] = A[12];
		A[12] = A[19];
		A[19] = A[15];
		A[15] = A[24];
		A[24] = A[ 8];
		A[ 8] = t;
		t = A[ 1];
		A[ 1] = A[ 9];
		A[ 9] = A[14];
		A[14] = A[ 2];
		A[ 2] = A[13];
		A[13] = A[23];
		A[23] = A[ 4];
		A[ 4] = A[21];
		A[21] = A[16];
		A[16] = A[ 3];
		A[ 3] = A[17];
		A[17] = A[ 7];
		A[ 7] = t;
	}

	/*
	 * Invert some words back to normal representation.
	 */
	A[ 1] = ~A[ 1];
	A[ 2] = ~A[ 2];
	A[ 8] = ~A[ 8];
	A[12] = ~A[12];
	A[17] = ~A[17];
	A[20] = ~A[20];
}

__device__ void process_block_2(uint64_t *A)
{
    uint64_t t0, t1, t2, t3, t4;
    uint64_t tt0, tt1, tt2, tt3;
    uint64_t t, kt;
    uint64_t c0, c1, c2, c3, c4, bnn;
    int j;

    /*
     * Invert some words (alternate internal representation, which
     * saves some operations).
     */
    A[ 1] = ~A[ 1];
    A[ 2] = ~A[ 2];
    A[ 8] = ~A[ 8];
    A[12] = ~A[12];
    A[17] = ~A[17];
    A[20] = ~A[20];

    /*
     * Compute the 24 rounds. This loop is partially unrolled (each
     * iteration computes two rounds).
     */
    for (j = 2 * threadIdx.x; j < 24; j +=blockDim.x) {

        tt0 = A[ 1] ^ A[ 6];
        tt1 = A[11] ^ A[16];
        tt0 ^= A[21] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[ 4] ^ A[ 9];
        tt3 = A[14] ^ A[19];
        tt0 ^= A[24];
        tt2 ^= tt3;
        t0 = tt0 ^ tt2;

        tt0 = A[ 2] ^ A[ 7];
        tt1 = A[12] ^ A[17];
        tt0 ^= A[22] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[ 0] ^ A[ 5];
        tt3 = A[10] ^ A[15];
        tt0 ^= A[20];
        tt2 ^= tt3;
        t1 = tt0 ^ tt2;

        tt0 = A[ 3] ^ A[ 8];
        tt1 = A[13] ^ A[18];
        tt0 ^= A[23] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[ 1] ^ A[ 6];
        tt3 = A[11] ^ A[16];
        tt0 ^= A[21];
        tt2 ^= tt3;
        t2 = tt0 ^ tt2;

        tt0 = A[ 4] ^ A[ 9];
        tt1 = A[14] ^ A[19];
        tt0 ^= A[24] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[ 2] ^ A[ 7];
        tt3 = A[12] ^ A[17];
        tt0 ^= A[22];
        tt2 ^= tt3;
        t3 = tt0 ^ tt2;

        tt0 = A[ 0] ^ A[ 5];
        tt1 = A[10] ^ A[15];
        tt0 ^= A[20] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[ 3] ^ A[ 8];
        tt3 = A[13] ^ A[18];
        tt0 ^= A[23];
        tt2 ^= tt3;
        t4 = tt0 ^ tt2;

        A[ 0] = A[ 0] ^ t0;
        A[ 5] = A[ 5] ^ t0;
        A[10] = A[10] ^ t0;
        A[15] = A[15] ^ t0;
        A[20] = A[20] ^ t0;
        A[ 1] = A[ 1] ^ t1;
        A[ 6] = A[ 6] ^ t1;
        A[11] = A[11] ^ t1;
        A[16] = A[16] ^ t1;
        A[21] = A[21] ^ t1;
        A[ 2] = A[ 2] ^ t2;
        A[ 7] = A[ 7] ^ t2;
        A[12] = A[12] ^ t2;
        A[17] = A[17] ^ t2;
        A[22] = A[22] ^ t2;
        A[ 3] = A[ 3] ^ t3;
        A[ 8] = A[ 8] ^ t3;
        A[13] = A[13] ^ t3;
        A[18] = A[18] ^ t3;
        A[23] = A[23] ^ t3;
        A[ 4] = A[ 4] ^ t4;
        A[ 9] = A[ 9] ^ t4;
        A[14] = A[14] ^ t4;
        A[19] = A[19] ^ t4;
        A[24] = A[24] ^ t4;
        A[ 5] = (A[ 5] << 36) | (A[ 5] >> (64 - 36));
        A[10] = (A[10] <<  3) | (A[10] >> (64 -  3));
        A[15] = (A[15] << 41) | (A[15] >> (64 - 41));
        A[20] = (A[20] << 18) | (A[20] >> (64 - 18));
        A[ 1] = (A[ 1] <<  1) | (A[ 1] >> (64 -  1));
        A[ 6] = (A[ 6] << 44) | (A[ 6] >> (64 - 44));
        A[11] = (A[11] << 10) | (A[11] >> (64 - 10));
        A[16] = (A[16] << 45) | (A[16] >> (64 - 45));
        A[21] = (A[21] <<  2) | (A[21] >> (64 - 2));
        A[ 2] = (A[ 2] << 62) | (A[ 2] >> (64 - 62));
        A[ 7] = (A[ 7] <<  6) | (A[ 7] >> (64 -  6));
        A[12] = (A[12] << 43) | (A[12] >> (64 - 43));
        A[17] = (A[17] << 15) | (A[17] >> (64 - 15));
        A[22] = (A[22] << 61) | (A[22] >> (64 - 61));
        A[ 3] = (A[ 3] << 28) | (A[ 3] >> (64 - 28));
        A[ 8] = (A[ 8] << 55) | (A[ 8] >> (64 - 55));
        A[13] = (A[13] << 25) | (A[13] >> (64 - 25));
        A[18] = (A[18] << 21) | (A[18] >> (64 - 21));
        A[23] = (A[23] << 56) | (A[23] >> (64 - 56));
        A[ 4] = (A[ 4] << 27) | (A[ 4] >> (64 - 27));
        A[ 9] = (A[ 9] << 20) | (A[ 9] >> (64 - 20));
        A[14] = (A[14] << 39) | (A[14] >> (64 - 39));
        A[19] = (A[19] <<  8) | (A[19] >> (64 -  8));
        A[24] = (A[24] << 14) | (A[24] >> (64 - 14));

        bnn = ~A[12];
        kt = A[ 6] | A[12];
        c0 = A[ 0] ^ kt;
        kt = bnn | A[18];
        c1 = A[ 6] ^ kt;
        kt = A[18] & A[24];
        c2 = A[12] ^ kt;
        kt = A[24] | A[ 0];
        c3 = A[18] ^ kt;
        kt = A[ 0] & A[ 6];
        c4 = A[24] ^ kt;
        A[ 0] = c0;
        A[ 6] = c1;
        A[12] = c2;
        A[18] = c3;
        A[24] = c4;
        bnn = ~A[22];
        kt = A[ 9] | A[10];
        c0 = A[ 3] ^ kt;
        kt = A[10] & A[16];
        c1 = A[ 9] ^ kt;
        kt = A[16] | bnn;
        c2 = A[10] ^ kt;
        kt = A[22] | A[ 3];
        c3 = A[16] ^ kt;
        kt = A[ 3] & A[ 9];
        c4 = A[22] ^ kt;
        A[ 3] = c0;
        A[ 9] = c1;
        A[10] = c2;
        A[16] = c3;
        A[22] = c4;
        bnn = ~A[19];
        kt = A[ 7] | A[13];
        c0 = A[ 1] ^ kt;
        kt = A[13] & A[19];
        c1 = A[ 7] ^ kt;
        kt = bnn & A[20];
        c2 = A[13] ^ kt;
        kt = A[20] | A[ 1];
        c3 = bnn ^ kt;
        kt = A[ 1] & A[ 7];
        c4 = A[20] ^ kt;
        A[ 1] = c0;
        A[ 7] = c1;
        A[13] = c2;
        A[19] = c3;
        A[20] = c4;
        bnn = ~A[17];
        kt = A[ 5] & A[11];
        c0 = A[ 4] ^ kt;
        kt = A[11] | A[17];
        c1 = A[ 5] ^ kt;
        kt = bnn | A[23];
        c2 = A[11] ^ kt;
        kt = A[23] & A[ 4];
        c3 = bnn ^ kt;
        kt = A[ 4] | A[ 5];
        c4 = A[23] ^ kt;
        A[ 4] = c0;
        A[ 5] = c1;
        A[11] = c2;
        A[17] = c3;
        A[23] = c4;
        bnn = ~A[ 8];
        kt = bnn & A[14];
        c0 = A[ 2] ^ kt;
        kt = A[14] | A[15];
        c1 = bnn ^ kt;
        kt = A[15] & A[21];
        c2 = A[14] ^ kt;
        kt = A[21] | A[ 2];
        c3 = A[15] ^ kt;
        kt = A[ 2] & A[ 8];
        c4 = A[21] ^ kt;
        A[ 2] = c0;
        A[ 8] = c1;
        A[14] = c2;
        A[15] = c3;
        A[21] = c4;
        A[ 0] = A[ 0] ^ RC[j + 0];

        tt0 = A[ 6] ^ A[ 9];
        tt1 = A[ 7] ^ A[ 5];
        tt0 ^= A[ 8] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[24] ^ A[22];
        tt3 = A[20] ^ A[23];
        tt0 ^= A[21];
        tt2 ^= tt3;
        t0 = tt0 ^ tt2;

        tt0 = A[12] ^ A[10];
        tt1 = A[13] ^ A[11];
        tt0 ^= A[14] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[ 0] ^ A[ 3];
        tt3 = A[ 1] ^ A[ 4];
        tt0 ^= A[ 2];
        tt2 ^= tt3;
        t1 = tt0 ^ tt2;

        tt0 = A[18] ^ A[16];
        tt1 = A[19] ^ A[17];
        tt0 ^= A[15] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[ 6] ^ A[ 9];
        tt3 = A[ 7] ^ A[ 5];
        tt0 ^= A[ 8];
        tt2 ^= tt3;
        t2 = tt0 ^ tt2;

        tt0 = A[24] ^ A[22];
        tt1 = A[20] ^ A[23];
        tt0 ^= A[21] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[12] ^ A[10];
        tt3 = A[13] ^ A[11];
        tt0 ^= A[14];
        tt2 ^= tt3;
        t3 = tt0 ^ tt2;

        tt0 = A[ 0] ^ A[ 3];
        tt1 = A[ 1] ^ A[ 4];
        tt0 ^= A[ 2] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[18] ^ A[16];
        tt3 = A[19] ^ A[17];
        tt0 ^= A[15];
        tt2 ^= tt3;
        t4 = tt0 ^ tt2;

        A[ 0] = A[ 0] ^ t0;
        A[ 3] = A[ 3] ^ t0;
        A[ 1] = A[ 1] ^ t0;
        A[ 4] = A[ 4] ^ t0;
        A[ 2] = A[ 2] ^ t0;
        A[ 6] = A[ 6] ^ t1;
        A[ 9] = A[ 9] ^ t1;
        A[ 7] = A[ 7] ^ t1;
        A[ 5] = A[ 5] ^ t1;
        A[ 8] = A[ 8] ^ t1;
        A[12] = A[12] ^ t2;
        A[10] = A[10] ^ t2;
        A[13] = A[13] ^ t2;
        A[11] = A[11] ^ t2;
        A[14] = A[14] ^ t2;
        A[18] = A[18] ^ t3;
        A[16] = A[16] ^ t3;
        A[19] = A[19] ^ t3;
        A[17] = A[17] ^ t3;
        A[15] = A[15] ^ t3;
        A[24] = A[24] ^ t4;
        A[22] = A[22] ^ t4;
        A[20] = A[20] ^ t4;
        A[23] = A[23] ^ t4;
        A[21] = A[21] ^ t4;
        A[ 3] = (A[ 3] << 36) | (A[ 3] >> (64 - 36));
        A[ 1] = (A[ 1] <<  3) | (A[ 1] >> (64 -  3));
        A[ 4] = (A[ 4] << 41) | (A[ 4] >> (64 - 41));
        A[ 2] = (A[ 2] << 18) | (A[ 2] >> (64 - 18));
        A[ 6] = (A[ 6] <<  1) | (A[ 6] >> (64 -  1));
        A[ 9] = (A[ 9] << 44) | (A[ 9] >> (64 - 44));
        A[ 7] = (A[ 7] << 10) | (A[ 7] >> (64 - 10));
        A[ 5] = (A[ 5] << 45) | (A[ 5] >> (64 - 45));
        A[ 8] = (A[ 8] <<  2) | (A[ 8] >> (64 - 2));
        A[12] = (A[12] << 62) | (A[12] >> (64 - 62));
        A[10] = (A[10] <<  6) | (A[10] >> (64 -  6));
        A[13] = (A[13] << 43) | (A[13] >> (64 - 43));
        A[11] = (A[11] << 15) | (A[11] >> (64 - 15));
        A[14] = (A[14] << 61) | (A[14] >> (64 - 61));
        A[18] = (A[18] << 28) | (A[18] >> (64 - 28));
        A[16] = (A[16] << 55) | (A[16] >> (64 - 55));
        A[19] = (A[19] << 25) | (A[19] >> (64 - 25));
        A[17] = (A[17] << 21) | (A[17] >> (64 - 21));
        A[15] = (A[15] << 56) | (A[15] >> (64 - 56));
        A[24] = (A[24] << 27) | (A[24] >> (64 - 27));
        A[22] = (A[22] << 20) | (A[22] >> (64 - 20));
        A[20] = (A[20] << 39) | (A[20] >> (64 - 39));
        A[23] = (A[23] <<  8) | (A[23] >> (64 -  8));
        A[21] = (A[21] << 14) | (A[21] >> (64 - 14));

        bnn = ~A[13];
        kt = A[ 9] | A[13];
        c0 = A[ 0] ^ kt;
        kt = bnn | A[17];
        c1 = A[ 9] ^ kt;
        kt = A[17] & A[21];
        c2 = A[13] ^ kt;
        kt = A[21] | A[ 0];
        c3 = A[17] ^ kt;
        kt = A[ 0] & A[ 9];
        c4 = A[21] ^ kt;
        A[ 0] = c0;
        A[ 9] = c1;
        A[13] = c2;
        A[17] = c3;
        A[21] = c4;
        bnn = ~A[14];
        kt = A[22] | A[ 1];
        c0 = A[18] ^ kt;
        kt = A[ 1] & A[ 5];
        c1 = A[22] ^ kt;
        kt = A[ 5] | bnn;
        c2 = A[ 1] ^ kt;
        kt = A[14] | A[18];
        c3 = A[ 5] ^ kt;
        kt = A[18] & A[22];
        c4 = A[14] ^ kt;
        A[18] = c0;
        A[22] = c1;
        A[ 1] = c2;
        A[ 5] = c3;
        A[14] = c4;
        bnn = ~A[23];
        kt = A[10] | A[19];
        c0 = A[ 6] ^ kt;
        kt = A[19] & A[23];
        c1 = A[10] ^ kt;
        kt = bnn & A[ 2];
        c2 = A[19] ^ kt;
        kt = A[ 2] | A[ 6];
        c3 = bnn ^ kt;
        kt = A[ 6] & A[10];
        c4 = A[ 2] ^ kt;
        A[ 6] = c0;
        A[10] = c1;
        A[19] = c2;
        A[23] = c3;
        A[ 2] = c4;
        bnn = ~A[11];
        kt = A[ 3] & A[ 7];
        c0 = A[24] ^ kt;
        kt = A[ 7] | A[11];
        c1 = A[ 3] ^ kt;
        kt = bnn | A[15];
        c2 = A[ 7] ^ kt;
        kt = A[15] & A[24];
        c3 = bnn ^ kt;
        kt = A[24] | A[ 3];
        c4 = A[15] ^ kt;
        A[24] = c0;
        A[ 3] = c1;
        A[ 7] = c2;
        A[11] = c3;
        A[15] = c4;
        bnn = ~A[16];
        kt = bnn & A[20];
        c0 = A[12] ^ kt;
        kt = A[20] | A[ 4];
        c1 = bnn ^ kt;
        kt = A[ 4] & A[ 8];
        c2 = A[20] ^ kt;
        kt = A[ 8] | A[12];
        c3 = A[ 4] ^ kt;
        kt = A[12] & A[16];
        c4 = A[ 8] ^ kt;
        A[12] = c0;
        A[16] = c1;
        A[20] = c2;
        A[ 4] = c3;
        A[ 8] = c4;
        A[ 0] = A[ 0] ^ RC[j + 1];
        t = A[ 5];
        A[ 5] = A[18];
        A[18] = A[11];
        A[11] = A[10];
        A[10] = A[ 6];
        A[ 6] = A[22];
        A[22] = A[20];
        A[20] = A[12];
        A[12] = A[19];
        A[19] = A[15];
        A[15] = A[24];
        A[24] = A[ 8];
        A[ 8] = t;
        t = A[ 1];
        A[ 1] = A[ 9];
        A[ 9] = A[14];
        A[14] = A[ 2];
        A[ 2] = A[13];
        A[13] = A[23];
        A[23] = A[ 4];
        A[ 4] = A[21];
        A[21] = A[16];
        A[16] = A[ 3];
        A[ 3] = A[17];
        A[17] = A[ 7];
        A[ 7] = t;
    }

    /*
     * Invert some words back to normal representation.
     */
    A[ 1] = ~A[ 1];
    A[ 2] = ~A[ 2];
    A[ 8] = ~A[ 8];
    A[12] = ~A[12];
    A[17] = ~A[17];
    A[20] = ~A[20];
}


/* see inner.h */
__global__ void i_shake256_init_gpu(uint64_t *scA, uint64_t *d_scdptr)
{
	uint32_t bid = blockIdx.x, i;
	d_scdptr[bid] = 0;
	for(i=0; i<25; i++) scA[bid*25 + i] = 0;
}

__global__ void i_shake256_init_gpu_kernel(uint64_t *d_scA, uint64_t *d_scdptr, size_t d_mem_pool_pitch)
{
    uint32_t i;
    uint64_t *scA = d_scA + blockIdx.x * d_mem_pool_pitch / sizeof(uint64_t);
    uint64_t *scdptr = d_scdptr + blockIdx.x * d_mem_pool_pitch / sizeof(uint64_t);


    *scdptr = 0;
    for(i=0; i<25; i++) scA[i] = 0;
}

/* see inner.h */
__global__ void i_shake256_inject_gpu(uint64_t *scA, uint64_t *d_scdptr, const uint8_t *in, uint32_t *d_len)
{
	size_t dptr;
	uint32_t bid = blockIdx.x, i;
	uint64_t len = NONCELEN + d_len[bid];

	dptr = (uint64_t)d_scdptr[bid];
	
	for(i=0; i<25; i++) scA[bid*25 + i] = 0;
		__syncthreads();
	while (len > 0) {
		uint32_t clen, u;

		clen = 136 - dptr;
		if (clen > len) {
			clen = len;
		}
		for (u = 0; u < clen; u ++) {
			uint32_t v;

			v = u + dptr;
			scA[(bid*25) + (v >> 3)] ^= (uint64_t)in[bid*((MLEN + CRYPTO_BYTES)) + u] << ((v & 7) << 3);
		}
		dptr += clen;
		in += clen;
		len -= clen;
		if (dptr == 136) {
			process_block(scA+ bid*25);
			dptr = 0;
		}
	}
	d_scdptr[bid] = dptr;
}

__global__ void i_shake256_inject_gpu2(uint64_t *scA, uint64_t *d_scdptr, const uint8_t *in, uint32_t len)
{
	size_t dptr;
	uint32_t bid = blockIdx.x;
	dptr = (uint64_t)d_scdptr[bid];

	while (len > 0) {
		uint32_t clen, u;

		clen = 136 - dptr;
		if (clen > len) {
			clen = len;
		}
		for (u = 0; u < clen; u ++) {
			uint32_t v;

			v = u + dptr;
			scA[(bid*25) + (v >> 3)] ^= (uint64_t)in[bid*len+u] << ((v & 7) << 3);
		}
		dptr += clen;
		in += clen;
		len -= clen;
		if (dptr == 136) {
			process_block(scA+ bid*25);
			dptr = 0;
		}
	}
	d_scdptr[bid] = dptr;
}

__global__ void i_shake256_inject_gpu_kernel(uint64_t *d_scA, uint64_t *d_scdptr, const uint8_t *d_in, uint32_t len, size_t d_mem_pool_pitch)
{
    size_t dptr;
    uint64_t *tmp = d_scdptr + blockIdx.x * d_mem_pool_pitch / sizeof(uint64_t);
    dptr = (uint64_t)*tmp;

    uint64_t *scA = d_scA + blockIdx.x * d_mem_pool_pitch / sizeof(uint64_t);
    auto *in = d_in + blockIdx.x * d_mem_pool_pitch / sizeof(uint8_t);


    while (len > 0) {
        uint32_t clen, u;

        clen = 136 - dptr;
        if (clen > len) {
            clen = len;
        }
        for (u = 0; u < clen; u ++) {
            uint32_t v;

            v = u + dptr;
            scA[(v >> 3)] ^= (uint64_t)in[u] << ((v & 7) << 3);
        }

        dptr += clen;
        in += clen;
        len -= clen;

        if (dptr == 136) {
            process_block(scA);
            dptr = 0;
        }
    }
    *tmp = dptr;
}

__global__ void i_shake256_inject_gpu_kernel_2(uint64_t *d_scA, uint64_t *d_scdptr, const uint8_t *d_in, uint32_t len, size_t d_mem_pool_pitch)
{
    size_t dptr;
    uint64_t *tmp = d_scdptr + blockIdx.x * d_mem_pool_pitch/sizeof(uint64_t);
    dptr = *tmp;


    uint64_t *scA = d_scA + blockIdx.x * d_mem_pool_pitch/sizeof(uint64_t);
    auto *in = d_in + blockIdx.x * d_mem_pool_pitch;

    while (len > 0) {
        uint32_t clen, u;

        clen = 136 - dptr;
        if (clen > len) {
            clen = len;
        }
        for (u = 0; u < clen; u ++) {
            uint32_t v;

            v = u + dptr;
            scA[(v >> 3)] ^= (uint64_t)in[u] << ((v & 7) << 3);
        }
        dptr += clen;
        in += clen;

        len -= clen;
        if (dptr == 136) {
            process_block(scA);
            dptr = 0;
        }
    }
    *tmp = dptr;
}


__global__ void i_shake256_inject_gpu_2(uint64_t *scA, uint64_t *d_scdptr, const uint8_t *in, uint32_t len)
{
    size_t dptr;
    uint32_t bid = blockIdx.x;
    // uint64_t len = d_len[bid];

    dptr = (uint64_t)d_scdptr[bid];

    while (len > 0) {
        uint32_t clen, u;

        clen = 136 - dptr;
        if (clen > len) {
            clen = len;
        }
        for (u = 0; u < clen; u ++) {
            uint32_t v;

            v = u + dptr;
            scA[(bid*25) + (v >> 3)] ^= (uint64_t)in[bid*len+u] << ((v & 7) << 3);
        }

        dptr += clen;
        in += clen;
        len -= clen;
        if (dptr == 136) {
            process_block_2(scA+ bid*25);
            dptr = 0;
        }
    }
    d_scdptr[bid] = dptr;
}

__global__ void i_shake256_inject_gpu2_2(uint64_t *scA, uint64_t *d_scdptr, const uint8_t *in, uint32_t len)
{
	size_t dptr;
	uint32_t bid = blockIdx.x;

	dptr = (uint64_t)d_scdptr[bid];

	while (len > 0) {
		uint32_t clen, u;

		clen = 136 - dptr;
		if (clen > len) {
			clen = len;
		}
		for (u = 0; u < clen; u ++) {
			uint32_t v;

			v = u + dptr;
			scA[(bid*25) + (v >> 3)] ^= (uint64_t)in[bid*len+u] << ((v & 7) << 3);
		}
		dptr += clen;
		in += clen;
		len -= clen;
		if (dptr == 136) {
			process_block(scA+ bid*25);
			dptr = 0;
		}
	}
	d_scdptr[bid] = dptr;
}

/* see falcon.h */
__global__ void i_shake256_flip_gpu(uint64_t *scA, uint64_t  *scdptr)
{
	/*
	 * We apply padding and pre-XOR the value into the state. We set dptr to the end of the buffer, so that first call to shake_extract() will process the block.
	 */
	unsigned v;
	uint32_t bid = blockIdx.x;
	v = scdptr[bid];
	scA[(bid*25) + (v >> 3)] ^= (uint64_t)0x1F << ((v & 7) << 3);
	scA[bid*25 +16] ^= (uint64_t)0x80 << 56;
	scdptr[bid] = 136;
}

__global__ void i_shake256_flip_gpu_kernel(uint64_t *d_scA, uint64_t  *d_scdptr, size_t d_mem_pool_pitch)
{
    /*
     * We apply padding and pre-XOR the value into the state. We set dptr to the end of the buffer, so that first call to shake_extract() will process the block.
     */
    unsigned v;

    uint64_t *scA = d_scA + blockIdx.x * d_mem_pool_pitch / sizeof(uint64_t);
    uint64_t *scdptr = d_scdptr + blockIdx.x * d_mem_pool_pitch / sizeof(uint64_t);


    v = *scdptr;
    scA[(v >> 3)] ^= (uint64_t)0x1F << ((v & 7) << 3);
    scA[16] ^= (uint64_t)0x80 << 56;
    *scdptr = 136;
}


__global__ void i_shake256_inject_gpu_kernel_2(uint64_t *d_scA, uint64_t *d_d_scdptr, const uint8_t *d_in, uint32_t *d_d_len, size_t d_mem_pool_pitch)
{

    uint64_t *scA = d_scA + blockIdx.x * d_mem_pool_pitch/sizeof(uint64_t);
    uint64_t *d_scdptr = d_d_scdptr + blockIdx.x * d_mem_pool_pitch/sizeof(uint64_t);
    auto *in = d_in + blockIdx.x * d_mem_pool_pitch;
    uint32_t *d_len = d_d_len+ blockIdx.x * d_mem_pool_pitch/sizeof(uint32_t);

    size_t dptr;
    uint32_t i;
    uint64_t len = NONCELEN + *d_len;

    dptr = (uint64_t)*d_scdptr;

    for(i=0; i<25; i++) scA[i] = 0;
    __syncthreads();
    while (len > 0) {
        uint32_t clen, u;

        clen = 136 - dptr;
        if (clen > len) {
            clen = len;
        }
        for (u = 0; u < clen; u ++) {
            uint32_t v;

            v = u + dptr;
            scA[(v >> 3)] ^= (uint64_t)in[u] << ((v & 7) << 3);
        }
        dptr += clen;
        in += clen;
        len -= clen;
        if (dptr == 136) {
            process_block(scA);
            dptr = 0;
        }
    }
    *d_scdptr = dptr;
}



/* see falcon.h */
__device__ void i_shake256_extract_gpu(uint64_t *scA, uint64_t  *scdptr, uint8_t *out, uint32_t len)
{
	size_t dptr;	
	uint32_t bid = blockIdx.x;
	
	dptr = (size_t)scdptr[bid];
	while (len > 0) {
		size_t clen;

		if (dptr == 136) {
			process_block(scA);
			dptr = 0;
		}
		clen = 136 - dptr;
		if (clen > len) {
			clen = len;
		}
		len -= clen;
		while (clen -- > 0) {
			*out ++ = scA[dptr >> 3] >> ((dptr & 7) << 3);
			dptr ++;
		}
	}
	scdptr[bid] = dptr;
}

__global__ void test_shake256(uint64_t *in)
{
	process_block(in);
}

__device__ void shake256_gpu(uint64_t *A) 
{
    uint32_t tid = threadIdx.x;
    uint32_t s = threadIdx.x % 5;
    
    __shared__ uint64_t C[25];
    __shared__ uint64_t D[25];

    if (tid < 25) {                      
        for (int i = 0; i<NROUNDS; ++i) {
            C[tid] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
            D[tid] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
            C[tid] = R64(A[a[tid]] ^ D[b[tid]], ro[tid][0], ro[tid][1]);
            A[d[tid]] = C[c[tid][0]] ^ ((~C[c[tid][1]]) & C[c[tid][2]]);            
            A[tid] ^= rc[(tid == 0) ? 0 : 1][i];
        }          
    }       
}

__device__ void shake256_gpu_warp(uint64_t *d_data) {

	uint32_t tid = threadIdx.x;
	uint32_t s = threadIdx.x % 5;
	uint64_t _C = 0, _D = 0;

	__shared__ uint64_t A[25];
	uint32_t a_s, b_s, d_s;
	uint64_t A_s;

	if (tid < 25) {
		a_s = a[tid];
		b_s = b[tid];
		d_s = d[tid];
		A[tid] = d_data[tid];

		for (int i = 0; i<24; ++i) {

			_C = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
			_D = __shfl_sync(0xffffffff, _C, __shfl_sync(0xffffffff,b_s, 20 + s)) ^ R64(__shfl_sync(0xffffffff,_C, __shfl_sync(0xffffffff,b_s, 5 + s)), 1, 63);
			_C = R64(A[__shfl_sync(0xffffffff,a_s, tid)] ^ __shfl_sync(0xffffffff,_D, __shfl_sync(0xffffffff,b_s, tid)), ro[tid][0], ro[tid][1]);
			A_s = __shfl_sync(0xffffffff,_C, c[tid][0]) ^ ((~(__shfl_sync(0xffffffff,_C, c[tid][1]))) & __shfl_sync(0xffffffff,_C, c[tid][2]));
			A[__shfl_sync(0xffffffff,d_s, tid)] = A_s;

			A[tid] ^= rc[(tid == 0) ? 0 : 1][i];
		}

		d_data[tid] = A[tid];
	}
}

__global__ void hash_to_point_vartime_par(uint64_t *g_scA, uint64_t *scdptr, uint16_t *x)
{
	uint64_t dptr;
	uint32_t tid = threadIdx.x, bid = blockIdx.x;
	uint32_t len, countN;	
	uint8_t out[2];
	uint32_t w, n = Falcon_N;
	__shared__ uint64_t scA[25];

	dptr = scdptr[bid];
	scA[tid] = g_scA[bid*25 + tid];
	__syncthreads();
	countN = 0;
	
	while(n > 0){
		len = 2;
		while (len > 0) {			

			if (dptr == 136) {
				shake256_gpu(scA);
				dptr = 0;
			}
			__syncthreads();
			len -= 2;			
			
			out[0] = scA[(dptr >> 3)] >> ((dptr & 7) << 3);
			dptr ++;
			out[1] = scA[(dptr >> 3)] >> ((dptr & 7) << 3);
			dptr ++;
		}
        __syncwarp();
		
		w = ((unsigned)out[0] << 8) | (unsigned)out[1];
		if (w < 61445) {
			while (w >= 12289) {
				w -= 12289;
			}				
			x[bid * Falcon_N + countN] = (uint16_t)w;
			n--;		
			countN++;		
		}
	}
    __syncwarp();
	scdptr[bid] = dptr;
}

__global__ void hash_to_point_vartime_par_kernel(uint64_t *d_scA, uint64_t *d_scdptr, uint16_t *d_x, size_t d_mem_pool_pitch)
{
    uint64_t dptr;
    uint32_t tid = threadIdx.x;
    uint32_t len, countN;
    uint8_t out[2];
    uint32_t w, n = Falcon_N;
    __shared__ uint64_t scA[25];

    uint64_t *g_scA = d_scA + blockIdx.x * d_mem_pool_pitch /sizeof(uint64_t);
    uint64_t *scdptr = d_scdptr + blockIdx.x * d_mem_pool_pitch /sizeof(uint64_t);
    uint16_t *x = d_x + blockIdx.x * d_mem_pool_pitch /sizeof(uint16_t);

    dptr = *scdptr;
    scA[tid] = g_scA[tid];
    __syncthreads();
    countN = 0;

    while(n > 0){
        len = 2;
        while (len > 0) {

            if (dptr == 136) {
                shake256_gpu(scA);
                dptr = 0;
            }
            __syncthreads();
            len -= 2;

            out[0] = scA[(dptr >> 3)] >> ((dptr & 7) << 3);
            dptr ++;
            out[1] = scA[(dptr >> 3)] >> ((dptr & 7) << 3);
            dptr ++;
        }
        __syncwarp();

        w = ((unsigned)out[0] << 8) | (unsigned)out[1];
        if (w < 61445) {
            while (w >= 12289) {
                w -= 12289;
            }
            x[countN] = (uint16_t)w;
            n--;
            countN++;
        }
    }
    __syncwarp();
    *scdptr = dptr;
}

__global__ void hash_to_point_vartime(uint64_t *scA, uint64_t  *scdptr, uint16_t *x)
{
	/*
	 * This is the straightforward per-the-spec implementation. It
	 * is not constant-time, thus it might reveal information on the
	 * plaintext (at least, enough to check the plaintext against a
	 * list of potential plaintexts) in a scenario where the
	 * attacker does not have access to the signature value or to
	 * the public key, but knows the nonce (without knowledge of the
	 * nonce, the hashed output cannot be matched against potential
	 * plaintexts).
	 */
	uint32_t bid = blockIdx.x; 
	uint32_t n = 0;
	while (n < Falcon_N) {
		uint8_t buf[2];
		uint32_t w;

		i_shake256_extract_gpu(scA+bid*25, scdptr, (uint8_t*) buf, 2*sizeof(uint8_t));
		w = ((unsigned)buf[0] << 8) | (unsigned)buf[1];
		if (w < 61445) {
			while (w >= 12289) {
				w -= 12289;
			}
			x[bid * Falcon_N+n] = (uint16_t)w;
			n ++;
		}
	}
}