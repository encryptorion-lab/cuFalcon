#pragma once

#include "fpr.cuh"

static uint8_t max_fg_bits[] = {
        0, /* unused */
        8,
        8,
        8,
        8,
        8,
        7,
        7,
        6,
        6,
        5
};

__global__ void trim_i8_decode_kernel(int8_t *d_x, int8_t *d_y, int8_t *d_z, unsigned logn, unsigned bits, uint8_t *d_buf, size_t max_in_len, size_t d_mem_pool_pitch);

__global__ void complete_private_to_fpr_kernel(fpr *r, int8_t *G, const int8_t *f, const int8_t *g, const int8_t *F, size_t d_mem_pool_pitch);
