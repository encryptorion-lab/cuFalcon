#pragma once

#include "fpr.cuh"

#ifndef BATCH
#define BATCH    2048
#endif
#define Falcon_N           512
#define LOGN        9

/*
 * Constants for NTT.
 *
 *   n = 2^logn  (2 <= n <= 1024)
 *   phi = X^n + 1
 *   q = 12289
 *   q0i = -1/q mod 2^16
 *   R = 2^16 mod q
 *   R2 = 2^32 mod q
 */

#define Q           12289
#define Q0I         12287
#define R           4091
#define R2          10952

#define CRYPTO_SECRETKEYBYTES   1281
#define CRYPTO_PUBLICKEYBYTES   897
#define CRYPTO_BYTES            690

#define NONCELEN    40
#define MLEN        33
#define ALIGN_TO_128_BYTES(x) ((((x) + 127) / 128) * 128)
//#define ALIGN_TO_128_BYTES(x) ((((x) + 255) / 256) * 256)

void crypto_sign_tree(uint8_t *h_sm, uint8_t *d_sign_mem_pool, size_t d_sign_mem_pool_pitch, cudaStream_t stream = nullptr);

void crypto_sign(uint8_t *h_sm,uint8_t *h_seed,uint8_t  *h_nonce,uint8_t  *h_esig,uint8_t  *h_sk,uint8_t  *h_m, uint8_t *d_sign_mem_pool, size_t d_sign_mem_pool_pitch, cudaStream_t stream = nullptr);

void crypto_sign_balance(fpr *d_nttbb,uint8_t *d_sign_mem_pool, size_t d_sign_mem_pool_pitch, cudaStream_t stream = nullptr);

void crypto_ver(uint8_t *h_sm, uint8_t *h_m, uint8_t *d_ver_mem_pool, size_t d_ver_mem_pool_pitch, cudaStream_t stream = nullptr);







