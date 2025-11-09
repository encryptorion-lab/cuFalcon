/*
 * This file is part of cuFalcon.
 *
 * Copyright (c) 2025 Wenqian Li, et al.
 *
 * Licensed under the GNU General Public License v3.0 (GPLv3)
 * See the LICENSE file in the project root for license details.
 */

#include "../include/api.cuh"
#include "../include/sign.cuh"
#include "../include/fft.cuh"
#include "../include/ffSampling.cuh"
#include "../include/common.cuh"
#include "../include/shake.cuh"
#include "../include/expanded_key.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>


void crypto_sign_tree(uint8_t *h_sm,
                                uint8_t *d_sign_mem_pool, size_t d_sign_mem_pool_pitch, cudaStream_t stream) {

    uint8_t *d_sk = d_sign_mem_pool;
    uint8_t *d_m = d_sk + ALIGN_TO_128_BYTES(CRYPTO_SECRETKEYBYTES);
    uint8_t *d_esig = d_m + ALIGN_TO_128_BYTES(MLEN);
    uint8_t *d_seed = d_esig + ALIGN_TO_128_BYTES(CRYPTO_BYTES - 2 - NONCELEN);
    uint8_t *d_nonce = d_seed + ALIGN_TO_128_BYTES(48);  
    uint8_t *d_sm = d_nonce + ALIGN_TO_128_BYTES(NONCELEN);
    uint16_t *d_hm = (uint16_t *) (d_sm + ALIGN_TO_128_BYTES(MLEN + CRYPTO_BYTES));  
    uint32_t *d_sqn = (uint32_t *) (d_hm + ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) / sizeof(uint16_t));
    uint32_t *d_s = d_sqn + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t);
    uint32_t *d_esiglen = d_s + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t);
    uint64_t *d_scA = (uint64_t *) (d_esiglen + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t));
    uint64_t *d_scdptr = d_scA + ALIGN_TO_128_BYTES(25 * sizeof(uint64_t)) / sizeof(uint64_t);
    fpr *d_expanded_key = (fpr *) (d_scdptr + ALIGN_TO_128_BYTES(sizeof(uint64_t)) / sizeof(uint64_t));
    uint8_t *d_tmp = (uint8_t *) (d_expanded_key + ALIGN_TO_128_BYTES(15 * Falcon_N * sizeof(fpr)) / sizeof(fpr));

    i_shake256_inject_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_nonce, NONCELEN, d_sign_mem_pool_pitch);
    i_shake256_inject_gpu_kernel_2<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_m, MLEN, d_sign_mem_pool_pitch);
    i_shake256_flip_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_sign_mem_pool_pitch);
    hash_to_point_vartime_par_kernel<<<BATCH, 32, 0, stream>>>(d_scA, d_scdptr, d_hm, d_sign_mem_pool_pitch);
    i_shake256_init_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_sign_mem_pool_pitch);
    i_shake256_inject_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_seed, 48, d_sign_mem_pool_pitch);
    i_shake256_flip_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_sign_mem_pool_pitch);

    /*---------keygen---------*/
    /* d_bb0: nttg; d_bb1: nttf; d_bb2: nttG; d_bb3: nttF */
    fpr *d_b00, *d_b01, *d_b10, *d_b11, *d_tree;
    fpr *d_t0, *d_t1, *d_tx, *d_ty;
    d_t0 = (fpr *) d_tmp;
    d_t1 = d_t0 + Falcon_N;
    d_tx = d_t1 + Falcon_N;
    d_ty = d_tx + Falcon_N;

    d_b00 = d_expanded_key;
    d_b01 = d_expanded_key + Falcon_N;
    d_b10 = d_expanded_key + 2 * Falcon_N;
    d_b11 = d_expanded_key + 3 * Falcon_N;
    d_tree = d_expanded_key + 4 * Falcon_N;

    sign_tree<<<BATCH, Falcon_N / 4, 0, stream>>>(d_t0, d_t1, d_b01, d_b11, d_hm, LOGN, d_sign_mem_pool_pitch);
    ffSampling_fft_tree<<<BATCH, 32, 0, stream>>>(d_tx, d_ty, d_t0, d_t1, d_tree, d_ty + Falcon_N, LOGN, LOGN, d_scA, d_scdptr, d_sign_mem_pool_pitch);
    Get_lattice_point<<<BATCH, Falcon_N / 2, 0, stream>>>(d_t0, d_t1, d_tx, d_ty, d_b00, d_b01, d_b10, d_b11, LOGN, d_sign_mem_pool_pitch);
    ifft_t<<<BATCH, Falcon_N / 4, 0, stream>>>(d_t0, d_t1, d_sign_mem_pool_pitch);

    auto d_s1tmp = (int16_t *) d_tx;
    auto d_s2tmp = (int16_t *) d_t0;
    check1<<<BATCH, 1, 0, stream>>>(d_s1tmp, d_t0, d_hm, d_sqn, d_sign_mem_pool_pitch);
    check2<<<BATCH, Falcon_N, 0, stream>>>(d_s2tmp, d_t1, d_sign_mem_pool_pitch);
    is_short_half_gpu<<<BATCH, 1, 0, stream>>>(d_sqn, d_s2tmp, d_s, d_sign_mem_pool_pitch);
    comp_encode_gpu<<<BATCH, 32, 0, stream>>>(d_esig + 1, (CRYPTO_BYTES - 2 - NONCELEN - 1), d_s2tmp, LOGN, d_esiglen, d_sign_mem_pool_pitch);
    byte_copy<<<BATCH, MLEN, 0, stream>>>(d_sm + 2 + NONCELEN, d_m, (MLEN + CRYPTO_BYTES), MLEN, d_sign_mem_pool_pitch);
    write_smlen_gpu<<<BATCH, 1, 0, stream>>>(d_sm, d_esiglen, d_sign_mem_pool_pitch);
    byte_copy<<<BATCH, NONCELEN, 0, stream>>>(d_sm + 2, d_nonce, (MLEN + CRYPTO_BYTES), NONCELEN, d_sign_mem_pool_pitch);
    byte_copy_2<<<BATCH, 128, 0, stream>>>(d_sm + 2 + NONCELEN + MLEN, d_esig, (MLEN + CRYPTO_BYTES), d_esiglen, d_sign_mem_pool_pitch);

    cudaMemcpy2DAsync(h_sm, MLEN + CRYPTO_BYTES, d_sm, d_sign_mem_pool_pitch, MLEN + CRYPTO_BYTES, BATCH, cudaMemcpyDeviceToHost, stream);

}


void crypto_sign(uint8_t *h_sm, uint8_t *h_seed, uint8_t *h_nonce, uint8_t *h_esig, uint8_t *h_sk, uint8_t *h_m,
                        uint8_t *d_sign_mem_pool, size_t d_sign_mem_pool_pitch, cudaStream_t stream) {

    uint8_t *d_sk = d_sign_mem_pool;
    uint8_t *d_m = d_sk + ALIGN_TO_128_BYTES(CRYPTO_SECRETKEYBYTES);
    uint8_t *d_esig = d_m + ALIGN_TO_128_BYTES(MLEN);
    uint8_t *d_seed = d_esig + ALIGN_TO_128_BYTES(CRYPTO_BYTES - 2 - NONCELEN);
    uint8_t *d_nonce = d_seed + ALIGN_TO_128_BYTES(48);  // m
    uint8_t *d_sm = d_nonce + ALIGN_TO_128_BYTES(NONCELEN);
    int8_t *d_F = (int8_t *) (d_sm + ALIGN_TO_128_BYTES(MLEN + CRYPTO_BYTES));  // d_bb
    int8_t *d_G = d_F + ALIGN_TO_128_BYTES(Falcon_N);
    int8_t *d_f = d_G + ALIGN_TO_128_BYTES(Falcon_N);
    int8_t *d_g = d_f + ALIGN_TO_128_BYTES(Falcon_N);
    uint16_t *d_hm = (uint16_t *) (d_g + ALIGN_TO_128_BYTES(Falcon_N));
    uint32_t *d_sqn = (uint32_t *) (d_hm + ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) / sizeof(uint16_t));
    uint32_t *d_s = d_sqn + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t);
    uint32_t *d_esiglen = d_s + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t);
    uint64_t *d_scA = (uint64_t *) (d_esiglen + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t));
    uint64_t *d_scdptr = d_scA + ALIGN_TO_128_BYTES(25 * sizeof(uint64_t)) / sizeof(uint64_t);
    fpr *d_bb = (fpr *) (d_scdptr + ALIGN_TO_128_BYTES(sizeof(uint64_t)) / sizeof(uint64_t));

    cudaMemcpy2DAsync(d_m, d_sign_mem_pool_pitch, h_m, (MLEN + CRYPTO_BYTES) * sizeof(uint8_t), MLEN * sizeof(uint8_t), BATCH, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_esig, d_sign_mem_pool_pitch, h_esig, (CRYPTO_BYTES - 2 - NONCELEN) * sizeof(uint8_t), (CRYPTO_BYTES - 2 - NONCELEN) * sizeof(uint8_t), BATCH, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_seed, d_sign_mem_pool_pitch, h_seed, 48 * sizeof(uint8_t), 48 * sizeof(uint8_t), BATCH, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_nonce, d_sign_mem_pool_pitch, h_nonce, NONCELEN * sizeof(uint8_t), NONCELEN * sizeof(uint8_t), BATCH, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_sk, d_sign_mem_pool_pitch, h_sk, CRYPTO_SECRETKEYBYTES * sizeof(uint8_t), CRYPTO_SECRETKEYBYTES * sizeof(uint8_t), BATCH, cudaMemcpyHostToDevice, stream);

    cudaMemset2DAsync(d_G, d_sign_mem_pool_pitch, 0, sizeof(int8_t), BATCH, stream);
    cudaMemset2DAsync(d_f, d_sign_mem_pool_pitch, 0, sizeof(int8_t), BATCH, stream);
    cudaMemset2DAsync(d_g, d_sign_mem_pool_pitch, 0, sizeof(int8_t), BATCH, stream);
    cudaMemset2DAsync(d_F, d_sign_mem_pool_pitch, 0, sizeof(int8_t), BATCH, stream);

    trim_i8_decode_kernel<<<BATCH, 1, 0, stream>>>(d_f, d_g, d_F, 9, max_fg_bits[9], d_sk + 1, CRYPTO_SECRETKEYBYTES - 1, d_sign_mem_pool_pitch);
    complete_private_to_fpr_kernel<<<BATCH, 256, 0, stream>>>(d_bb, d_G, d_f, d_g, d_F, d_sign_mem_pool_pitch);
    i_shake256_inject_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_nonce, NONCELEN, d_sign_mem_pool_pitch);
    i_shake256_inject_gpu_kernel_2<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_m, MLEN, d_sign_mem_pool_pitch);
    i_shake256_flip_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_sign_mem_pool_pitch);
    hash_to_point_vartime_par_kernel<<<BATCH, 32, 0, stream>>>(d_scA, d_scdptr, d_hm, d_sign_mem_pool_pitch);
    i_shake256_init_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_sign_mem_pool_pitch);
    i_shake256_inject_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_seed, 48, d_sign_mem_pool_pitch);
    i_shake256_flip_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_sign_mem_pool_pitch);
    // Compute_B
    convert_B_fft<<<BATCH, Falcon_N / 4, 0, stream>>>(d_bb, d_bb + Falcon_N, d_bb + 2 * Falcon_N, d_bb + 3 * Falcon_N, d_sign_mem_pool_pitch);
    // Compute_G_t
    sign_dyn<<<BATCH, 128, 0, stream>>>(d_bb, d_bb + Falcon_N, d_bb + 2 * Falcon_N, d_bb + 3 * Falcon_N, d_bb + 4 * Falcon_N, d_bb + 5 * Falcon_N, d_hm, d_sign_mem_pool_pitch);
    ffSampling_fft_dyntree<<<BATCH, 32, 0, stream>>>(d_bb + 3 * Falcon_N, d_bb, d_bb + 2 * Falcon_N, LOGN, LOGN, d_scA, d_scdptr, d_sign_mem_pool_pitch);
    recompute_B_fft<<<BATCH, 128, 0, stream>>>(d_bb, d_bb + Falcon_N, d_bb + 2 * Falcon_N, d_bb + 3 * Falcon_N, d_bb + 4 * Falcon_N, d_bb + 5 * Falcon_N, d_G, d_f, d_g, d_F, d_sign_mem_pool_pitch);
    // TargetVec_s
    fft_polymul<<<BATCH, Falcon_N / 2, 0, stream>>>(d_bb, d_bb + Falcon_N, d_bb + 2 * Falcon_N, d_bb + 3 * Falcon_N, d_bb + 4 * Falcon_N, d_bb + 5 * Falcon_N, d_bb + 6 * Falcon_N, d_bb + 7 * Falcon_N, d_sign_mem_pool_pitch);
    ifft_t<<<BATCH, 128, 0, stream>>>(d_bb + 4 * Falcon_N, d_bb + 5 * Falcon_N, d_sign_mem_pool_pitch);

    auto d_s1tmp = (int16_t *) d_bb + 6 * Falcon_N;  
    auto d_s2tmp = (int16_t *) d_bb;       
    check1<<<BATCH, 1, 0, stream>>>(d_s1tmp, d_bb + 4 * Falcon_N, d_hm, d_sqn, d_sign_mem_pool_pitch);
    check2<<<BATCH, Falcon_N, 0, stream>>>(d_s2tmp, d_bb + 5 * Falcon_N, d_sign_mem_pool_pitch);
    is_short_half_gpu<<<BATCH, 1, 0, stream>>>(d_sqn, d_s2tmp, d_s, d_sign_mem_pool_pitch);
    comp_encode_gpu<<<BATCH, 32, 0, stream>>>(d_esig + 1, (CRYPTO_BYTES - 2 - NONCELEN - 1), d_s2tmp, LOGN, d_esiglen, d_sign_mem_pool_pitch);
    byte_copy<<<BATCH, MLEN, 0, stream>>>(d_sm + 2 + NONCELEN, d_m, (MLEN + CRYPTO_BYTES), MLEN, d_sign_mem_pool_pitch);
    write_smlen_gpu<<<BATCH, 1, 0, stream>>>(d_sm, d_esiglen, d_sign_mem_pool_pitch);
    byte_copy<<<BATCH, NONCELEN, 0, stream>>>(d_sm + 2, d_nonce, (MLEN + CRYPTO_BYTES), NONCELEN, d_sign_mem_pool_pitch);
    byte_copy_2<<<BATCH, 128, 0, stream>>>(d_sm + 2 + NONCELEN + MLEN, d_esig, (MLEN + CRYPTO_BYTES), d_esiglen, d_sign_mem_pool_pitch);

    cudaMemcpy2DAsync(h_sm, MLEN + CRYPTO_BYTES, d_sm, d_sign_mem_pool_pitch, MLEN + CRYPTO_BYTES, BATCH, cudaMemcpyDeviceToHost, stream);
}

void crypto_sign_balance(fpr *d_nttbb, uint8_t *d_sign_mem_pool, size_t d_sign_mem_pool_pitch, cudaStream_t stream) {

    uint8_t *d_sk = d_sign_mem_pool;
    uint8_t *d_m = d_sk + ALIGN_TO_128_BYTES(CRYPTO_SECRETKEYBYTES);
    uint8_t *d_esig = d_m + ALIGN_TO_128_BYTES(MLEN);
    uint8_t *d_seed = d_esig + ALIGN_TO_128_BYTES(CRYPTO_BYTES - 2 - NONCELEN);
    uint8_t *d_nonce = d_seed + ALIGN_TO_128_BYTES(48);  // m
    uint8_t *d_sm = d_nonce + ALIGN_TO_128_BYTES(NONCELEN);
    uint16_t *d_hm = (uint16_t *) (d_sm + ALIGN_TO_128_BYTES(MLEN + CRYPTO_BYTES));  // d_bb
    uint32_t *d_sqn = (uint32_t *) (d_hm + ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) / sizeof(uint16_t));
    uint32_t *d_s = d_sqn + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t);
    uint32_t *d_esiglen = d_s + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t);
    uint64_t *d_scA = (uint64_t *) (d_esiglen + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t));
    uint64_t *d_scdptr = d_scA + ALIGN_TO_128_BYTES(25 * sizeof(uint64_t)) / sizeof(uint64_t);
    fpr *d_bb = (fpr *) (d_scdptr + ALIGN_TO_128_BYTES(sizeof(uint64_t)) / sizeof(uint64_t));

    i_shake256_inject_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_nonce, NONCELEN, d_sign_mem_pool_pitch);
    i_shake256_inject_gpu_kernel_2<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_m, MLEN, d_sign_mem_pool_pitch);
    i_shake256_flip_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_sign_mem_pool_pitch);
    hash_to_point_vartime_par_kernel<<<BATCH, 32, 0, stream>>>(d_scA, d_scdptr, d_hm, d_sign_mem_pool_pitch);
    i_shake256_init_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_sign_mem_pool_pitch);
    i_shake256_inject_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_seed, 48, d_sign_mem_pool_pitch);
    i_shake256_flip_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_sign_mem_pool_pitch);

    sign_balance<<<BATCH, 128, 0, stream>>>(d_bb, d_nttbb, d_hm, d_sign_mem_pool_pitch);
    ffSampling_fft_dyntree<<<BATCH, 32, 0, stream>>>(d_bb + 3 * Falcon_N, d_bb, d_bb + 2 * Falcon_N, LOGN, LOGN, d_scA, d_scdptr, d_sign_mem_pool_pitch);
    arrange_kernel<<<BATCH, 128, 0, stream>>>(d_bb, d_bb + Falcon_N, d_bb + 2 * Falcon_N, d_bb + 3 * Falcon_N, d_bb + 4 * Falcon_N, d_bb + 5 * Falcon_N, d_nttbb, d_sign_mem_pool_pitch);
    fft_polymul<<<BATCH, 256, 0, stream>>>(d_bb, d_bb + Falcon_N, d_bb + 2 * Falcon_N, d_bb + 3 * Falcon_N, d_bb + 4 * Falcon_N, d_bb + 5 * Falcon_N, d_bb + 6 * Falcon_N, d_bb + 7 * Falcon_N, d_sign_mem_pool_pitch);
    ifft_t<<<BATCH, 128, 0, stream>>>(d_bb + 4 * Falcon_N, d_bb + 5 * Falcon_N, d_sign_mem_pool_pitch);

    auto d_s1tmp = (int16_t *) d_bb + 6 * Falcon_N;  
    auto d_s2tmp = (int16_t *) d_bb;    
    check1<<<BATCH, 1, 0, stream>>>(d_s1tmp, d_bb + 4 * Falcon_N, d_hm, d_sqn, d_sign_mem_pool_pitch);
    check2<<<BATCH, Falcon_N, 0, stream>>>(d_s2tmp, d_bb + 5 * Falcon_N, d_sign_mem_pool_pitch); 
    is_short_half_gpu<<<BATCH, 1, 0, stream>>>(d_sqn, d_s2tmp, d_s, d_sign_mem_pool_pitch);
    comp_encode_gpu<<<BATCH, 32, 0, stream>>>(d_esig + 1, (CRYPTO_BYTES - 2 - NONCELEN - 1), d_s2tmp, LOGN, d_esiglen, d_sign_mem_pool_pitch);
    byte_copy<<<BATCH, MLEN, 0, stream>>>(d_sm + 2 + NONCELEN, d_m, (MLEN + CRYPTO_BYTES), MLEN, d_sign_mem_pool_pitch);
    write_smlen_gpu<<<BATCH, 1, 0, stream>>>(d_sm, d_esiglen, d_sign_mem_pool_pitch);
    byte_copy<<<BATCH, NONCELEN, 0, stream>>>(d_sm + 2, d_nonce, (MLEN + CRYPTO_BYTES), NONCELEN, d_sign_mem_pool_pitch);
    byte_copy_2<<<BATCH, 128, 0, stream>>>(d_sm + 2 + NONCELEN + MLEN, d_esig, (MLEN + CRYPTO_BYTES), d_esiglen, d_sign_mem_pool_pitch);

}


void crypto_ver(uint8_t *h_sm, uint8_t *h_m,
                       uint8_t *d_ver_mem_pool, size_t d_ver_mem_pool_pitch, cudaStream_t stream) {

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float total_time = 0.0f; 

    cudaEventRecord(startEvent);

    uint8_t *d_pk = d_ver_mem_pool;
    uint8_t *d_sm = d_pk + ALIGN_TO_128_BYTES(CRYPTO_PUBLICKEYBYTES);   // d_pk
    uint8_t *d_m = d_sm + ALIGN_TO_128_BYTES(MLEN + CRYPTO_BYTES) ;   // d_test
    uint16_t *d_tmp = (uint16_t *)(d_m + ALIGN_TO_128_BYTES(25 * sizeof(uint64_t)));  // d_sm
    uint16_t *d_h = d_tmp + ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) / sizeof(uint16_t);  // d_tmp
    uint16_t *d_hm = d_h + ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) / sizeof(uint16_t);
    int16_t *d_sig = (int16_t *)(d_hm + ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) / sizeof(uint16_t));  // d_h
    uint32_t *d_s = (uint32_t *)(d_sig + ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) / sizeof(int16_t));  // d_hm
    uint32_t *d_smlen = d_s + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t);  // d_s
    uint32_t *d_msg_len = d_smlen + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t);   // d_smlen
    uint32_t *d_sig_len = d_msg_len + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t);   // d_msg_len
    uint64_t *d_scA = (uint64_t *)(d_sig_len + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t));   // d_sig_len
    uint64_t *d_scdptr = d_scA + ALIGN_TO_128_BYTES(25 * sizeof(uint64_t)) / sizeof(uint64_t);  // d_scA
    uint64_t *d_test = d_scdptr + ALIGN_TO_128_BYTES(sizeof(uint64_t)) / sizeof(uint64_t);   // d_scdptr

    modq_decode_gpu<<<BATCH, Falcon_N/4, 0, stream>>>(d_h, LOGN, d_pk+1, CRYPTO_PUBLICKEYBYTES - 1,d_ver_mem_pool_pitch);
    mq_NTT_tomonty<<<BATCH, Falcon_N/2, 0, stream>>>(d_h, LOGN, d_ver_mem_pool_pitch);
    msg_len_gpu<<<BATCH, 1, 0, stream>>>(d_sm, d_msg_len, d_smlen, d_sig_len, d_ver_mem_pool_pitch);
    comp_decode_gpu<<<BATCH, 1, 0, stream>>>(d_sig, LOGN, d_sm, d_sig_len, d_msg_len, d_ver_mem_pool_pitch);

    i_shake256_inject_gpu_kernel_2<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_sm + 2, d_msg_len, d_ver_mem_pool_pitch);
    i_shake256_flip_gpu_kernel<<<BATCH, 1, 0, stream>>>(d_scA, d_scdptr, d_ver_mem_pool_pitch);
    hash_to_point_vartime_par_kernel<<<BATCH,32, 0, stream>>>(d_scA, d_scdptr, d_hm, d_ver_mem_pool_pitch);

    comb_all_kernels<<<BATCH, Falcon_N/2, 0, stream>>>(d_tmp, d_sig, d_h, d_hm, d_ver_mem_pool_pitch);
    is_short_gpu<<<BATCH, 1, 0, stream>>>((int16_t *)d_tmp, d_sig, d_s, d_ver_mem_pool_pitch);
    byte_cmp<<<BATCH, MLEN, 0, stream>>>(d_m, d_sm + 2 + NONCELEN, d_ver_mem_pool_pitch);

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&total_time, startEvent, stopEvent);

}



