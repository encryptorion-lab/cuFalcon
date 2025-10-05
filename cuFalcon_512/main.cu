#include <iostream>
#include <cuda_runtime.h>
#include "include/api.cuh"
#include "include/fpr.cuh"
#include "include/test_vector.cuh"

#ifndef stream_num
#define stream_num 16
#endif


int main() {
    uint8_t *h_sm, *h_m;
    unsigned int i, j;

    cudaMallocHost((void **) &h_sm, stream_num * BATCH * (MLEN + CRYPTO_BYTES) * sizeof(uint8_t));
    cudaMallocHost((void **) &h_m, stream_num * BATCH * (MLEN + CRYPTO_BYTES) * sizeof(uint8_t));

    for (int s = 0; s < stream_num; s++) {
        for (j = 0; j < BATCH; j++) {
            for (i = 0; i < MLEN; i++) {
                h_m[j * (MLEN + CRYPTO_BYTES) + i + s * BATCH * (MLEN + CRYPTO_BYTES)] = m_tv[i];
            }
            for (i = 0; i < CRYPTO_BYTES; i++) {
                h_m[j * (MLEN + CRYPTO_BYTES) + MLEN + i + s * BATCH * (MLEN + CRYPTO_BYTES)] = 0;
            }
        }
    }

    cudaStream_t stream[stream_num];
    for (auto &s: stream) cudaStreamCreate(&s);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float total_time = 0.0f;

    cudaEvent_t startEvent_ver, stopEvent_ver;
    cudaEventCreate(&startEvent_ver);
    cudaEventCreate(&stopEvent_ver);


#ifdef TARGET_STREAM_OPT
        /* ----- sign-opt-512 ----- */
        uint8_t *h_seed, *h_nonce, *h_esig, *h_sk;
        cudaMallocHost((void**) &h_esig, stream_num * BATCH * (CRYPTO_BYTES - 2 - NONCELEN) * sizeof(uint8_t));
        cudaError_t err1 = cudaMallocHost((void**) &h_seed, stream_num * BATCH * 48 * sizeof(uint8_t));
        cudaMallocHost((void**) &h_nonce,stream_num * BATCH * NONCELEN * sizeof(uint8_t));
        cudaMallocHost((void**) &h_sk, stream_num * BATCH * CRYPTO_SECRETKEYBYTES * sizeof(uint8_t));
        if (err1 != cudaSuccess) {
            printf("Unified Memory allocation failed: %s\n", cudaGetErrorString(err1));
            return 0;
        }

        for(int s = 0 ; s < stream_num; s++) {
            for (j = 0; j < BATCH; j++)
                for (i = 0; i < Falcon_N; i++)
                    h_esig[j * (CRYPTO_BYTES - 2 - NONCELEN) + 0 + s * BATCH * (CRYPTO_BYTES - 2 - NONCELEN) ] = 0x20 + 9;

            for (j = 0; j < BATCH; j++) for (i = 0; i < 48; i++) h_seed[j * 48 + i + s * BATCH * 48] = seed_tv[i];
            for (j = 0; j < BATCH; j++) for (i = 0; i < NONCELEN; i++) h_nonce[j * NONCELEN + i + s * BATCH * NONCELEN] = nonce_tv[i];
            for (j = 0; j < BATCH; j++)
                for (i = 0; i < CRYPTO_SECRETKEYBYTES; i++)
                    h_sk[j * CRYPTO_SECRETKEYBYTES + i + s * BATCH * CRYPTO_SECRETKEYBYTES] = sk[i];
        }
        uint8_t *d_sign_mem_pool;
        size_t sign_mem_pool_pitch;
        size_t byte_size_per_sign = ALIGN_TO_128_BYTES(CRYPTO_SECRETKEYBYTES) +   // d_sk
                                    ALIGN_TO_128_BYTES(MLEN) +  // d_m
                                    ALIGN_TO_128_BYTES(CRYPTO_BYTES - 2 - NONCELEN) +  // d_esig
                                    ALIGN_TO_128_BYTES(48) +  // d_seed
                                    ALIGN_TO_128_BYTES(NONCELEN) +  // d_nonce
                                    ALIGN_TO_128_BYTES(MLEN+CRYPTO_BYTES) +  // d_sm
                                    ALIGN_TO_128_BYTES(Falcon_N) +  // d_F
                                    ALIGN_TO_128_BYTES(Falcon_N) +  // d_G
                                    ALIGN_TO_128_BYTES(Falcon_N) +  // d_f
                                    ALIGN_TO_128_BYTES(Falcon_N) +  // d_g
                                    ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) +  // d_hm
                                    ALIGN_TO_128_BYTES(sizeof(uint32_t)) +  // d_sqn
                                    ALIGN_TO_128_BYTES(sizeof(uint32_t)) +  // d_s
                                    ALIGN_TO_128_BYTES(sizeof(uint32_t)) +  // d_esiglen
                                    ALIGN_TO_128_BYTES(25 * sizeof(uint64_t)) + // d_scA
                                    ALIGN_TO_128_BYTES(sizeof(uint64_t)) +  // d_scdptr
                                    ALIGN_TO_128_BYTES(10 * Falcon_N * sizeof(fpr));  // d_bb

        size_t mem_size_per_sign = byte_size_per_sign; // align to 128 bytes
        cudaError_t err2 = cudaMallocPitch(&d_sign_mem_pool, &sign_mem_pool_pitch, mem_size_per_sign, BATCH * stream_num);

        if (err2 != cudaSuccess) {
            printf("Unified Memory allocation failed: %s\n", cudaGetErrorString(err2));
            return 0;
        }

        cudaEventRecord(startEvent);
        for(int s = 0 ; s < stream_num; s++) {
            uint8_t *stream_mem_pool = d_sign_mem_pool + s * BATCH * byte_size_per_sign;

            uint8_t *s_h_seed  = h_seed  + s * BATCH * 48 * sizeof(uint8_t);
            uint8_t *s_h_nonce = h_nonce + s * BATCH * NONCELEN * sizeof(uint8_t);
            uint8_t *s_h_esig  = h_esig  + s * BATCH * (CRYPTO_BYTES - 2 - NONCELEN) * sizeof(uint8_t);
            uint8_t *s_h_sk    = h_sk    + s * BATCH * CRYPTO_SECRETKEYBYTES * sizeof(uint8_t);
            uint8_t *s_h_m = h_m + s * BATCH * (MLEN+CRYPTO_BYTES) * sizeof(uint8_t);
            uint8_t *s_h_sm = h_sm + s * BATCH * (MLEN+CRYPTO_BYTES) * sizeof(uint8_t);

            crypto_sign(s_h_sm, s_h_seed,   s_h_nonce,  s_h_esig,  s_h_sk, s_h_m, stream_mem_pool, sign_mem_pool_pitch, stream[s]);

        }

        cudaDeviceSynchronize();
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);

        cudaEventElapsedTime(&total_time, startEvent, stopEvent);
        printf("sign_opt-512::Total execution time for all streams: %.2f ms\n", total_time);
//        printf("stream_num-BATCH :  %d-%d\n", stream_num,BATCH);

        cudaFree(d_sign_mem_pool);

#elif defined(TARGET_STREAM_TREE)
    /* ----- sign-tree-512 ----- */
    uint8_t *d_sign_mem_pool_offline;
    size_t sign_mem_pool_pitch_offline;

    size_t byte_size_per_sign_offline = ALIGN_TO_128_BYTES(CRYPTO_SECRETKEYBYTES)
                                        + ALIGN_TO_128_BYTES(MLEN)
                                        + ALIGN_TO_128_BYTES(CRYPTO_BYTES - 2 - NONCELEN)
                                        + ALIGN_TO_128_BYTES(48)  // m
                                        + ALIGN_TO_128_BYTES(NONCELEN)
                                        + ALIGN_TO_128_BYTES(MLEN + CRYPTO_BYTES)  // d_bb
                                        + ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t))
                                        + ALIGN_TO_128_BYTES(sizeof(uint32_t))
                                        + ALIGN_TO_128_BYTES(sizeof(uint32_t))
                                        + ALIGN_TO_128_BYTES(sizeof(uint32_t))
                                        + ALIGN_TO_128_BYTES(25 * sizeof(uint64_t))
                                        + ALIGN_TO_128_BYTES(sizeof(uint64_t))
                                        + ALIGN_TO_128_BYTES(15 * Falcon_N * sizeof(fpr)) // expanded_key
                                        + ALIGN_TO_128_BYTES(20 * Falcon_N * sizeof(fpr));


    size_t mem_size_per_sign = byte_size_per_sign_offline; // align to 128 bytes
    cudaMallocPitch(&d_sign_mem_pool_offline, &sign_mem_pool_pitch_offline, mem_size_per_sign, BATCH * stream_num);

    uint8_t *h_seed, *h_nonce, *h_esig, *h_sk;
    fpr *h_expanded_key;
    cudaMallocHost((void **) &h_esig, BATCH * stream_num * (CRYPTO_BYTES - 2 - NONCELEN) * sizeof(uint8_t));
    cudaMallocHost((void **) &h_seed, BATCH * stream_num * 48 * sizeof(uint8_t));
    cudaMallocHost((void **) &h_nonce, BATCH * stream_num * NONCELEN * sizeof(uint8_t));
    cudaMallocHost((void **) &h_sk, BATCH * stream_num * CRYPTO_SECRETKEYBYTES * sizeof(uint8_t));
    cudaError_t err1 = cudaMallocHost((void **) &h_expanded_key, BATCH * stream_num * 15 * Falcon_N * sizeof(fpr));

    if (err1 != cudaSuccess) {
        printf("Unified Memory allocation failed: %s\n", cudaGetErrorString(err1));
        return 0;
    }

    for (int s = 0; s < stream_num; s++) {

        uint8_t *s_h_seed = h_seed + s * BATCH * 48 * sizeof(uint8_t);
        uint8_t *s_h_nonce = h_nonce + s * BATCH * NONCELEN * sizeof(uint8_t);
        uint8_t *s_h_esig = h_esig + s * BATCH * (CRYPTO_BYTES - 2 - NONCELEN) * sizeof(uint8_t);
        uint8_t *s_h_sk = h_sk + s * BATCH * CRYPTO_SECRETKEYBYTES * sizeof(uint8_t);

        for (j = 0; j < BATCH; j++)
            for (i = 0; i < Falcon_N; i++)
                s_h_esig[j * (CRYPTO_BYTES - 2 - NONCELEN) + 0] = 0x20 + 9;
        for (j = 0; j < BATCH; j++) for (i = 0; i < 48; i++) s_h_seed[j * 48 + i] = seed_tv[i];
        for (j = 0; j < BATCH; j++) for (i = 0; i < NONCELEN; i++) s_h_nonce[j * NONCELEN + i] = nonce_tv[i];
        for (j = 0; j < BATCH; j++)
            for (i = 0; i < CRYPTO_SECRETKEYBYTES; i++)
                s_h_sk[j * CRYPTO_SECRETKEYBYTES + i] = sk[i];
        for (j = 0; j < BATCH; j++)
            for (i = 0; i < 15 * Falcon_N; i++)
                h_expanded_key[j * (15 * Falcon_N) + i].v = *(double *) &test_expanded_key[i];
    }

    // 记录起始时间
    cudaEventRecord(startEvent);
    for (int s = 0; s < stream_num; s++) {
        uint8_t *s_h_seed = h_seed + s * BATCH * 48 * sizeof(uint8_t);
        uint8_t *s_h_nonce = h_nonce + s * BATCH * NONCELEN * sizeof(uint8_t);
        uint8_t *s_h_esig = h_esig + s * BATCH * (CRYPTO_BYTES - 2 - NONCELEN) * sizeof(uint8_t);
        uint8_t *s_h_sk = h_sk + s * BATCH * CRYPTO_SECRETKEYBYTES * sizeof(uint8_t);

        uint8_t *stream_mem_pool_offline = d_sign_mem_pool_offline + s * BATCH * byte_size_per_sign_offline;

        uint8_t *d_sk = stream_mem_pool_offline;
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
        fpr *d_expanded_key = (fpr *) (d_scdptr + ALIGN_TO_128_BYTES(sizeof(uint64_t)) / sizeof(uint64_t));

        cudaMemcpy2DAsync(d_m, sign_mem_pool_pitch_offline, h_m, (MLEN + CRYPTO_BYTES) * sizeof(uint8_t),
                          MLEN * sizeof(uint8_t),
                          BATCH, cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpy2DAsync(d_esig, sign_mem_pool_pitch_offline, s_h_esig, (CRYPTO_BYTES - 2 - NONCELEN) * sizeof(uint8_t),
                          (CRYPTO_BYTES - 2 - NONCELEN) * sizeof(uint8_t), BATCH, cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpy2DAsync(d_seed, sign_mem_pool_pitch_offline, s_h_seed, 48 * sizeof(uint8_t), 48 * sizeof(uint8_t),
                          BATCH,
                          cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpy2DAsync(d_nonce, sign_mem_pool_pitch_offline, s_h_nonce, NONCELEN * sizeof(uint8_t),
                          NONCELEN * sizeof(uint8_t),
                          BATCH, cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpy2DAsync(d_sk, sign_mem_pool_pitch_offline, s_h_sk, CRYPTO_SECRETKEYBYTES * sizeof(uint8_t),
                          CRYPTO_SECRETKEYBYTES * sizeof(uint8_t), BATCH, cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpy2DAsync(d_expanded_key, sign_mem_pool_pitch_offline, h_expanded_key, 15 * Falcon_N * sizeof(fpr),
                          15 * Falcon_N * sizeof(fpr), BATCH, cudaMemcpyHostToDevice, stream[s]);

        uint8_t *s_h_sm = h_sm + s * BATCH * (MLEN + CRYPTO_BYTES) * sizeof(uint8_t);
        crypto_sign_tree(s_h_sm, stream_mem_pool_offline, sign_mem_pool_pitch_offline, stream[s]);

    }

//    cudaDeviceSynchronize();
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    cudaEventElapsedTime(&total_time, startEvent, stopEvent);
    printf("sign_tree-512::Total execution time for all streams: %.2f ms\n", total_time);
    printf("stream_num-BATCH :  %d-%d\n", stream_num, BATCH);

    cudaFree(d_sign_mem_pool_offline);
    cudaFreeHost(h_expanded_key);

#elif defined(TARGET_STREAM_BALANCE)
    /* --- sign-balance-512 --- */
    uint8_t *h_seed, *h_nonce, *h_esig, *h_sk;
    fpr *h_nttfgFG;
    cudaMallocHost((void **) &h_esig, BATCH * stream_num * (CRYPTO_BYTES - 2 - NONCELEN) * sizeof(uint8_t));
    cudaMallocHost((void **) &h_seed, BATCH * stream_num * 48 * sizeof(uint8_t));
    cudaMallocHost((void **) &h_nonce, BATCH * stream_num * NONCELEN * sizeof(uint8_t));
    cudaMallocHost((void **) &h_sk, BATCH * stream_num * CRYPTO_SECRETKEYBYTES * sizeof(uint8_t));
    cudaMallocHost((void **) &h_nttfgFG, 4 * Falcon_N * sizeof(fpr));

    for (i = 0; i < Falcon_N; i++) {
        h_nttfgFG[i] = nttg[i];
    }
    for (i = 0; i < Falcon_N; i++) {
        h_nttfgFG[i + Falcon_N] = nttf[i];
    }
    for (i = 0; i < Falcon_N; i++) {
        h_nttfgFG[i + 2 * Falcon_N] = nttG[i];
    }
    for (i = 0; i < Falcon_N; i++) {
        h_nttfgFG[i + 3 * Falcon_N] = nttF[i];
    }

    for (int s = 0; s < stream_num; s++) {
        uint8_t *s_h_seed = h_seed + s * BATCH * 48;
        uint8_t *s_h_nonce = h_nonce + s * BATCH * NONCELEN;
        uint8_t *s_h_esig = h_esig + s * BATCH * (CRYPTO_BYTES - 2 - NONCELEN);
        uint8_t *s_h_sk = h_sk + s * BATCH * CRYPTO_SECRETKEYBYTES;

        for (j = 0; j < BATCH; j++)
            s_h_esig[j * (CRYPTO_BYTES - 2 - NONCELEN) + 0] = 0x20 + 9;
        for (j = 0; j < BATCH; j++) for (i = 0; i < 48; i++) s_h_seed[j * 48 + i] = seed_tv[i];
        for (j = 0; j < BATCH; j++) for (i = 0; i < NONCELEN; i++) s_h_nonce[j * NONCELEN + i] = nonce_tv[i];
        for (j = 0; j < BATCH; j++)
            for (i = 0; i < CRYPTO_SECRETKEYBYTES; i++)
                s_h_sk[j * CRYPTO_SECRETKEYBYTES + i] = sk[i];

    }

    uint8_t *d_sign_mem_pool_server;
    size_t sign_mem_pool_pitch_server;
    size_t byte_size_per_sign_server = ALIGN_TO_128_BYTES(CRYPTO_SECRETKEYBYTES) +   // d_sk
                                       ALIGN_TO_128_BYTES(MLEN) +  // d_m
                                       ALIGN_TO_128_BYTES(CRYPTO_BYTES - 2 - NONCELEN) +  // d_esig
                                       ALIGN_TO_128_BYTES(48) +  // d_seed
                                       ALIGN_TO_128_BYTES(NONCELEN) +  // d_nonce
                                       ALIGN_TO_128_BYTES(MLEN + CRYPTO_BYTES) +  // d_sm
                                       ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) +  // d_hm
                                       ALIGN_TO_128_BYTES(sizeof(uint32_t)) +  // d_sqn
                                       ALIGN_TO_128_BYTES(sizeof(uint32_t)) +  // d_s
                                       ALIGN_TO_128_BYTES(sizeof(uint32_t)) +  // d_esiglen
                                       ALIGN_TO_128_BYTES(25 * sizeof(uint64_t)) + // d_scA
                                       ALIGN_TO_128_BYTES(sizeof(uint64_t)) +  // d_scdptr
                                       ALIGN_TO_128_BYTES(10 * Falcon_N * sizeof(fpr));  // d_bb


    size_t mem_size_per_sign_server = byte_size_per_sign_server; // align to 128 bytes
    cudaMallocPitch(&d_sign_mem_pool_server, &sign_mem_pool_pitch_server, mem_size_per_sign_server, BATCH * stream_num);

    fpr *d_nttbb;
    size_t size = ALIGN_TO_128_BYTES(4 * Falcon_N * sizeof(fpr));
    cudaMalloc((void **) &d_nttbb, size);
    cudaMemcpy(d_nttbb, h_nttfgFG, 4 * Falcon_N * sizeof(fpr), cudaMemcpyHostToDevice);

    cudaEventRecord(startEvent);
    for (int s = 0; s < stream_num; s++) {

        uint8_t *stream_mem_pool_server = d_sign_mem_pool_server + s * BATCH * byte_size_per_sign_server;

        uint8_t *d_sk = stream_mem_pool_server;
        uint8_t *d_m = d_sk + ALIGN_TO_128_BYTES(CRYPTO_SECRETKEYBYTES);
        uint8_t *d_esig = d_m + ALIGN_TO_128_BYTES(MLEN);
        uint8_t *d_seed = d_esig + ALIGN_TO_128_BYTES(CRYPTO_BYTES - 2 - NONCELEN);
        uint8_t *d_nonce = d_seed + ALIGN_TO_128_BYTES(48);  // m
        uint8_t *d_sm = d_nonce + ALIGN_TO_128_BYTES(NONCELEN);

        uint8_t *s_h_seed = h_seed + s * BATCH * 48;
        uint8_t *s_h_nonce = h_nonce + s * BATCH * NONCELEN;
        uint8_t *s_h_esig = h_esig + s * BATCH * (CRYPTO_BYTES - 2 - NONCELEN) * sizeof(uint8_t);
        uint8_t *t_h_m = h_m + s * BATCH * (MLEN + CRYPTO_BYTES);
        uint8_t *s_h_sm = h_sm + s * BATCH * (MLEN + CRYPTO_BYTES) * sizeof(uint8_t);

        cudaMemcpy2DAsync(d_m, sign_mem_pool_pitch_server, t_h_m, (MLEN + CRYPTO_BYTES) * sizeof(uint8_t),
                          MLEN * sizeof(uint8_t), BATCH, cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpy2DAsync(d_esig, sign_mem_pool_pitch_server, s_h_esig, (CRYPTO_BYTES - 2 - NONCELEN) * sizeof(uint8_t),
                          (CRYPTO_BYTES - 2 - NONCELEN) * sizeof(uint8_t), BATCH, cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpy2DAsync(d_seed, sign_mem_pool_pitch_server, s_h_seed, 48 * sizeof(uint8_t), 48 * sizeof(uint8_t),
                          BATCH, cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpy2DAsync(d_nonce, sign_mem_pool_pitch_server, s_h_nonce, NONCELEN * sizeof(uint8_t),
                          NONCELEN * sizeof(uint8_t), BATCH, cudaMemcpyHostToDevice, stream[s]);

        crypto_sign_balance(d_nttbb, stream_mem_pool_server, sign_mem_pool_pitch_server, stream[s]);

        cudaMemcpy2DAsync(s_h_sm, MLEN + CRYPTO_BYTES, d_sm, sign_mem_pool_pitch_server, MLEN + CRYPTO_BYTES, BATCH,
                          cudaMemcpyDeviceToHost, stream[s]);

    }
//    cudaDeviceSynchronize();
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    cudaEventElapsedTime(&total_time, startEvent, stopEvent);
    printf("signing_balance-512::Total execution time for all streams: %.2f ms\n", total_time);
//    printf("stream_num-BATCH :  %d-%d\n", stream_num, BATCH);

    cudaFree(d_sign_mem_pool_server);
    cudaFreeHost(h_nttfgFG);

#endif
//    // debug
//    for (j = 0; j < BATCH; j++) {
//        for (i = 0; i < (MLEN + CRYPTO_BYTES); i++) {
//            if (h_sm[j * (MLEN + CRYPTO_BYTES) + i] != test_sm[i]) {
//                printf("Wrong at batch %u-sign loc %u: %u %u\n", j, i, h_sm[j * (MLEN + CRYPTO_BYTES) + i], test_sm[i]);
//                break;
//            }
//        }
//    }

//    printf("OK!\n");

    /* ---verify-512--- */
    uint16_t *h_tmp;
    uint32_t *h_smlen, *h_msg_len;
    uint8_t *h_pk;
    uint64_t *h_test;

    cudaMallocHost((void **) &h_tmp, BATCH * stream_num * Falcon_N * sizeof(uint16_t));
    cudaMallocHost((void **) &h_pk, BATCH * stream_num * CRYPTO_PUBLICKEYBYTES * sizeof(uint8_t));
    cudaMallocHost((void **) &h_smlen, BATCH * stream_num * sizeof(uint32_t));
    cudaMallocHost((void **) &h_msg_len, BATCH * stream_num * sizeof(uint32_t));
    cudaMallocHost((void **) &h_test, BATCH * stream_num * 25 * sizeof(uint64_t));


    for (int s = 0; s < stream_num; s++) {
        uint32_t *s_h_smlen = h_smlen + s * BATCH;
        uint8_t *s_h_pk = h_pk + s * BATCH * CRYPTO_PUBLICKEYBYTES;
        uint64_t *s_h_test = h_test + s * BATCH * 25;
        for (j = 0; j < BATCH; j++)
            for (i = 0; i < CRYPTO_PUBLICKEYBYTES; i++)
                s_h_pk[j * CRYPTO_PUBLICKEYBYTES + i] = pk[i];
        for (j = 0; j < BATCH; j++) {
            s_h_smlen[j] = 691;
        }
        for (j = 0; j < BATCH; j++) for (i = 0; i < 25; i++) s_h_test[i + j * 25] = 2 * i;
    }

    uint8_t *d_ver_mem_pool;
    size_t ver_mem_pool_pitch;
    size_t byte_size_per_ver = ALIGN_TO_128_BYTES(CRYPTO_PUBLICKEYBYTES) +   // d_pk
                               ALIGN_TO_128_BYTES(MLEN + CRYPTO_BYTES) +  // d_sm
                               ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) +  // d_tmp
                               ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) +  // d_h
                               ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) +  // d_sig
                               ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) +  // d_hm
                               ALIGN_TO_128_BYTES(sizeof(uint32_t)) +  // d_s
                               ALIGN_TO_128_BYTES(sizeof(uint32_t)) +  // d_smlen
                               ALIGN_TO_128_BYTES(sizeof(uint32_t)) +  // d_msg_len
                               ALIGN_TO_128_BYTES(sizeof(uint32_t)) +  // d_sig_len
                               ALIGN_TO_128_BYTES(25 * sizeof(uint64_t)) +  // d_scA
                               ALIGN_TO_128_BYTES(sizeof(uint64_t)) +  // d_scdptr
                               ALIGN_TO_128_BYTES(25 * sizeof(uint64_t)) +  // d_test
                               ALIGN_TO_128_BYTES(25 * sizeof(uint64_t));   // d_m

    size_t mem_size_per_ver = byte_size_per_ver; // align to 128 bytes
    cudaMallocPitch(&d_ver_mem_pool, &ver_mem_pool_pitch, mem_size_per_ver, BATCH * stream_num);


    cudaEventRecord(startEvent_ver);
    for (int s = 0; s < stream_num; s++) {
        uint8_t *stream_mem_pool_verify = d_ver_mem_pool + s * BATCH * byte_size_per_ver;

        uint32_t *s_h_smlen = h_smlen + s * BATCH;
        uint8_t *s_h_pk = h_pk + s * BATCH * CRYPTO_PUBLICKEYBYTES;
        uint64_t *s_h_test = h_test + s * BATCH * 25;
        uint32_t *s_h_msg_len = h_msg_len + s * BATCH;
        uint8_t *s_h_sm = h_sm + s * BATCH * (MLEN + CRYPTO_BYTES);
        uint8_t *s_h_m = h_m + s * BATCH * (MLEN + CRYPTO_BYTES);
        uint16_t *s_h_tmp = h_tmp + s * BATCH * Falcon_N;

        uint8_t *d_pk = stream_mem_pool_verify;
        uint8_t *d_sm = d_pk + ALIGN_TO_128_BYTES(CRYPTO_PUBLICKEYBYTES);   // d_pk
        uint8_t *d_m = d_sm + ALIGN_TO_128_BYTES(MLEN + CRYPTO_BYTES);   // d_test
        uint16_t *d_tmp = (uint16_t *) (d_m + ALIGN_TO_128_BYTES(25 * sizeof(uint64_t)));  // d_sm
        uint16_t *d_h = d_tmp + ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) / sizeof(uint16_t);  // d_tmp
        uint16_t *d_hm = d_h + ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) / sizeof(uint16_t);
        int16_t *d_sig = (int16_t *) (d_hm + ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) / sizeof(uint16_t));  // d_h
        uint32_t *d_s = (uint32_t *) (d_sig + ALIGN_TO_128_BYTES(Falcon_N * sizeof(uint16_t)) / sizeof(int16_t));  // d_hm
        uint32_t *d_smlen = d_s + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t);  // d_s
        uint32_t *d_msg_len = d_smlen + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t);   // d_smlen
        uint32_t *d_sig_len = d_msg_len + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t);   // d_msg_len
        uint64_t *d_scA = (uint64_t *) (d_sig_len + ALIGN_TO_128_BYTES(sizeof(uint32_t)) / sizeof(uint32_t));   // d_sig_len
        uint64_t *d_scdptr = d_scA + ALIGN_TO_128_BYTES(25 * sizeof(uint64_t)) / sizeof(uint64_t);  // d_scA
        uint64_t *d_test = d_scdptr + ALIGN_TO_128_BYTES(sizeof(uint64_t)) / sizeof(uint64_t);   // d_scdptr

        cudaMemset2DAsync(d_scdptr, ver_mem_pool_pitch, 0, sizeof(uint64_t), BATCH, stream[s]);
        cudaMemset2DAsync(d_hm, ver_mem_pool_pitch, 0, sizeof(uint16_t), BATCH, stream[s]);
        cudaMemcpy2DAsync(d_pk, ver_mem_pool_pitch, s_h_pk, CRYPTO_PUBLICKEYBYTES * sizeof(uint8_t), CRYPTO_PUBLICKEYBYTES * sizeof(uint8_t), BATCH, cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpy2DAsync(d_sm, ver_mem_pool_pitch, s_h_sm, (MLEN + CRYPTO_BYTES) * sizeof(uint8_t), (MLEN + CRYPTO_BYTES) * sizeof(uint8_t), BATCH, cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpy2DAsync(d_test, ver_mem_pool_pitch, s_h_test, 25 * sizeof(uint64_t), 25 * sizeof(uint64_t), BATCH, cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpy2DAsync(d_smlen, ver_mem_pool_pitch, s_h_smlen, sizeof(uint32_t), sizeof(uint32_t), BATCH, cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpy2DAsync(d_msg_len, ver_mem_pool_pitch, s_h_msg_len, sizeof(uint32_t), sizeof(uint32_t), BATCH, cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpy2DAsync(d_m, ver_mem_pool_pitch, s_h_m, (MLEN + CRYPTO_BYTES) * sizeof(uint8_t), MLEN * sizeof(uint8_t), BATCH, cudaMemcpyHostToDevice, stream[s]);

        crypto_ver(d_sm, d_m, stream_mem_pool_verify, ver_mem_pool_pitch, stream[s]);

        cudaMemcpy2DAsync(s_h_tmp, Falcon_N * sizeof(uint16_t), d_tmp, ver_mem_pool_pitch, Falcon_N * sizeof(uint16_t), BATCH, cudaMemcpyDeviceToHost, stream[s]);
    }
//    cudaDeviceSynchronize();
    cudaEventRecord(stopEvent_ver);
    cudaEventSynchronize(stopEvent_ver);

    float total_time_ver = 0.0f;
    cudaEventElapsedTime(&total_time_ver, startEvent_ver, stopEvent_ver);
    printf("verify-512: Total execution time for all streams: %.2f ms\n", total_time_ver);
//    printf("  %d-%d\n", stream_num, BATCH);
//   //debug
//    for (j = 0; j < BATCH; j++) {
//        for (i = 0; i < Falcon_N; i++) {
//            if (h_tmp[j * Falcon_N + i] != test_tmp[i]) {
//                printf("Wrong at batch %u loc %u: %u %u\n", j, i, h_tmp[j * Falcon_N + i], test_tmp[i]);
//                break;
//            }
//        }
//    }

    for (auto &pUstreamSt: stream) cudaStreamDestroy(pUstreamSt);
    cudaFree(d_ver_mem_pool);
    cudaFreeHost(h_sm);
    cudaFreeHost(h_m);
    cudaFreeHost(h_seed);
    cudaFreeHost(h_nonce);
    cudaFreeHost(h_esig);
    cudaFreeHost(h_sk);

    printf("Falcon is correct!\n");
    return 0;

}
